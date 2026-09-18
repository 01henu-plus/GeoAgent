import asyncio

from app.core.models import AgentRequest, Run
from app.demo import seed_demo


def test_api_health_and_dataset_listing(application, authenticated_client):
    seed_demo(application)
    with authenticated_client as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["service"] == "geoagent"
        datasets = client.get("/api/v1/datasets")
        assert datasets.status_code == 200
        assert len(datasets.json()) >= 3


def test_api_dataset_registration_is_idempotent(application, authenticated_client):
    seed_demo(application)
    with authenticated_client as client:
        user_id = application.store.get_user_by_username("test-user").id
        source = application.workspace.input_dir / "roads.geojson"
        target = application.workspace.for_user(user_id).input_dir / "roads.geojson"
        target.write_bytes(source.read_bytes())
        response = client.post("/api/v1/datasets", json={"path": "input/roads.geojson", "name": "roads"})
        second = client.post("/api/v1/datasets", json={"path": "input/roads.geojson", "name": "roads"})

    assert response.status_code == 200
    assert second.status_code == 200
    assert response.json()["id"] == second.json()["id"]


def test_api_exposes_trace_checkpoint_and_artifact(application, authenticated_client):
    ids = seed_demo(application)
    with authenticated_client as client:
        response = client.post("/api/v1/ask", json={"message": "检查 roads 并生成 500 米缓冲区", "dataset_ids": [ids["roads"]]})
        assert response.status_code == 200
        result = response.json()
        checkpoint = client.get(f"/api/v1/runs/{result['trace_id']}/checkpoint")
        assert checkpoint.status_code == 200
        artifacts = client.get("/api/v1/artifacts", params={"run_id": result["trace_id"]}).json()
        assert artifacts
        content = client.get(f"/api/v1/artifacts/{artifacts[0]['id']}/content")
        assert content.status_code == 200


def test_waiting_for_a_run_is_idempotent_for_assistant_messages(application):
    ids = seed_demo(application)

    async def run_case():
        request = AgentRequest(user_input="检查 roads", conversation_id="conversation-idempotent", dataset_ids=[ids["roads"]])
        run = await application.conversations.submit(request)
        first = await application.conversations.wait(run.id)
        second = await application.conversations.wait(run.id)
        return first, second

    first, second = asyncio.run(run_case())
    messages = application.store.list_messages("conversation-idempotent")

    assert first.trace_id == second.trace_id
    assert sum(message.role == "assistant" and message.run_id == first.trace_id for message in messages) == 1


def test_run_record_can_be_deleted(application, authenticated_client):
    with authenticated_client as client:
        user_id = application.store.get_user_by_username("test-user").id
        conversation = application.store.create_conversation(user_id=user_id)
        run = Run(task_id="task-delete", conversation_id=conversation.id, agent_id="main")
        application.store.save_run(run)
        response = client.delete(f"/api/v1/runs/{run.id}")
        assert response.status_code == 200
        assert response.json() == {"deleted": True}
        assert client.get(f"/api/v1/runs/{run.id}").status_code == 404


def test_run_records_can_be_deleted_in_bulk(application, authenticated_client):
    with authenticated_client as client:
        user_id = application.store.get_user_by_username("test-user").id
        conversation = application.store.create_conversation(user_id=user_id)
        runs = [Run(task_id=f"task-delete-{index}", conversation_id=conversation.id, agent_id="main") for index in range(2)]
        for run in runs:
            application.store.save_run(run)
        response = client.request("DELETE", "/api/v1/runs", json={"run_ids": [run.id for run in runs]})
        assert response.status_code == 200
        assert set(response.json()["deleted"]) == {run.id for run in runs}

import asyncio

from fastapi.testclient import TestClient

from app.api import create_app
from app.core.models import AgentRequest, Run
from app.demo import seed_demo


def test_api_health_and_dataset_listing(application):
    seed_demo(application)
    with TestClient(create_app(application)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["service"] == "geoagent"
        datasets = client.get("/api/v1/datasets")
        assert datasets.status_code == 200
        assert len(datasets.json()) >= 3


def test_api_dataset_registration_is_idempotent(application):
    ids = seed_demo(application)
    with TestClient(create_app(application)) as client:
        response = client.post("/api/v1/datasets", json={"path": "input/roads.geojson", "name": "roads"})

    assert response.status_code == 200
    assert response.json()["id"] == ids["roads"]
    assert len(application.registry.list()) == 3


def test_api_exposes_trace_checkpoint_and_artifact(application):
    ids = seed_demo(application)
    with TestClient(create_app(application)) as client:
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
        run = application.conversations.submit(request)
        first = await application.conversations.wait(run.id)
        second = await application.conversations.wait(run.id)
        return first, second

    first, second = asyncio.run(run_case())
    messages = application.store.list_messages("conversation-idempotent")

    assert first.trace_id == second.trace_id
    assert sum(message.role == "assistant" and message.run_id == first.trace_id for message in messages) == 1


def test_run_record_can_be_deleted(application):
    run = Run(task_id="task-delete", agent_id="main")
    application.store.save_run(run)

    with TestClient(create_app(application)) as client:
        response = client.delete(f"/api/v1/runs/{run.id}")
        assert response.status_code == 200
        assert response.json() == {"deleted": True}
        assert client.get(f"/api/v1/runs/{run.id}").status_code == 404


def test_run_records_can_be_deleted_in_bulk(application):
    runs = [Run(task_id=f"task-delete-{index}", agent_id="main") for index in range(2)]
    for run in runs:
        application.store.save_run(run)

    with TestClient(create_app(application)) as client:
        response = client.request("DELETE", "/api/v1/runs", json={"run_ids": [run.id for run in runs]})
        assert response.status_code == 200
        assert set(response.json()["deleted"]) == {run.id for run in runs}

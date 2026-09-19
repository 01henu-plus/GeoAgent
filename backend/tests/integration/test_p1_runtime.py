import asyncio

from app.core.models import AgentRequest, AgentResultStatus
from app.demo import seed_demo


def test_resume_skips_completed_offline_steps(application, authenticated_client):
    ids = seed_demo(application)
    with authenticated_client:
        user_id = application.store.get_user_by_username("test-user").id
    conversation = application.conversations.create("恢复测试", user_id=user_id)
    handlers = application.main_agent.runtime_action_handlers
    original_checkpoint = handlers._checkpoint_runtime
    stopped = False

    async def stop_after_inspection(request, request_frame, session, phase):
        nonlocal stopped
        await original_checkpoint(request, request_frame, session, phase)
        if not stopped and "inspect" in session.completed_steps:
            stopped = True
            raise asyncio.CancelledError()

    handlers._checkpoint_runtime = stop_after_inspection
    try:
        cancelled = asyncio.run(application.ask(AgentRequest(user_input="检查 roads 并生成 500 米缓冲区", conversation_id=conversation.id, user_id=user_id, dataset_ids=[ids["roads"]])))
    finally:
        handlers._checkpoint_runtime = original_checkpoint

    assert cancelled.status is AgentResultStatus.CANCELLED
    checkpoint = application.checkpoints.latest(cancelled.trace_id)
    assert checkpoint is not None
    assert "inspect" in checkpoint.state["completed_steps"]

    with authenticated_client as client:
        response = client.post(f"/api/v1/runs/{cancelled.trace_id}/resume")

    assert response.status_code == 200
    resumed_run_id = response.json()["run_id"]
    assert response.json()["result"]["status"] == "SUCCESS"
    resumed_events = application.store.list_events(resumed_run_id)
    assert not any(event.payload.get("tool") == "dataset.inspect" for event in resumed_events)

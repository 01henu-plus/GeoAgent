import asyncio

from fastapi.testclient import TestClient

from app.api import create_app
from app.core.models import AgentResultStatus
from app.demo import seed_demo


def test_resume_skips_completed_offline_steps(application):
    ids = seed_demo(application)
    original_checkpoint = application.main_agent._step_checkpoint
    stopped = False

    async def stop_after_inspection(*args, **kwargs):
        nonlocal stopped
        await original_checkpoint(*args, **kwargs)
        completed = args[5]
        if not stopped and "inspect" in completed:
            stopped = True
            raise asyncio.CancelledError()

    application.main_agent._step_checkpoint = stop_after_inspection
    cancelled = asyncio.run(application.ask("检查 roads 并生成 500 米缓冲区", dataset_ids=[ids["roads"]]))
    application.main_agent._step_checkpoint = original_checkpoint

    assert cancelled.status is AgentResultStatus.CANCELLED
    checkpoint = application.checkpoints.latest(cancelled.trace_id)
    assert checkpoint is not None
    assert "inspect" in checkpoint.state["completed_steps"]

    with TestClient(create_app(application)) as client:
        response = client.post(f"/api/v1/runs/{cancelled.trace_id}/resume")

    assert response.status_code == 200
    resumed_run_id = response.json()["run_id"]
    assert response.json()["result"]["status"] == "SUCCESS"
    resumed_events = application.store.list_events(resumed_run_id)
    assert not any(event.payload.get("tool") == "dataset.inspect" for event in resumed_events)

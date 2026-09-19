import asyncio

from app.core.models import AgentResultStatus
from app.demo import seed_demo


def test_offline_request_uses_agent_runtime_and_not_legacy_plan_loop(application):
    ids = seed_demo(application)
    runtime_calls = []
    original_runtime = application.main_agent.agent_runtime.run

    async def wrapped_runtime(*args, **kwargs):
        runtime_calls.append(True)
        return await original_runtime(*args, **kwargs)

    application.main_agent.agent_runtime.run = wrapped_runtime
    try:
        result = asyncio.run(application.ask("检查道路数据", dataset_ids=[ids["roads"]]))
    finally:
        application.main_agent.agent_runtime.run = original_runtime

    assert result.status is AgentResultStatus.SUCCESS
    assert runtime_calls == [True]
    assert any(event.event_type == "ToolStarted" for event in application.store.list_events(result.trace_id))


def test_offline_delegation_uses_runtime_observation_not_terminal_legacy_wrapper(application):
    ids = seed_demo(application)
    result = asyncio.run(application.ask("综合道路、人口和 DEM，从三个方面评价当前区域", dataset_ids=list(ids.values())))

    assert result.status is AgentResultStatus.SUCCESS
    assert len(result.findings) == 3
    events = application.store.list_events(result.trace_id)
    assert any(event.event_type == "DelegationCompleted" for event in events)
    assert any(event.event_type == "DecisionMade" and event.payload.get("source") == "offline" for event in events)
    checkpoint = application.checkpoints.latest(result.trace_id)
    assert checkpoint is not None
    assert "delegation_result" not in checkpoint.state


def test_offline_chat_returns_through_runtime_finalization(application):
    result = asyncio.run(application.ask("你好"))

    assert result.status is AgentResultStatus.SUCCESS
    run = application.store.get_run(result.trace_id)
    assert run is not None
    assert run.status.value == "COMPLETED"
    assert any(event.event_type == "DecisionMade" and event.payload.get("source") == "offline" for event in application.store.list_events(result.trace_id))

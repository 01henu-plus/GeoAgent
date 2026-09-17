import asyncio

from app.core.models import AgentResultStatus
from app.demo import seed_demo


def test_buffer_flow_repairs_geographic_crs(application):
    ids = seed_demo(application)
    result = asyncio.run(application.ask("检查 roads 并生成 500 米缓冲区", dataset_ids=[ids["roads"]]))
    assert result.status is AgentResultStatus.SUCCESS
    assert result.artifacts
    events = application.store.list_events(result.trace_id)
    event_types = [event.event_type for event in events]
    assert "ToolFailed" in event_types
    assert "RepairSelected" in event_types
    assert "RetryStarted" in event_types
    assert "Verification" not in "".join(event_types) or "ToolCompleted" in event_types
    assert application.store.list_lineage()


def test_multi_agent_flow_is_parallel_and_synthesized(application):
    ids = seed_demo(application)
    result = asyncio.run(application.ask("综合道路、人口和 DEM，从三个方面评价当前区域", dataset_ids=list(ids.values())))
    assert result.status is AgentResultStatus.SUCCESS
    assert len(result.findings) == 3
    events = application.store.list_events(result.trace_id)
    assert sum(event.event_type == "SubAgentSpawned" for event in events) == 3
    assert sum(event.event_type == "SubAgentCompleted" for event in events) == 3
    runs = application.store.list_runs()
    child_runs = [item for item in runs if item.parent_run_id == result.trace_id]
    assert len(child_runs) == 3
    assert all(item.started_at and item.finished_at and item.metadata.get("goal") for item in child_runs)


def test_composed_reprojection_and_buffer_flow_uses_plan_dependencies(application):
    ids = seed_demo(application)

    result = asyncio.run(application.ask("先将 roads 重投影到 EPSG:3857，再生成 500 米缓冲区", dataset_ids=[ids["roads"]]))

    assert result.status is AgentResultStatus.SUCCESS
    events = application.store.list_events(result.trace_id)
    tools = [event.payload.get("tool") for event in events if event.event_type in {"ToolStarted", "ToolCompleted"}]
    assert "crs.reproject" in tools
    assert "vector.buffer" in tools
    assert result.artifacts


def test_natural_dataset_name_is_resolved_without_frontend_preselecting_first_items(application):
    ids = seed_demo(application)
    asyncio.run(application.ask("给 roads 生成 100 米缓冲区", dataset_ids=[ids["roads"]]))

    result = asyncio.run(application.ask("给 roads 生成 500 米缓冲区"))

    assert result.status is AgentResultStatus.SUCCESS
    events = application.store.list_events(result.trace_id)
    buffer_start = next(event for event in events if event.event_type == "ToolStarted" and event.payload.get("tool") == "vector.buffer")
    assert buffer_start.payload["arguments"]["dataset_id"] == ids["roads"]

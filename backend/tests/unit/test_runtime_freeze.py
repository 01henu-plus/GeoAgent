import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.models import (
    AgentRequest,
    IntentType,
    Plan,
    PlanStep,
    Run,
    RunStatus,
    ToolResult,
    ToolStatus,
)
from app.execution.tools import RawToolExecutor
from app.runtime.agent_runtime import AgentRuntime
from app.runtime.checkpoint_codec import RuntimeCheckpointCodec
from app.runtime.session import AgentRuntimeSession


def test_runtime_freeze_removes_legacy_symbols_and_keeps_one_loop():
    app_root = Path(__file__).resolve().parents[2] / "app"

    assert not hasattr(__import__("app.core.models", fromlist=["IntentResult"]), "IntentResult")
    assert AgentRuntime.__name__ == "AgentRuntime"
    assert not (app_root / "decision" / "intent.py").exists()
    assert not (app_root / "decision" / "router.py").exists()
    assert not (app_root / "runtime" / "agent_loop.py").exists()


def test_plan_contains_only_executable_tool_steps():
    with pytest.raises(ValidationError, match="不可执行步骤"):
        Plan(
            goal="不完整计划",
            intent=IntentType.SPATIAL_ANALYSIS,
            steps=[PlanStep(id="missing-tool", title="缺少工具", action="unknown")],
        )


def test_checkpoint_codec_writes_only_canonical_fields():
    run = Run(id="run-freeze", task_id="task-freeze", agent_id="main")
    plan = Plan(goal="检查数据", intent=IntentType.DATA_INSPECTION)
    session = AgentRuntimeSession(run=run, current_plan=plan, original_plan=plan.model_copy(deep=True))
    payload = RuntimeCheckpointCodec.encode(
        AgentRequest(user_input="检查数据", conversation_id="conversation-freeze"),
        None,
        session,
    )

    assert set(RuntimeCheckpointCodec.CANONICAL_FIELDS).issubset(payload)
    assert not {
        "intent",
        "plan",
        "messages",
        "model_findings",
        "model_dataset_ids",
        "model_artifact_ids",
        "delegation_result",
        "legacy_result",
    }.intersection(payload)
    assert payload["current_plan"]["goal"] == "检查数据"


def test_legacy_checkpoint_is_read_only_compatibility():
    restored = RuntimeCheckpointCodec.decode(
        {
            "plan": Plan(goal="旧计划", intent=IntentType.DATA_INSPECTION).model_dump(mode="json"),
            "messages": [
                {"role": "system", "content": "系统"},
                {"role": "user", "content": "继续"},
                {"role": "assistant", "tool_calls": [{"id": "call-legacy", "function": {"name": "dataset.inspect", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call-legacy", "content": "{}"},
            ],
            "model_findings": ["旧观察"],
            "model_dataset_ids": ["legacy-dataset"],
            "model_artifact_ids": ["legacy-artifact"],
        }
    )

    assert restored.current_plan is not None
    assert restored.current_plan.goal == "旧计划"
    assert restored.protocol_messages[0]["role"] == "assistant"
    assert restored.findings == ["旧观察"]
    assert restored.dataset_ids == ["legacy-dataset"]
    assert restored.artifact_ids == ["legacy-artifact"]


def test_agents_have_no_raw_compatibility_methods(application):
    assert not hasattr(application.main_agent, "_tool")
    assert not hasattr(application.main_agent, "_raw_tool_for_cycle")
    assert not hasattr(application.main_agent, "_execute_tool_raw")
    assert not hasattr(application.sub_agent, "_call")
    assert not hasattr(application.sub_agent, "_raw_tool_for_cycle")
    assert not hasattr(application.sub_agent, "_execute_tool_raw")
    assert isinstance(application.main_agent.tool_execution_cycle.raw_executor.__self__, RawToolExecutor)
    assert isinstance(application.sub_agent.tool_execution_cycle.raw_executor.__self__, RawToolExecutor)


def test_raw_executor_preserves_call_attempt_and_run_tool_state(application):
    run = Run(id="run-raw-freeze", agent_id="main", status=RunStatus.RUNNING)
    application.store.save_run(run)
    captured = {}

    async def fake_execute(call, *, agent_id, services):
        captured["call"] = call
        captured["agent_id"] = agent_id
        captured["services"] = services
        return ToolResult(call_id=call.id, status=ToolStatus.SUCCESS)

    original_execute = application.main_agent.executor.execute
    application.main_agent.executor.execute = fake_execute
    try:
        result = asyncio.run(
            application.main_agent.raw_tool_executor.execute(
                run,
                "dataset.inspect",
                {"dataset_id": "dataset-freeze"},
                call_id="call-freeze",
                attempt=2,
            )
        )
    finally:
        application.main_agent.executor.execute = original_execute

    saved = application.store.get_run(run.id)
    assert result.status is ToolStatus.SUCCESS
    assert captured["call"].id == "call-freeze"
    assert captured["call"].attempt == 2
    assert captured["call"].run_id == run.id
    assert captured["call"].agent_id == "main"
    assert saved.status is RunStatus.WAITING_TOOL
    assert saved.tool_call_count == 1

import asyncio
from types import SimpleNamespace

import pytest

from app.core.models import (
    CRSInfo,
    Dataset,
    DatasetKind,
    FailureAction,
    Run,
    RunBudget,
    ToolError,
    ToolResult,
    ToolStatus,
)
from app.decision.failure_analyzer import FailureAnalyzer
from app.decision.verifier import ResultVerifier
from app.runtime.tool_execution_cycle import ToolExecutionCycle


class FakeTrace:
    def __init__(self):
        self.events = []

    async def emit(self, run_id, event_type, message, *, payload=None, agent_id=None):
        self.events.append((run_id, event_type, message, payload, agent_id))


class FakeMemoryUpdater:
    def __init__(self):
        self.accepted = []

    def update_from_tool_result(self, task_id, result, *, run_id):
        self.accepted.append((task_id, result, run_id))


class FakeRegistry:
    def __init__(self, datasets=None):
        self.datasets = datasets or []
        self.user_ids = []

    def for_user(self, user_id):
        self.user_ids.append(user_id)
        return self

    def list(self):
        return list(self.datasets)

    def resolve(self, identifier):
        return next((item for item in self.datasets if item.id == identifier), None)


class FakeToolRegistry:
    def __init__(self, produces_dataset=False):
        self.produces_dataset = produces_dataset

    def get(self, name):
        return SimpleNamespace(metadata=SimpleNamespace(produces_dataset=self.produces_dataset))


class SequenceExecutor:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def __call__(self, run, name, arguments, *, call_id=None):
        self.calls.append((name, arguments, call_id))
        result = self.results.pop(0)
        return result


class FixedFailureAnalyzer(FailureAnalyzer):
    def __init__(self, action):
        self.action = action

    def analyze(self, result):
        return self.action, f"测试恢复动作：{self.action.value}"


def _run():
    return Run(task_id="task-1", conversation_id="conversation-1", agent_id="main")


def _cycle(executor, *, verifier=None, failure_analyzer=None, produces_dataset=False, budget=None, registry=None, updater=None):
    return ToolExecutionCycle(
        raw_executor=executor,
        tool_registry=FakeToolRegistry(produces_dataset=produces_dataset),
        registry=registry or FakeRegistry(),
        trace=FakeTrace(),
        failure_analyzer=failure_analyzer or FailureAnalyzer(),
        verifier=verifier or ResultVerifier(),
        budget=budget or RunBudget(max_retry_per_action=1),
        working_memory_updater=updater or FakeMemoryUpdater(),
    )


def test_success_and_verification_update_working_memory():
    executor = SequenceExecutor([ToolResult(call_id="call", status=ToolStatus.SUCCESS, datasets=["out"])])
    updater = FakeMemoryUpdater()
    verifier = SimpleNamespace(verify=lambda result, datasets: (True, []))
    cycle = _cycle(executor, produces_dataset=True, verifier=verifier, updater=updater)

    outcome = asyncio.run(cycle.execute(_run(), "raster.slope", {"dataset_id": "dem"}, user_id="user-a"))

    assert outcome.accepted is True
    assert outcome.verified is True
    assert outcome.attempts == 1
    assert updater.accepted[0][1].datasets == ["out"]


def test_verification_failure_does_not_update_working_memory():
    executor = SequenceExecutor([ToolResult(call_id="call", status=ToolStatus.SUCCESS, datasets=["bad"])])
    updater = FakeMemoryUpdater()
    verifier = SimpleNamespace(verify=lambda result, datasets: (False, ["结果不可读"]))
    cycle = _cycle(executor, produces_dataset=True, verifier=verifier, updater=updater)

    outcome = asyncio.run(cycle.execute(_run(), "raster.slope", {"dataset_id": "dem"}, user_id="user-a"))

    assert outcome.accepted is False
    assert outcome.verification_problems == ["结果不可读"]
    assert updater.accepted == []


def test_retry_success_accepts_only_final_result():
    executor = SequenceExecutor(
        [
            ToolResult(call_id="first", status=ToolStatus.FAILED, retryable=True, error=ToolError(code="EXECUTION_TIMEOUT", message="超时")),
            ToolResult(call_id="second", status=ToolStatus.SUCCESS, output={"ok": True}),
        ]
    )
    updater = FakeMemoryUpdater()
    cycle = _cycle(executor, updater=updater)

    outcome = asyncio.run(cycle.execute(_run(), "dataset.inspect", {}, user_id="user-a"))

    assert outcome.accepted is True
    assert outcome.attempts == 2
    assert outcome.original_result.call_id == "first"
    assert updater.accepted[0][1].call_id == "second"


def test_retry_stops_at_budget():
    executor = SequenceExecutor(
        [
            ToolResult(call_id="first", status=ToolStatus.FAILED, retryable=True, error=ToolError(code="EXECUTION_TIMEOUT", message="超时")),
            ToolResult(call_id="second", status=ToolStatus.FAILED, retryable=True, error=ToolError(code="EXECUTION_TIMEOUT", message="仍然超时")),
        ]
    )
    cycle = _cycle(executor, budget=RunBudget(max_retry_per_action=1))

    outcome = asyncio.run(cycle.execute(_run(), "dataset.inspect", {}, user_id="user-a"))

    assert outcome.accepted is False
    assert outcome.attempts == 2
    assert len(executor.calls) == 2


@pytest.mark.parametrize("action", [FailureAction.ASK_USER, FailureAction.REPLAN, FailureAction.ABORT])
def test_non_automatic_recovery_actions_return_to_caller(action):
    executor = SequenceExecutor([ToolResult(call_id="call", status=ToolStatus.FAILED, error=ToolError(code="INPUT", message="无法继续"))])
    cycle = _cycle(executor, failure_analyzer=FixedFailureAnalyzer(action))

    outcome = asyncio.run(cycle.execute(_run(), "dataset.inspect", {}, user_id="user-a"))

    assert outcome.accepted is False
    assert outcome.recovery_action is action
    assert len(executor.calls) == 1


def test_repair_success_retries_original_tool_without_accepting_repair_dataset():
    executor = SequenceExecutor(
        [
            ToolResult(call_id="first", status=ToolStatus.FAILED, error=ToolError(code="CRS_MISMATCH", message="坐标系不一致")),
            ToolResult(call_id="final", status=ToolStatus.SUCCESS, datasets=["slope"]),
        ]
    )

    class RepairCycle(ToolExecutionCycle):
        async def _repair_arguments(self, run, tool_name, arguments, error_code, *, user_id):
            return {**arguments, "dataset_id": "projected"}

    updater = FakeMemoryUpdater()
    cycle = RepairCycle(
        raw_executor=executor,
        tool_registry=FakeToolRegistry(produces_dataset=True),
        registry=FakeRegistry(),
        trace=FakeTrace(),
        failure_analyzer=FixedFailureAnalyzer(FailureAction.REPAIR),
        verifier=SimpleNamespace(verify=lambda result, datasets: (True, [])),
        budget=RunBudget(max_retry_per_action=1),
        working_memory_updater=updater,
    )

    outcome = asyncio.run(cycle.execute(_run(), "raster.slope", {"dataset_id": "dem"}, user_id="user-a"))

    assert outcome.accepted is True
    assert outcome.attempts == 2
    assert [call[0] for call in executor.calls] == ["raster.slope", "raster.slope"]
    assert updater.accepted[0][1].datasets == ["slope"]


def test_repair_failure_does_not_loop():
    executor = SequenceExecutor([ToolResult(call_id="first", status=ToolStatus.FAILED, error=ToolError(code="CRS_MISMATCH", message="坐标系不一致"))])

    class RepairCycle(ToolExecutionCycle):
        async def _repair_arguments(self, run, tool_name, arguments, error_code, *, user_id):
            return None

    cycle = RepairCycle(
        raw_executor=executor,
        tool_registry=FakeToolRegistry(),
        registry=FakeRegistry(),
        trace=FakeTrace(),
        failure_analyzer=FixedFailureAnalyzer(FailureAction.REPAIR),
        verifier=ResultVerifier(),
        budget=RunBudget(max_retry_per_action=1),
        working_memory_updater=FakeMemoryUpdater(),
    )

    outcome = asyncio.run(cycle.execute(_run(), "raster.slope", {}, user_id="user-a"))

    assert outcome.accepted is False
    assert outcome.attempts == 1
    assert len(executor.calls) == 1


def test_crs_repair_tool_is_internal_and_only_final_result_is_accepted():
    left = Dataset(id="left", name="left", kind=DatasetKind.VECTOR, path="left.geojson", format="geojson", crs=CRSInfo(authority="EPSG:4326"))
    right = Dataset(id="right", name="right", kind=DatasetKind.VECTOR, path="right.geojson", format="geojson", crs=CRSInfo(authority="EPSG:3857"))
    executor = SequenceExecutor(
        [
            ToolResult(call_id="first", status=ToolStatus.FAILED, error=ToolError(code="CRS_MISMATCH", message="坐标系不一致")),
            ToolResult(call_id="repair", status=ToolStatus.SUCCESS, datasets=["projected-right"]),
            ToolResult(call_id="final", status=ToolStatus.SUCCESS, datasets=["intersection"]),
        ]
    )
    updater = FakeMemoryUpdater()
    cycle = _cycle(executor, registry=FakeRegistry([left, right]), updater=updater)

    outcome = asyncio.run(
        cycle.execute(
            _run(),
            "vector.intersection",
            {"left_dataset_id": "left", "right_dataset_id": "right"},
            user_id="user-a",
        )
    )

    assert outcome.accepted is True
    assert [call[0] for call in executor.calls] == ["vector.intersection", "crs.reproject", "vector.intersection"]
    assert updater.accepted[0][1].datasets == ["intersection"]


def test_cycle_uses_user_scoped_registry_for_verification():
    executor = SequenceExecutor([ToolResult(call_id="call", status=ToolStatus.SUCCESS, datasets=["out"])])
    verifier = SimpleNamespace(verify=lambda result, datasets: (True, []))
    registry = FakeRegistry()
    cycle = _cycle(executor, produces_dataset=True, verifier=verifier, registry=registry)

    asyncio.run(cycle.execute(_run(), "raster.slope", {}, user_id="user-a"))

    assert registry.user_ids == ["user-a"]

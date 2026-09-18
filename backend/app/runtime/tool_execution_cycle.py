"""统一 Tool 执行语义：Execute -> Observe -> Verify -> Recover -> Accept。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.core.models import FailureAction, LoopDirective, Run, RunBudget, ToolResult, ToolStatus
from app.decision.failure_analyzer import FailureAnalyzer
from app.decision.verifier import ResultVerifier
from app.events import EventType
from app.gis.crs.service import CRSService
from app.state.working_memory import WorkingMemoryUpdater

RawToolExecutor = Callable[..., Awaitable[ToolResult]]


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    """Raw ToolResult 经过恢复和验证后的执行结论。"""

    result: ToolResult
    verified: bool
    verification_problems: list[str]
    recovery_action: FailureAction | None
    attempts: int
    accepted: bool
    directive: LoopDirective = LoopDirective.CONTINUE
    original_result: ToolResult | None = None
    rationale: str | None = None


class ToolExecutionCycle:
    """统一 Planner 和 Model 已选定 Tool 后的执行、恢复、验证和状态接收。"""

    def __init__(
        self,
        *,
        raw_executor: RawToolExecutor,
        tool_registry,
        registry,
        trace,
        failure_analyzer: FailureAnalyzer,
        verifier: ResultVerifier,
        budget: RunBudget,
        working_memory_updater: WorkingMemoryUpdater,
        default_crs: str = "EPSG:3857",
    ) -> None:
        self.raw_executor = raw_executor
        self.tool_registry = tool_registry
        self.registry = registry
        self.trace = trace
        self.failure_analyzer = failure_analyzer
        self.verifier = verifier
        self.budget = budget
        self.working_memory_updater = working_memory_updater
        self.default_crs = default_crs

    async def execute(
        self,
        run: Run,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        user_id: str | None = None,
        call_id: str | None = None,
    ) -> ExecutionOutcome:
        original = await self.raw_executor(run, tool_name, arguments, call_id=call_id)
        current = original
        attempts = 1
        recovery_action: FailureAction | None = None
        rationale: str | None = None
        retry_count = 0
        repair_used = False

        while current.status is ToolStatus.FAILED and current.error is not None:
            action, rationale = self.failure_analyzer.analyze(current)
            recovery_action = action
            await self._trace_recovery(run, tool_name, current, action, rationale)

            if action is FailureAction.RETRY and current.retryable and retry_count < self.budget.max_retry_per_action:
                retry_count += 1
                await self.trace.emit(
                    run.id,
                    EventType.RETRY_STARTED,
                    f"重试 {tool_name}",
                    payload={"tool": tool_name, "attempt": attempts + 1},
                    agent_id=run.agent_id,
                )
                current = await self.raw_executor(run, tool_name, arguments)
                attempts += 1
                continue

            if action is FailureAction.REPAIR and not repair_used:
                repair_used = True
                repaired_arguments = await self._repair_arguments(
                    run,
                    tool_name,
                    arguments,
                    current.error.code,
                    user_id=user_id,
                )
                if repaired_arguments is None:
                    break
                await self.trace.emit(
                    run.id,
                    EventType.RETRY_STARTED,
                    f"修复输入后重试 {tool_name}",
                    payload={"tool": tool_name, "arguments": repaired_arguments},
                    agent_id=run.agent_id,
                )
                current = await self.raw_executor(run, tool_name, repaired_arguments)
                attempts += 1
                arguments = repaired_arguments
                continue

            break

        if current.status not in {ToolStatus.SUCCESS, ToolStatus.PARTIAL_SUCCESS}:
            directive = _directive_for_failure(recovery_action)
            return ExecutionOutcome(
                result=current,
                verified=False,
                verification_problems=[],
                recovery_action=recovery_action,
                attempts=attempts,
                accepted=False,
                directive=directive,
                original_result=original,
                rationale=rationale,
            )

        verified = True
        verification_problems: list[str] = []
        if self._produces_dataset(tool_name) and current.datasets:
            await self.trace.emit(
                run.id,
                EventType.VERIFICATION_STARTED,
                f"开始验证 {tool_name} 的输出",
                payload={"tool": tool_name, "dataset_ids": current.datasets},
                agent_id=run.agent_id,
            )
            visible_datasets = {item.id: item for item in self.registry.for_user(user_id).list()}
            verified, verification_problems = self.verifier.verify(current, visible_datasets)
            if not verified:
                recovery_action = FailureAction.ABORT
                rationale = "Tool 输出未通过 Dataset 验证，未自动接受该结果。"
                await self.trace.emit(
                    run.id,
                    EventType.VERIFICATION_FAILED,
                    "；".join(verification_problems),
                    payload={"tool": tool_name, "problems": verification_problems},
                    agent_id=run.agent_id,
                )

        accepted = verified
        if accepted:
            self.working_memory_updater.update_from_tool_result(run.task_id, current, run_id=run.id)

        return ExecutionOutcome(
            result=current,
            verified=verified,
            verification_problems=verification_problems,
            recovery_action=recovery_action,
            attempts=attempts,
            accepted=accepted,
            directive=LoopDirective.CONTINUE if accepted else LoopDirective.ABORT,
            original_result=original,
            rationale=rationale,
        )

    async def _trace_recovery(self, run: Run, tool_name: str, result: ToolResult, action: FailureAction, rationale: str) -> None:
        event_type = {
            FailureAction.REPAIR: EventType.REPAIR_SELECTED,
            FailureAction.REPLAN: EventType.REPLAN_STARTED,
            FailureAction.ASK_USER: EventType.DECISION_MADE,
            FailureAction.ABORT: EventType.DECISION_MADE,
        }.get(action, EventType.DECISION_MADE)
        await self.trace.emit(
            run.id,
            event_type,
            rationale,
            payload={"action": action.value, "tool": tool_name, "error": result.error.model_dump(mode="json") if result.error else None},
            agent_id=run.agent_id,
        )

    def _produces_dataset(self, tool_name: str) -> bool:
        try:
            return bool(self.tool_registry.get(tool_name).metadata.produces_dataset)
        except KeyError:
            return False

    async def _repair_arguments(
        self,
        run: Run,
        tool_name: str,
        arguments: dict[str, Any],
        error_code: str,
        *,
        user_id: str | None,
    ) -> dict[str, Any] | None:
        registry = self.registry.for_user(user_id)
        if error_code in {"CRS_UNIT_MISMATCH", "CRS_MISSING"} and "dataset_id" in arguments:
            dataset = registry.resolve(str(arguments["dataset_id"]))
            if dataset is None or dataset.crs is None:
                return None
            target_crs = CRSService(default_crs=self.default_crs).choose_projected_crs(dataset)
            reprojection_tool = "raster.reproject" if dataset.kind.value == "RASTER" else "crs.reproject"
            repaired = await self.raw_executor(run, reprojection_tool, {"dataset_id": dataset.id, "target_crs": target_crs})
            if repaired.status is not ToolStatus.SUCCESS or not repaired.datasets:
                return None
            updated = dict(arguments)
            updated["dataset_id"] = repaired.datasets[-1]
            return updated

        if error_code == "CRS_UNIT_MISMATCH" and "source_dataset_id" in arguments:
            source = registry.resolve(str(arguments["source_dataset_id"]))
            if source is None or source.crs is None:
                return None
            target_crs = CRSService(default_crs=self.default_crs).choose_projected_crs(source)
            updated = dict(arguments)
            for key in ("source_dataset_id", "target_dataset_id"):
                identifier = updated.get(key)
                dataset = registry.resolve(str(identifier)) if identifier else None
                if dataset is None:
                    continue
                reprojection_tool = "raster.reproject" if dataset.kind.value == "RASTER" else "crs.reproject"
                repaired = await self.raw_executor(run, reprojection_tool, {"dataset_id": dataset.id, "target_crs": target_crs})
                if repaired.status is not ToolStatus.SUCCESS or not repaired.datasets:
                    return None
                updated[key] = repaired.datasets[-1]
            return updated

        if error_code == "CRS_MISMATCH":
            left_id = arguments.get("left_dataset_id") or arguments.get("source_dataset_id")
            right_key = "right_dataset_id" if arguments.get("right_dataset_id") else "mask_dataset_id" if arguments.get("mask_dataset_id") else "target_dataset_id"
            right_id = arguments.get(right_key)
            left = registry.resolve(str(left_id)) if left_id else None
            right = registry.resolve(str(right_id)) if right_id else None
            if left is None or right is None or left.crs is None or not left.crs.authority:
                return None
            updated = dict(arguments)
            reprojection_tool = "raster.reproject" if right.kind.value == "RASTER" else "crs.reproject"
            repaired = await self.raw_executor(run, reprojection_tool, {"dataset_id": right.id, "target_crs": left.crs.authority})
            if repaired.status is not ToolStatus.SUCCESS or not repaired.datasets:
                return None
            updated[right_key] = repaired.datasets[-1]
            return updated

        if error_code == "INVALID_GEOMETRY":
            updated = dict(arguments)
            input_keys = [key for key in ("dataset_id", "left_dataset_id", "right_dataset_id", "mask_dataset_id") if key in updated]
            changed = False
            for key in input_keys:
                dataset = registry.resolve(str(updated[key]))
                if dataset is None or dataset.kind.value != "VECTOR":
                    continue
                repaired = await self.raw_executor(run, "vector.repair", {"dataset_id": dataset.id})
                if repaired.status is ToolStatus.SUCCESS and repaired.datasets:
                    updated[key] = repaired.datasets[-1]
                    changed = True
            return updated if changed else None
        return None


__all__ = ["ExecutionOutcome", "ToolExecutionCycle"]


def _directive_for_failure(action: FailureAction | None) -> LoopDirective:
    """把最终失败动作映射为上层循环信号。

    RETRY/REPAIR 表示执行过程中发生过恢复尝试；当尝试已经用尽时，
    它们不再是下一步控制信号，默认安全终止当前步骤。
    """

    return {
        FailureAction.ASK_USER: LoopDirective.ASK_USER,
        FailureAction.REPLAN: LoopDirective.REPLAN,
        FailureAction.ABORT: LoopDirective.ABORT,
    }.get(action, LoopDirective.ABORT)

"""验证 RequestFrame 是否符合当前状态事实。"""

from __future__ import annotations

from app.core.models import (
    InteractionMode,
    RequestFrame,
    RequestResolutionStatus,
    ResolvedReference,
    RunStatus,
    StateSnapshot,
)


class RequestFrameValidator:
    def validate(self, frame: RequestFrame, state: StateSnapshot) -> RequestFrame:
        unresolved = list(frame.unresolved_references)
        blocking: list[str] = list(frame.blocking_issues)
        valid_references: list[ResolvedReference] = []
        known = {
            "task": set(state.known_task_ids),
            "run": set(state.known_run_ids),
            "artifact": set(state.known_artifact_ids),
            "dataset": set(state.known_dataset_ids),
        }
        for reference in frame.references:
            if reference.target_id and reference.target_id in known.get(reference.type, set()):
                valid_references.append(reference)
            else:
                unresolved.append(reference.mention or reference.target_id or reference.type)

        target_task_id = None if frame.mode in {InteractionMode.NEW_TASK, InteractionMode.CHAT} else self._valid_target(frame.target_task_id, "task", known, unresolved)
        target_run_id = None if frame.mode in {InteractionMode.NEW_TASK, InteractionMode.CHAT} else self._valid_target(frame.target_run_id, "run", known, unresolved)
        confidence = frame.confidence

        if frame.mode in {InteractionMode.NEW_TASK, InteractionMode.CHAT}:
            target_task_id = None
            target_run_id = None

        if frame.mode in {InteractionMode.CONTINUE_TASK, InteractionMode.MODIFY_TASK}:
            if target_task_id is None and state.active_task_id:
                target_task_id = state.active_task_id
            if target_task_id is None:
                unresolved.append("当前任务")
                blocking.append("当前会话没有可继续或修改的任务")

        if frame.mode is InteractionMode.RETRY_TASK:
            failed_run = _find_failed_run(target_run_id, state)
            if failed_run is None:
                target_run_id = None
                unresolved.append("可重试的失败运行")
                blocking.append("当前会话没有可重试的失败运行")
            else:
                target_task_id = failed_run.task_id
                if target_task_id is None or target_task_id not in known["task"]:
                    blocking.append("失败运行没有关联当前会话任务")

        if frame.mode is InteractionMode.CANCEL_TASK:
            if target_task_id is None and state.active_task_id:
                target_task_id = state.active_task_id
            if target_task_id is None:
                unresolved.append("当前任务")
                blocking.append("当前会话没有可取消的任务")
            if _find_active_run(target_run_id, state) is None:
                target_run_id = None
                unresolved.append("当前运行")
                blocking.append("当前会话没有可取消的运行")

        unresolved = list(dict.fromkeys(item for item in unresolved if item))
        if unresolved and frame.mode is not InteractionMode.CHAT:
            blocking.extend(f"无法解析引用：{item}" for item in unresolved)
        blocking = list(dict.fromkeys(blocking))
        if unresolved:
            confidence = min(confidence, 0.6)
        if blocking:
            confidence = min(confidence, 0.35)

        return frame.model_copy(
            update={
                "references": _dedupe(valid_references),
                "target_task_id": target_task_id,
                "target_run_id": target_run_id,
                "unresolved_references": list(dict.fromkeys(unresolved)),
                "confidence": confidence,
                "needs_planning": frame.needs_planning and not blocking,
                "needs_tool": frame.needs_tool and not blocking,
                "resolution_status": RequestResolutionStatus.NEEDS_CLARIFICATION if blocking else RequestResolutionStatus.RESOLVED,
                "blocking_issues": blocking,
            }
        )

    @staticmethod
    def _valid_target(target_id: str | None, target_type: str, known: dict[str, set[str]], unresolved: list[str]) -> str | None:
        if target_id is None:
            return None
        if target_id in known[target_type]:
            return target_id
        unresolved.append(f"{target_type}:{target_id}")
        return None


def _find_failed_run(run_id: str | None, state: StateSnapshot):
    candidates = state.recent_runs if run_id is None else [item for item in state.recent_runs if item.id == run_id]
    return next((item for item in candidates if item.status is RunStatus.FAILED or (item.error and item.status not in {RunStatus.WAITING_USER, RunStatus.WAITING_APPROVAL, RunStatus.CANCELLED})), None)


def _find_active_run(run_id: str | None, state: StateSnapshot):
    candidates = state.recent_runs if run_id is None else [item for item in state.recent_runs if item.id == run_id]
    active = {
        RunStatus.CREATED,
        RunStatus.PLANNING,
        RunStatus.RUNNING,
        RunStatus.WAITING_TOOL,
        RunStatus.WAITING_SUBAGENT,
        RunStatus.WAITING_USER,
        RunStatus.WAITING_APPROVAL,
        RunStatus.RETRYING,
        RunStatus.REPLANNING,
        RunStatus.VALIDATING,
    }
    return next((item for item in candidates if item.status in active), None)


def _dedupe(references: list[ResolvedReference]) -> list[ResolvedReference]:
    seen: set[tuple[str, str | None]] = set()
    result: list[ResolvedReference] = []
    for item in references:
        key = (item.type, item.target_id)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result

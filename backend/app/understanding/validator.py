"""验证 RequestFrame 是否符合当前状态事实。"""

from __future__ import annotations

from app.core.models import InteractionMode, RequestFrame, ResolvedReference, StateSnapshot


class RequestFrameValidator:
    def validate(self, frame: RequestFrame, state: StateSnapshot) -> RequestFrame:
        unresolved = list(frame.unresolved_references)
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

        target_task_id = self._valid_target(frame.target_task_id, "task", known, unresolved)
        target_run_id = self._valid_target(frame.target_run_id, "run", known, unresolved)
        confidence = frame.confidence

        if frame.mode in {InteractionMode.CONTINUE_TASK, InteractionMode.MODIFY_TASK, InteractionMode.CANCEL_TASK}:
            if target_task_id is None and state.active_task_id:
                target_task_id = state.active_task_id
            if target_task_id is None:
                unresolved.append("当前任务")
                confidence = min(confidence, 0.35)

        if frame.mode is InteractionMode.RETRY_TASK:
            if target_run_id is None:
                target_run_id = _failed_run_id(state)
            if target_run_id is None:
                unresolved.append("可重试的失败运行")
                confidence = min(confidence, 0.35)

        if frame.mode is InteractionMode.CANCEL_TASK and state.active_task_id is None:
            unresolved.append("当前任务")
            confidence = min(confidence, 0.35)

        if unresolved:
            confidence = min(confidence, 0.6)

        return frame.model_copy(
            update={
                "references": _dedupe(valid_references),
                "target_task_id": target_task_id,
                "target_run_id": target_run_id,
                "unresolved_references": list(dict.fromkeys(unresolved)),
                "confidence": confidence,
                "needs_tool": frame.needs_tool and not unresolved,
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


def _failed_run_id(state: StateSnapshot) -> str | None:
    for run in state.recent_runs:
        if run.status.value == "FAILED" or run.error:
            return run.id
    return None


def _dedupe(references: list[ResolvedReference]) -> list[ResolvedReference]:
    seen: set[tuple[str, str | None]] = set()
    result: list[ResolvedReference] = []
    for item in references:
        key = (item.type, item.target_id)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


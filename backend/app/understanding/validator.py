"""验证 RequestFrame 是否符合当前状态事实。"""

from __future__ import annotations

from app.core.models import (
    InteractionMode,
    RequestFrame,
    RequestResolutionStatus,
    RequestResources,
    ResolvedReference,
    StateSnapshot,
)
from app.run.predicates import is_active_run, is_retryable_failed_run

_ALLOWED_PARAMETERS = {"distance", "target_crs", "field", "predicate"}
_ALLOWED_PREDICATES = {"intersects", "within", "nearest"}


class RequestFrameValidator:
    def validate(self, frame: RequestFrame, state: StateSnapshot, resources: RequestResources | None = None) -> RequestFrame:
        unresolved = list(frame.unresolved_references)
        blocking: list[str] = list(frame.blocking_issues)
        valid_references: list[ResolvedReference] = []
        known = {
            "task": set(state.known_task_ids),
            "run": set(state.known_run_ids) | ({item.id for item in resources.runs} if resources else set()),
            "artifact": set(state.known_artifact_ids),
            "dataset": set(state.known_dataset_ids) | ({item.id for item in resources.datasets} if resources else set()),
        }
        for reference in frame.references:
            if reference.target_id and reference.target_id in known.get(reference.type, set()):
                valid_references.append(reference)
            else:
                unresolved.append(reference.mention or reference.target_id or reference.type)

        target_task_id = None if frame.mode in {InteractionMode.NEW_TASK, InteractionMode.CHAT} else self._valid_target(frame.target_task_id, "task", known, unresolved)
        target_run_id = None if frame.mode in {InteractionMode.NEW_TASK, InteractionMode.CHAT} else self._valid_target(frame.target_run_id, "run", known, unresolved)
        confidence = frame.confidence
        operations = list(dict.fromkeys(str(item) for item in frame.operations if str(item).strip()))
        dataset_roles = list(dict.fromkeys(str(item) for item in frame.dataset_roles if str(item).strip()))
        parameters, parameter_issues = _validate_parameters(frame.parameters)
        blocking.extend(parameter_issues)

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
                "operations": operations,
                "parameters": parameters,
                "dataset_roles": dataset_roles,
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
    return next((item for item in candidates if is_retryable_failed_run(item)), None)


def _find_active_run(run_id: str | None, state: StateSnapshot):
    candidates = state.recent_runs if run_id is None else [item for item in state.recent_runs if item.id == run_id]
    return next((item for item in candidates if is_active_run(item)), None)


def _dedupe(references: list[ResolvedReference]) -> list[ResolvedReference]:
    seen: set[tuple[str, str | None]] = set()
    result: list[ResolvedReference] = []
    for item in references:
        key = (item.type, item.target_id)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _validate_parameters(values: dict[str, object]) -> tuple[dict[str, object], list[str]]:
    valid: dict[str, object] = {}
    issues: list[str] = []
    for key, value in values.items():
        if key not in _ALLOWED_PARAMETERS:
            issues.append(f"不支持的请求参数：{key}")
            continue
        if key == "distance":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                issues.append("distance 必须是大于 0 的数字")
                continue
        elif key in {"target_crs", "field"}:
            if not isinstance(value, str) or not value.strip():
                issues.append(f"{key} 必须是非空文本")
                continue
            value = value.strip()
        elif key == "predicate" and value not in _ALLOWED_PREDICATES:
            issues.append("predicate 只能是 intersects、within 或 nearest")
            continue
        valid[key] = value
    return valid, issues

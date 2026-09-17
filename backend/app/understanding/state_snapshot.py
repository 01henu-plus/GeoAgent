"""从 StateStore 构建请求理解所需的最小状态视图。"""

from __future__ import annotations

from app.core.models import RunStatus, StateSnapshot, TaskStatus
from app.state import StateStore

_ACTIVE_RUN_STATUSES = {
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
_ACTIVE_TASK_STATUSES = {
    TaskStatus.PENDING,
    TaskStatus.READY,
    TaskStatus.RUNNING,
    TaskStatus.WAITING,
    TaskStatus.BLOCKED,
}


class StateSnapshotLoader:
    """只读加载器；不把完整数据库或完整 Memory 暴露给理解器。"""

    def __init__(self, store: StateStore, *, limit: int = 20) -> None:
        self.store = store
        self.limit = max(1, limit)

    def load(self, conversation_id: str, *, exclude_run_id: str | None = None, exclude_task_id: str | None = None) -> StateSnapshot:
        messages = self.store.list_messages(conversation_id, limit=self.limit)
        runs = [item for item in self.store.list_runs(limit=self.limit * 4) if item.conversation_id == conversation_id]
        if exclude_run_id:
            runs = [item for item in runs if item.id != exclude_run_id]
        tasks = self.store.list_tasks(conversation_id, limit=self.limit * 2)
        if exclude_task_id:
            tasks = [item for item in tasks if item.id != exclude_task_id]
        active_run = next((item for item in runs if item.status in _ACTIVE_RUN_STATUSES), None)
        active_task = self.store.get_task(active_run.task_id) if active_run and active_run.task_id else next((item for item in tasks if item.status in _ACTIVE_TASK_STATUSES), None)

        run_ids = {item.id for item in runs}
        artifacts = [item for item in self.store.list_artifacts() if item.run_id in run_ids][: self.limit]
        datasets = [
            item
            for item in self.store.list_datasets()
            if item.created_by_run_id in run_ids
        ][: self.limit]
        last_run = runs[0] if runs else None
        last_result = _result_field(last_run, "summary")
        last_error = last_run.error if last_run else None
        if not last_error:
            last_error = _result_field(last_run, "error")
        last_action = None
        if last_run:
            events = self.store.list_events(last_run.id)
            if events:
                last_action = events[-1].message or events[-1].event_type

        return StateSnapshot(
            conversation_id=conversation_id,
            active_task_id=active_task.id if active_task else None,
            active_run_id=active_run.id if active_run else None,
            task_goal=active_task.goal if active_task else None,
            task_status=active_task.status if active_task else None,
            last_run_status=last_run.status if last_run else None,
            last_action=last_action,
            last_result=last_result,
            last_error=last_error,
            recent_messages=messages,
            recent_runs=runs[: self.limit],
            recent_artifacts=artifacts,
            recent_datasets=datasets,
            known_task_ids=[item.id for item in tasks],
            known_run_ids=[item.id for item in runs],
            known_artifact_ids=[item.id for item in artifacts],
            known_dataset_ids=[item.id for item in datasets],
        )


def _result_field(run, field: str) -> str | None:
    if run is None:
        return None
    result = run.metadata.get("result")
    if isinstance(result, dict):
        value = result.get(field)
        return str(value) if value else None
    return None

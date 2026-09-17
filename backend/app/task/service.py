"""创建和更新一个用户整体目标的 Task。"""

from __future__ import annotations

from datetime import UTC, datetime

from app.core.models import SubTask, Task, TaskStatus
from app.task.repository import TaskRepository


class TaskService:
    def __init__(self, repository: TaskRepository) -> None:
        self.repository = repository

    def create(self, goal: str, *, conversation_id: str | None = None, subtasks: list[SubTask] | None = None) -> Task:
        items = subtasks or []
        task = Task(goal=goal, conversation_id=conversation_id, subtasks=[item.id for item in items])
        self.repository.save(task)
        for item in items:
            self.repository.save_subtask(task.id, item)
        return task

    def update(self, task: Task, *, status: TaskStatus, result: str | None = None) -> Task:
        now = datetime.now(UTC)
        updated = task.model_copy(update={"status": status, "result": result, "updated_at": now})
        return self.repository.save(updated)

    def attach_subtasks(self, task: Task, subtasks: list[SubTask]) -> Task:
        """把本次拆解结果挂回整体 Task，避免子任务只存在于 Trace。"""

        updated = task.model_copy(update={
            "subtasks": [item.id for item in subtasks],
            "updated_at": datetime.now(UTC),
        })
        self.repository.save(updated)
        for item in subtasks:
            self.repository.save_subtask(updated.id, item)
        return updated

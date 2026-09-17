"""任务持久化门面。"""

from app.core.models import SubTask, Task
from app.state import StateStore


class TaskRepository:
    def __init__(self, store: StateStore) -> None:
        self.store = store

    def save(self, task: Task) -> Task:
        self.store.save_task(task)
        return task

    def save_subtask(self, task_id: str, subtask: SubTask) -> SubTask:
        self.store.save_subtask(task_id, subtask)
        return subtask


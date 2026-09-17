"""SubTask 依赖图。"""

from app.core.models import SubTask
from app.decision.parallelism import ParallelismAnalyzer


class TaskGraph:
    def __init__(self, tasks: list[SubTask]) -> None:
        self.tasks = tasks
        self._analyzer = ParallelismAnalyzer()

    def parallel_batches(self) -> list[list[SubTask]]:
        return self._analyzer.batches(self.tasks)


"""基于依赖关系的并行批次分析。"""

from __future__ import annotations

from app.core.models import SubTask


class ParallelismAnalyzer:
    def batches(self, tasks: list[SubTask]) -> list[list[SubTask]]:
        pending = {task.id: task for task in tasks}
        finished: set[str] = set()
        batches: list[list[SubTask]] = []
        while pending:
            ready = [task for task in pending.values() if set(task.dependencies).issubset(finished)]
            if not ready:
                raise ValueError("SubTask 依赖图存在环或缺失依赖。")
            serial = next((task for task in ready if not task.parallelizable), None)
            if serial is not None:
                ready = [serial]
            batches.append(ready)
            for task in ready:
                pending.pop(task.id)
                finished.add(task.id)
        return batches

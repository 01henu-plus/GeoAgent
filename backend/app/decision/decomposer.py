"""根据主题独立性生成一次性 SubTask。"""

from __future__ import annotations

from app.core.models import AgentRequest, Dataset, SubTask


class TaskDecomposer:
    def decompose(self, request: AgentRequest, datasets: list[Dataset]) -> list[SubTask]:
        text = request.user_input.casefold()
        tasks: list[SubTask] = []
        used: set[str] = set()
        role_keywords = {
            "road": ("道路", "road", "路网", "可达"),
            "population": ("人口", "population", "人群"),
            "terrain": ("dem", "高程", "地形", "坡度", "terrain"),
        }
        for role, keywords in role_keywords.items():
            if any(keyword in text for keyword in keywords):
                candidate = next((item for item in datasets if role in item.name.casefold() or any(keyword in item.name.casefold() for keyword in keywords)), None)
                if candidate:
                    operation = "vector.validate" if role == "road" else "raster.slope" if role == "terrain" else "dataset.inspect"
                    tasks.append(
                        SubTask(
                            goal=f"完成 {role} 主题分析",
                            description=f"使用 {candidate.name} 生成 {role} 的可验证摘要",
                            operation=operation,
                            dataset_ids=[candidate.id],
                            required=role != "terrain",
                            parallelizable=True,
                        )
                    )
                    used.add(candidate.id)
        if len(tasks) < 2:
            remaining = [item for item in datasets if item.id not in used]
            for item in remaining[:3 - len(tasks)]:
                tasks.append(SubTask(goal=f"检查 {item.name}", description="提供数据质量和空间范围摘要", operation="dataset.inspect", dataset_ids=[item.id], required=False, parallelizable=True))
        return tasks[:5]

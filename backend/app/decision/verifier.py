"""结果验证：Tool 成功并不等于空间任务成功。"""

from __future__ import annotations

from app.core.models import Dataset, ToolResult
from app.gis.dataset.inspector import DatasetInspector


class ResultVerifier:
    def __init__(self, inspector: DatasetInspector | None = None) -> None:
        self.inspector = inspector or DatasetInspector()

    def verify(self, result: ToolResult, datasets: dict[str, Dataset]) -> tuple[bool, list[str]]:
        problems: list[str] = []
        if result.status.value not in {"SUCCESS", "PARTIAL_SUCCESS"}:
            return False, [result.error.message if result.error else "Tool 未成功完成"]
        for dataset_id in result.datasets:
            dataset = datasets.get(dataset_id)
            if dataset is None:
                problems.append(f"结果引用了未知 Dataset：{dataset_id}")
                continue
            try:
                checked = self.inspector.inspect(dataset.path)
            except Exception as exc:
                problems.append(f"结果不可读：{dataset.name} ({exc})")
                continue
            if checked.schema and checked.schema.feature_count == 0:
                problems.append(f"结果为空：{dataset.name}")
        return not problems, problems


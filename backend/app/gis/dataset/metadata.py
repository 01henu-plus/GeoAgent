"""Dataset 元数据的展示格式。"""

from app.core.models import Dataset


def compact_metadata(dataset: Dataset) -> dict:
    """生成适合 Agent Context 的小型、稳定字典。"""

    return dataset.model_dump(mode="json")


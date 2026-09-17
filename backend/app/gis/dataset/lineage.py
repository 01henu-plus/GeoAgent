"""数据集谱系查询服务。"""

from app.state import StateStore


class LineageService:
    def __init__(self, store: StateStore) -> None:
        self.store = store

    def parents(self, dataset_id: str) -> list[dict]:
        return self.store.list_lineage(output_dataset_id=dataset_id)

    def all(self) -> list[dict]:
        return self.store.list_lineage()


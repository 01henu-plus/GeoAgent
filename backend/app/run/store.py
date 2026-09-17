"""Run 存储兼容门面。"""

from app.state import StateStore


class RunStore:
    def __init__(self, store: StateStore) -> None:
        self.store = store

    def get(self, run_id: str):
        return self.store.get_run(run_id)

    def list(self, limit: int = 50):
        return self.store.list_runs(limit)

    def delete(self, run_id: str) -> bool:
        return self.store.delete_run(run_id)

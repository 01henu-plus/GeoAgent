"""Checkpoint 存取门面。"""

from app.core.models import Checkpoint
from app.state import StateStore


class CheckpointStore:
    def __init__(self, store: StateStore) -> None:
        self.store = store

    def save(self, checkpoint: Checkpoint) -> Checkpoint:
        self.store.save_checkpoint(checkpoint)
        return checkpoint

    def latest(self, run_id: str) -> Checkpoint | None:
        return self.store.latest_checkpoint(run_id)


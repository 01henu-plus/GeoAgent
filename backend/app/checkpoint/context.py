"""Checkpoint 内容构造。"""

from typing import Any

from app.core.models import Checkpoint


def make_checkpoint(run_id: str, phase: str, state: dict[str, Any]) -> Checkpoint:
    return Checkpoint(run_id=run_id, phase=phase, state=state)


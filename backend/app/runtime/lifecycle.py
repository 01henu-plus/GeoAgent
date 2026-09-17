"""Run 生命周期辅助函数。"""

from datetime import UTC, datetime

from app.core.models import Run, RunStatus


def start_run(run: Run) -> Run:
    return run.model_copy(update={"status": RunStatus.RUNNING, "started_at": datetime.now(UTC)})


def finish_run(run: Run, status: RunStatus, *, error: str | None = None) -> Run:
    return run.model_copy(update={"status": status, "error": error, "finished_at": datetime.now(UTC)})


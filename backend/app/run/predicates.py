"""Run 状态领域谓词，集中维护活动和可重试失败的语义。"""

from app.core.models import Run, RunStatus

ACTIVE_RUN_STATUSES = frozenset(
    {
        RunStatus.CREATED,
        RunStatus.PLANNING,
        RunStatus.RUNNING,
        RunStatus.WAITING_TOOL,
        RunStatus.WAITING_SUBAGENT,
        RunStatus.WAITING_USER,
        RunStatus.WAITING_APPROVAL,
        RunStatus.RETRYING,
        RunStatus.REPLANNING,
        RunStatus.VALIDATING,
    }
)


def is_active_run(run: Run) -> bool:
    return run.status in ACTIVE_RUN_STATUSES


def is_retryable_failed_run(run: Run) -> bool:
    return run.status is RunStatus.FAILED or (
        bool(run.error)
        and run.status not in {RunStatus.WAITING_USER, RunStatus.WAITING_APPROVAL, RunStatus.CANCELLED}
    )


__all__ = ["ACTIVE_RUN_STATUSES", "is_active_run", "is_retryable_failed_run"]

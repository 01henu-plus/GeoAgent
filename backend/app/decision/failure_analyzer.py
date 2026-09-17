"""将结构化 ToolError 转成恢复动作。"""

from __future__ import annotations

from app.core.models import FailureAction, ToolResult


class FailureAnalyzer:
    def analyze(self, result: ToolResult) -> tuple[FailureAction, str]:
        if result.error is None:
            return FailureAction.ABORT, "没有错误信息，无法安全恢复。"
        code = result.error.code
        if code in {"EXECUTION_TIMEOUT", "RATE_LIMIT", "SHELL_EXECUTION_FAILED"} and result.retryable:
            return FailureAction.RETRY, "执行可能是临时性失败，且 Tool 标记为可重试。"
        if code in {"CRS_UNIT_MISMATCH", "CRS_MISMATCH", "CRS_MISSING", "INVALID_GEOMETRY"}:
            return FailureAction.REPAIR, "输入条件可通过 CRS 对齐、geometry 修复或字段重映射改善。"
        if code in {"MISSING_DATASET", "MISSING_FIELD", "UNSUPPORTED_FORMAT", "NO_OVERLAP"}:
            return FailureAction.ASK_USER, "缺少可靠的输入或必要约束，不能猜测。"
        if code in {"ALGORITHM_NOT_APPLICABLE", "EMPTY_DATASET"}:
            return FailureAction.REPLAN, "当前算法不适用于实际数据，应该改用更合适的方案。"
        return FailureAction.ABORT, "错误不可安全恢复。"

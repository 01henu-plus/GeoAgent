"""异步审批的最小数据协议；第一版默认由 API 返回 BLOCKED。"""

from pydantic import BaseModel


class ApprovalRequest(BaseModel):
    tool_name: str
    reason: str
    run_id: str | None = None
    approved: bool = False


class ApprovalGate:
    def request(self, tool_name: str, reason: str, *, run_id: str | None = None) -> ApprovalRequest:
        return ApprovalRequest(tool_name=tool_name, reason=reason, run_id=run_id)


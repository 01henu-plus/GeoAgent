"""第一批离线 Evaluation case 的可执行清单。"""

from app.core.models import FailureAction, ToolError, ToolResult, ToolStatus
from app.decision import FailureAnalyzer


def test_crs_failure_maps_to_repair():
    result = ToolResult(call_id="call", status=ToolStatus.FAILED, error=ToolError(code="CRS_UNIT_MISMATCH", message="degrees"))
    assert FailureAnalyzer().analyze(result)[0] is FailureAction.REPAIR


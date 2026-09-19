"""处理高置信、低复杂度请求的确定性规则门。"""

from __future__ import annotations

import re

from app.core.models import InteractionMode, RequestFrame, ResolvedReference, StateSnapshot
from app.understanding.models import ReferenceResolution
from app.understanding.patterns import (
    is_cancel_request,
    is_continue_request,
    is_modify_request,
    is_retry_request,
    strip_modify_prefix,
)

_GREETING_RE = re.compile(r"^(你好|您好|嗨|哈喽|hello|hi|hey|在吗)[。！!，,~～ ]*$", re.IGNORECASE)

_CAPABILITY_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("raster_analysis", ("栅格", "影像", "ndvi", "坡度", "坡向", "高程", "raster")),
    ("vector_analysis", ("矢量", "缓冲", "裁剪", "相交", "道路", "边界", "vector")),
    ("artifact_read", ("读取", "使用", "刚才", "上一个", "结果", "文件", "数据")),
    ("artifact_write", ("保存", "导出", "生成", "输出", "写入")),
    ("dataset_inspection", ("检查", "查看属性", "元数据", "字段", "数据质量")),
    ("crs_transform", ("重投影", "坐标系", "epsg", "投影转换")),
    ("python_execution", ("python", "脚本", "代码")),
    ("knowledge_lookup", ("什么是", "如何", "原理", "为什么", "区别")),
    ("result_query", ("运行状态", "生成了哪些", "有哪些文件", "结果在哪里")),
    ("run_diagnosis", ("运行状态", "运行记录", "为什么失败", "错误信息", "trace", "日志")),
)


class RuleGate:
    def match(self, message: str, state: StateSnapshot, resolution: ReferenceResolution) -> RequestFrame | None:
        compact = message.strip()
        lowered = compact.casefold()
        if _GREETING_RE.fullmatch(compact):
            return self._frame(InteractionMode.CHAT, compact, resolution, capabilities=["conversation"], confidence=0.99)
        if is_cancel_request(compact, strict=True):
            if not state.active_task_id:
                return None
            return self._frame(InteractionMode.CANCEL_TASK, "取消当前任务", resolution, target_task_id=state.active_task_id, target_run_id=state.active_run_id, confidence=0.99)
        if is_continue_request(compact):
            if not state.active_task_id:
                return None
            goal = state.task_goal or compact
            return self._frame(InteractionMode.CONTINUE_TASK, goal, resolution, target_task_id=state.active_task_id, target_run_id=state.active_run_id, capabilities=infer_capabilities(f"{compact} {goal}", references=resolution.references), confidence=0.98)
        if is_retry_request(compact, strict=True):
            if not _has_failed_run(state):
                return None
            return self._frame(InteractionMode.RETRY_TASK, state.task_goal or compact, resolution, target_task_id=state.active_task_id, target_run_id=_failed_run_id(state), capabilities=infer_capabilities(f"{compact} {state.task_goal or ''}", references=resolution.references), confidence=0.98)
        if state.active_task_id and is_modify_request(compact):
            constraint = strip_modify_prefix(compact)
            return self._frame(InteractionMode.MODIFY_TASK, compact, resolution, constraints=[constraint] if constraint else [], target_task_id=state.active_task_id, target_run_id=state.active_run_id, capabilities=infer_capabilities(compact, references=resolution.references), confidence=0.94)
        if _is_result_query(lowered):
            return self._frame(InteractionMode.QUERY, compact, resolution, target_task_id=state.active_task_id, target_run_id=state.active_run_id, capabilities=["result_query", "artifact_read"], confidence=0.95)
        return None

    @staticmethod
    def _frame(mode: InteractionMode, goal: str, resolution: ReferenceResolution, *, target_task_id: str | None = None, target_run_id: str | None = None, constraints: list[str] | None = None, capabilities: list[str] | None = None, confidence: float) -> RequestFrame:
        return RequestFrame(
            mode=mode,
            goal=goal,
            references=resolution.references,
            constraints=constraints or [],
            capabilities=capabilities or [],
            target_task_id=target_task_id,
            target_run_id=target_run_id,
            needs_planning=mode in {InteractionMode.NEW_TASK, InteractionMode.CONTINUE_TASK, InteractionMode.MODIFY_TASK, InteractionMode.RETRY_TASK},
            needs_tool=bool(capabilities and set(capabilities) - {"conversation", "knowledge_lookup", "result_query"}),
            unresolved_references=resolution.unresolved_references,
            confidence=confidence,
        )


def _has_failed_run(state: StateSnapshot) -> bool:
    return any(run.status.value == "FAILED" for run in state.recent_runs)


def _failed_run_id(state: StateSnapshot) -> str | None:
    for run in state.recent_runs:
        if run.status.value == "FAILED":
            return run.id
    return None


def _is_result_query(text: str) -> bool:
    return any(term in text for term in ("生成了哪些文件", "有哪些文件", "结果在哪里", "运行状态", "查看结果", "刚才生成了什么"))


def infer_capabilities(text: str, *, operations: list[str] | None = None, references: list[ResolvedReference] | None = None) -> list[str]:
    lowered = text.casefold()
    capabilities = [name for name, terms in _CAPABILITY_TERMS if any(term.casefold() in lowered for term in terms)]
    operation_set = set(operations or [])
    if operation_set.intersection({"slope", "zonal_statistics", "clip"}) and "raster_analysis" not in capabilities:
        capabilities.append("raster_analysis")
    if operation_set.intersection({"buffer", "intersection", "spatial_join", "distance", "dissolve", "repair"}) and "vector_analysis" not in capabilities:
        capabilities.append("vector_analysis")
    if "reproject" in operation_set and "crs_transform" not in capabilities:
        capabilities.append("crs_transform")
    if references and "artifact_read" not in capabilities:
        capabilities.append("artifact_read")
    return list(dict.fromkeys(capabilities))


__all__ = ["RuleGate", "infer_capabilities"]

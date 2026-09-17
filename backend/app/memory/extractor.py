"""从明确的用户表达中提取少量 ProjectMemory 候选。"""

from __future__ import annotations

import hashlib
import re

from app.core.models import AgentRequest, AgentResult, RequestFrame, Run

from .models import MemoryCandidate


class MemoryExtractor:
    """第一版只处理用户明确要求长期记住的稳定事实。"""

    def extract(
        self,
        request: AgentRequest,
        frame: RequestFrame | None,
        run: Run,
        result: AgentResult,
    ) -> list[MemoryCandidate]:
        text = request.user_input.strip()
        if not _is_durable_request(text):
            return []
        value = _clean_fact(text)
        if not value:
            return []
        key = _fact_key(value)
        category = "project_constraint" if frame and frame.constraints else "project_fact"
        return [
            MemoryCandidate(
                key=key,
                value=value,
                category=category,
                source_task_id=run.task_id,
                source_run_id=run.id,
                confidence=0.95,
                importance=0.7,
                durability="durable",
                metadata={"source": "explicit_user_request", "result_status": result.status.value},
            )
        ]


def _is_durable_request(text: str) -> bool:
    return any(marker in text for marker in ("记住", "以后都", "这个项目一直", "项目默认", "长期保留"))


def _clean_fact(text: str) -> str:
    cleaned = re.sub(r"^(请)?记住[：:，,\s]*", "", text)
    cleaned = re.sub(r"^以后都[：:，,\s]*", "", cleaned)
    return cleaned.strip(" 。.!！") or text.strip(" 。.!！")


def _fact_key(value: str) -> str:
    normalized = value.casefold()
    if "crs" in normalized or "坐标系" in value:
        return "project_default_crs"
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return f"project_fact_{digest}"


__all__ = ["MemoryExtractor"]

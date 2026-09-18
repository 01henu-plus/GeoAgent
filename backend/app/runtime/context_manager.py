"""Context Engineering 的兼容门面。

旧调用方继续使用 ``main_context`` / ``sub_context``，实际生成过程已经拆成：
Assembler -> ContextSection -> Compressor -> ModelContext。
"""

from __future__ import annotations

import json
from typing import Any

from app.core.models import (
    AgentRequest,
    ConversationMemory,
    Dataset,
    IntentResult,
    MemoryItem,
    Plan,
    RequestFrame,
    RequestResources,
    Run,
    UserProfile,
    WorkingMemory,
)
from app.runtime.context_assembler import ContextAssembler
from app.runtime.context_compressor import ContextCompressor


class ContextManager:
    """保持历史 API 的 facade；新代码以 token budget 为核心。"""

    def __init__(self, *, max_tokens: int = 6000, max_chars: int | None = None) -> None:
        self.max_tokens = max_tokens
        self._legacy_max_chars = max_chars
        self.assembler = ContextAssembler()
        self.compressor = ContextCompressor(estimator=self.assembler.estimator)

    def main_context(
        self,
        request: AgentRequest,
        datasets: list[Dataset],
        plan: Plan | None,
        memories: list[MemoryItem],
        *,
        conversation: list[dict[str, Any]] | None = None,
        tool_definitions: list[dict[str, Any]] | None = None,
        working_memory: WorkingMemory | dict[str, Any] | None = None,
        budget: dict[str, Any] | None = None,
        referenced_runs: list[dict[str, Any]] | None = None,
        findings: list[Any] | None = None,
        errors: list[str] | None = None,
        intent_hint: IntentResult | None = None,
        request_frame: RequestFrame | None = None,
        request_resources: RequestResources | None = None,
        task_goal: str | None = None,
        run_state: Run | dict[str, Any] | None = None,
        current_observation: Any = None,
        user_profile: UserProfile | dict[str, Any] | None = None,
        conversation_memory: ConversationMemory | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sections = self.assembler.main_sections(
            request,
            datasets,
            plan,
            memories,
            conversation=conversation,
            tool_definitions=tool_definitions,
            working_memory=working_memory,
            budget=budget,
            referenced_runs=referenced_runs,
            findings=findings,
            errors=errors,
            intent_hint=intent_hint,
            request_frame=request_frame,
            request_resources=request_resources,
            task_goal=task_goal,
            run_state=run_state,
            current_observation=current_observation,
            user_profile=user_profile,
            conversation_memory=conversation_memory,
        )
        context = self.compressor.compress(sections, max_tokens=self.max_tokens)
        if self._legacy_max_chars is not None:
            return self._legacy_bound(context)
        return context

    def sub_context(
        self,
        request: AgentRequest,
        subtask: dict[str, Any],
        datasets: list[Dataset],
        *,
        parent_findings: list[Any] | None = None,
        working_memory: WorkingMemory | dict[str, Any] | None = None,
        budget: dict[str, Any] | None = None,
        allowed_tools: list[str] | None = None,
    ) -> dict[str, Any]:
        sections = self.assembler.sub_sections(
            request,
            subtask,
            datasets,
            parent_findings=parent_findings,
            working_memory=working_memory,
            budget=budget,
            allowed_tools=allowed_tools,
        )
        context = self.compressor.compress(sections, max_tokens=self.max_tokens)
        if self._legacy_max_chars is not None:
            return self._legacy_bound(context)
        return context

    def _legacy_bound(self, context: dict[str, Any]) -> dict[str, Any]:
        """兼容旧的 ``max_chars`` 调用；新代码不依赖这个分支。"""

        limit = max(80, self._legacy_max_chars or 80)
        if len(_serialize(context)) <= limit:
            return context
        result: dict[str, Any] = {"user_request": str(context.get("user_request", "")), "truncated": True}
        while len(_serialize(result)) > limit and result["user_request"]:
            result["user_request"] = result["user_request"][:-max(1, len(result["user_request"]) // 10)]
        if len(_serialize(result)) > limit:
            result["user_request"] = result["user_request"][: max(0, limit - 38)]
        return result


def _serialize(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


__all__ = ["ContextManager"]

"""Context Engineering 的兼容门面。

旧调用方继续使用 ``main_context`` / ``sub_context``，实际生成过程已经拆成：
Assembler -> ContextSection -> Compressor -> ModelContext。
"""

from __future__ import annotations

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

    def __init__(self, *, max_tokens: int = 6000) -> None:
        self.max_tokens = max_tokens
        self.assembler = ContextAssembler()
        self.compressor = ContextCompressor(max_tokens=max_tokens)

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
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        resource_view = _request_resources_view(request, request_resources)
        sections = self.assembler.assemble_main(
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
            request_resources=resource_view,
            task_goal=task_goal,
            run_state=run_state,
            current_observation=current_observation,
            user_profile=user_profile,
            conversation_memory=conversation_memory,
        )
        return self.compressor.compress(sections, max_tokens=max_tokens)

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
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        sections = self.assembler.assemble_sub(
            request,
            subtask,
            datasets,
            parent_findings=parent_findings,
            working_memory=working_memory,
            budget=budget,
            allowed_tools=allowed_tools,
        )
        return self.compressor.compress(sections, max_tokens=max_tokens)


def _request_resources_view(request: AgentRequest, resources: RequestResources | None) -> dict[str, Any]:
    return {
        "dataset_ids": list(dict.fromkeys([*(item.id for item in resources.datasets)] if resources else [*request.dataset_ids])),
        "attachment_ids": list(dict.fromkeys(request.attachment_ids)),
        "referenced_run_ids": list(dict.fromkeys([*(item.id for item in resources.runs)] if resources else [*request.referenced_run_ids])),
    }


__all__ = ["ContextManager"]

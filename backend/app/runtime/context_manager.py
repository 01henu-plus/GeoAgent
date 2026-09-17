"""Context Engineering：只向 Agent 提供相关且有界的空间上下文。"""

from __future__ import annotations

import json
from typing import Any

from app.core.models import AgentRequest, Dataset, IntentResult, MemoryItem, Plan, RequestFrame


class ContextManager:
    def __init__(self, *, max_chars: int = 24000) -> None:
        self.max_chars = max_chars

    def main_context(
        self,
        request: AgentRequest,
        datasets: list[Dataset],
        plan: Plan | None,
        memories: list[MemoryItem],
        *,
        conversation: list[dict[str, Any]] | None = None,
        tool_definitions: list[dict[str, Any]] | None = None,
        working_memory: dict[str, Any] | None = None,
        budget: dict[str, Any] | None = None,
        referenced_runs: list[dict[str, Any]] | None = None,
        findings: list[Any] | None = None,
        errors: list[str] | None = None,
        intent_hint: IntentResult | None = None,
        request_frame: RequestFrame | None = None,
    ) -> dict[str, Any]:
        context = {
            "user_request": request.user_input,
            "goal": request.user_input,
            "request_context": request.context,
            "request_frame": request_frame.model_dump(mode="json") if request_frame else None,
            "deterministic_hint": {
                "intent": intent_hint.model_dump(mode="json") if intent_hint else None,
                "plan": plan.model_dump(mode="json") if plan else None,
                "instruction": "仅供模型参考，不代表已经确认的用户意图，也不是必须执行的步骤。",
            },
            "datasets": [dataset.model_dump(mode="json") for dataset in datasets],
            "conversation": conversation or [],
            "tool_definitions": tool_definitions or [],
            "working_memory": working_memory or {},
            "project_memory": [memory.model_dump(mode="json") for memory in memories],
            "referenced_runs": referenced_runs or [],
            "findings": findings or [],
            "errors": errors or [],
            "budget": budget or {},
        }
        return self._bound(context)

    def sub_context(
        self,
        request: AgentRequest,
        subtask: dict[str, Any],
        datasets: list[Dataset],
        *,
        parent_findings: list[Any] | None = None,
        working_memory: dict[str, Any] | None = None,
        budget: dict[str, Any] | None = None,
        allowed_tools: list[str] | None = None,
    ) -> dict[str, Any]:
        return self._bound(
            {
                "user_request": request.user_input,
                "subtask": subtask,
                "datasets": [item.model_dump(mode="json") for item in datasets],
                "parent_findings": parent_findings or [],
                "working_memory": working_memory or {},
                "budget": budget or {},
                "allowed_tools": allowed_tools or [
                    "dataset.inspect",
                    "vector.validate",
                    "analysis.distance",
                    "raster.inspect",
                    "raster.slope",
                ],
            }
        )

    def _bound(self, context: dict[str, Any]) -> dict[str, Any]:
        context = dict(context)
        if len(_serialize(context)) <= self.max_chars:
            return context

        context["truncated"] = True
        list_keys = (
            "conversation",
            "findings",
            "parent_findings",
            "tool_definitions",
            "referenced_runs",
            "project_memory",
            "datasets",
        )
        while len(_serialize(context)) > self.max_chars:
            candidates = [key for key in list_keys if isinstance(context.get(key), list) and context[key]]
            if candidates:
                key = max(candidates, key=lambda item: len(_serialize(context[item])))
                context[key] = context[key][1:]
                continue
            string_keys = [key for key, value in context.items() if isinstance(value, str) and value]
            if string_keys:
                key = max(string_keys, key=lambda item: len(context[item]))
                value = context[key]
                context[key] = value[: max(0, len(value) - max(1, len(_serialize(context)) - self.max_chars))]
                continue
            break

        if len(_serialize(context)) > self.max_chars:
            return {"user_request": str(context.get("user_request", ""))[: max(0, self.max_chars - 40)], "truncated": True}
        return context


def _serialize(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))

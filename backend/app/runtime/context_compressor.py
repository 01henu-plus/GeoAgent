"""Deterministic ContextCompressor：只压缩临时 ContextSection。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

from .context_assembler import ContextPriority, ContextSection, estimate_tokens


class ContextCompressor:
    def __init__(self, *, max_tokens: int = 6000) -> None:
        self.max_tokens = max(128, max_tokens)

    def compress(self, sections: list[ContextSection], *, max_tokens: int | None = None) -> dict[str, Any]:
        working = [replace(section, content=deepcopy(section.content)) for section in sections]
        changed: set[str] = set()
        dropped: list[str] = []
        budget = max(1, max_tokens if max_tokens is not None else self.max_tokens)
        target = max(64, budget - 40)

        for priority in (ContextPriority.LOW, ContextPriority.MEDIUM, ContextPriority.HIGH):
            while self._estimate(working) > target:
                candidates = [
                    section
                    for section in working
                    if section.priority == priority
                    and section.compressible
                    and not section.required
                    and self._can_reduce_optional(section)
                ]
                if not candidates:
                    break
                section = max(
                    candidates,
                    key=lambda item: (
                        int(item.metadata.get("drop_rank", 0)),
                        item.estimated_tokens,
                        item.name,
                    ),
                )
                replacement = self._shrink(section)
                index = working.index(section)
                if replacement is None:
                    dropped.append(section.name)
                    working.pop(index)
                else:
                    changed.add(section.name)
                    working[index] = replacement

        payload = {section.name: section.content for section in working}
        required_sections = {section.name: section.content for section in working if section.required}
        required_tokens = estimate_tokens(required_sections)
        payload["truncated"] = False
        payload["context_meta"] = {
            "truncated": False,
            "over_budget": False,
            "budget_tokens": budget,
            "estimated_tokens": estimate_tokens(payload),
            "overflow_tokens": 0,
            "required_tokens": required_tokens,
        }
        if changed:
            payload["context_meta"]["compressed_sections"] = sorted(changed)
        if dropped:
            payload["context_meta"]["dropped_sections"] = dropped
        # metadata 自身也占用少量 token，因此固定点迭代一次，确保顶层兼容
        # 字段和 context_meta 使用同一个 canonical truncated 值。
        for _ in range(2):
            estimated_tokens = estimate_tokens(payload)
            overflow_tokens = max(0, estimated_tokens - budget)
            truncated = bool(changed or dropped or overflow_tokens > 0)
            payload["context_meta"].update(
                {
                    "truncated": truncated,
                    "over_budget": overflow_tokens > 0,
                    "estimated_tokens": estimated_tokens,
                    "overflow_tokens": overflow_tokens,
                }
            )
            payload["truncated"] = truncated
        return payload

    @staticmethod
    def _can_reduce_optional(section: ContextSection) -> bool:
        """列表达到最小保留数时停止压缩，避免丢掉最高相关项。"""

        if not isinstance(section.content, list):
            return True
        minimum = max(0, int(section.metadata.get("min_items", 0)))
        return len(section.content) > minimum

    @staticmethod
    def _estimate(sections: list[ContextSection]) -> int:
        return estimate_tokens({section.name: section.content for section in sections})

    def _shrink(self, section: ContextSection) -> ContextSection | None:
        content = section.content
        kind = section.metadata.get("kind")

        if kind == "datasets" and isinstance(content, list):
            compacted = self._shrink_dataset_list(content)
            if compacted != content:
                return self._replace_content(section, compacted)

        if kind == "conversation_memory" and isinstance(content, dict):
            compacted = self._shrink_conversation_memory(content)
            if compacted != content:
                return self._replace_content(section, compacted)

        if kind == "working_memory_details" and isinstance(content, dict):
            compacted = self._shrink_working_details(content)
            if compacted != content:
                return self._replace_content(section, compacted)

        if isinstance(content, list):
            min_items = max(0, int(section.metadata.get("min_items", 0)))
            if len(content) > min_items:
                updated = list(content)
                if section.metadata.get("trim_side") == "left":
                    updated.pop(0)
                else:
                    updated.pop()
                if not updated and min_items == 0:
                    return None
                return self._replace_content(section, updated)
            return None

        if isinstance(content, dict):
            removable = [key for key, value in content.items() if value in (None, "", [], {})]
            if removable:
                updated = dict(content)
                updated.pop(removable[0], None)
                return self._replace_content(section, updated)
            return None

        if isinstance(content, str) and len(content) > 256:
            return self._replace_content(section, _clip(content, max(256, int(len(content) * 0.7))))
        return None

    @staticmethod
    def _shrink_dataset_list(content: list[Any]) -> list[Any]:
        updated = deepcopy(content)
        for index in range(len(updated) - 1, -1, -1):
            item = updated[index]
            if not isinstance(item, dict):
                continue
            if item.get("metadata"):
                item.pop("metadata", None)
                return updated
            schema = item.get("schema")
            if isinstance(schema, dict) and isinstance(schema.get("fields"), dict) and len(schema["fields"]) > 12:
                schema["fields"] = dict(list(schema["fields"].items())[: max(12, len(schema["fields"]) // 2)])
                return updated
        if len(updated) > 1:
            updated.pop()
        return updated

    @staticmethod
    def _shrink_conversation_memory(content: dict[str, Any]) -> dict[str, Any]:
        updated = deepcopy(content)
        for key in ("important_references", "key_facts", "decisions"):
            values = updated.get(key)
            if isinstance(values, list) and values:
                values.pop(0)
                if not values:
                    updated.pop(key, None)
                return updated
        summary = updated.get("summary")
        if isinstance(summary, str) and len(summary) > 500:
            updated["summary"] = _clip(summary, max(500, int(len(summary) * 0.7)))
            return updated
        return updated

    @staticmethod
    def _shrink_working_details(content: dict[str, Any]) -> dict[str, Any]:
        updated = deepcopy(content)
        for key in ("intermediate_results", "assumptions"):
            values = updated.get(key)
            if isinstance(values, list) and values:
                values.pop(0)
                if not values:
                    updated.pop(key, None)
                return updated
        return updated

    @staticmethod
    def _replace_content(section: ContextSection, content: Any) -> ContextSection:
        return replace(section, content=content, estimated_tokens=estimate_tokens(content))


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)] + "…"


__all__ = ["ContextCompressor"]

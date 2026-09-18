"""Deterministic ContextCompressor：只压缩临时 ContextSection。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

from .context_assembler import ContextPriority, ContextSection, estimate_tokens


class ContextCompressor:
    def __init__(self, *, max_tokens: int = 6000) -> None:
        self.max_tokens = max(128, max_tokens)

    def compress(self, sections: list[ContextSection]) -> dict[str, Any]:
        working = [replace(section, content=deepcopy(section.content)) for section in sections]
        changed: set[str] = set()
        dropped: list[str] = []
        target = max(64, self.max_tokens - 40)

        for priority in (ContextPriority.LOW, ContextPriority.MEDIUM, ContextPriority.HIGH):
            while self._estimate(working) > target:
                candidates = [
                    section
                    for section in working
                    if section.priority == priority and section.compressible and not section.required
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

        # 极端情况下才压缩 REQUIRED section；section 本身仍然保留。
        while self._estimate(working) > target:
            candidates = [
                section
                for section in working
                if section.required and section.compressible and self._can_shrink_required(section)
            ]
            if not candidates:
                break
            section = max(candidates, key=lambda item: (item.estimated_tokens, item.name))
            replacement = self._shrink_required(section)
            if replacement is None:
                break
            changed.add(section.name)
            working[working.index(section)] = replacement

        payload = {section.name: section.content for section in working}
        truncated = bool(changed or dropped or self._estimate(working) > target)
        payload["context_meta"] = {
            "truncated": truncated,
            "budget_tokens": self.max_tokens,
            "estimated_tokens": 0,
        }
        payload["context_meta"]["estimated_tokens"] = estimate_tokens(payload)
        if changed:
            payload["context_meta"]["compressed_sections"] = sorted(changed)
        if dropped:
            payload["context_meta"]["dropped_sections"] = dropped
        return payload

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

    def _can_shrink_required(self, section: ContextSection) -> bool:
        if isinstance(section.content, str):
            minimum = int(section.metadata.get("hard_min_chars", 128))
            return len(section.content) > minimum
        if section.name == "request_frame" and isinstance(section.content, dict):
            return any(key in section.content for key in ("capabilities", "confidence", "unresolved_references"))
        if section.name == "current_observation" and isinstance(section.content, dict):
            output = section.content.get("output")
            return isinstance(output, (str, dict, list)) and estimate_tokens(output) > 200
        return False

    def _shrink_required(self, section: ContextSection) -> ContextSection | None:
        if isinstance(section.content, str):
            minimum = int(section.metadata.get("hard_min_chars", 128))
            if len(section.content) <= minimum:
                return None
            return self._replace_content(section, _clip(section.content, max(minimum, int(len(section.content) * 0.7))))

        if section.name == "request_frame" and isinstance(section.content, dict):
            updated = dict(section.content)
            for key in ("capabilities", "confidence", "unresolved_references"):
                if key in updated:
                    updated.pop(key)
                    return self._replace_content(section, updated)
            return None

        if section.name == "current_observation" and isinstance(section.content, dict):
            updated = deepcopy(section.content)
            output = updated.get("output")
            if isinstance(output, str) and len(output) > 500:
                updated["output"] = _clip(output, max(500, int(len(output) * 0.6)))
                return self._replace_content(section, updated)
            if isinstance(output, list) and len(output) > 4:
                updated["output"] = output[:4]
                return self._replace_content(section, updated)
            if isinstance(output, dict) and len(output) > 8:
                updated["output"] = dict(list(output.items())[:8])
                return self._replace_content(section, updated)
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

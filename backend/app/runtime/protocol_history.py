"""模型工具调用协议历史的有界管理。

协议历史和动态 Context 不是同一种数据：协议历史必须保持 assistant tool_calls
与对应 tool messages 的完整关系，本模块只按完整 batch 压缩，不修改原始消息。
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from app.core.models import ToolResult
from app.runtime.context_assembler import estimate_tokens


@dataclass(frozen=True, slots=True)
class ProtocolBatch:
    """一个 assistant tool-call 消息及其全部 tool 响应。"""

    assistant: dict[str, Any]
    tool_messages: tuple[dict[str, Any], ...] = ()

    def messages(self) -> list[dict[str, Any]]:
        return [deepcopy(self.assistant), *(deepcopy(item) for item in self.tool_messages)]


def extract_protocol_messages(
    *,
    protocol_messages: list[dict[str, Any]] | None = None,
    legacy_messages: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """优先读取新 checkpoint 字段，兼容旧的 system + dynamic user + history。"""

    if protocol_messages is not None:
        return deepcopy(protocol_messages)
    messages = deepcopy(legacy_messages or [])
    if messages and messages[0].get("role") == "system":
        messages.pop(0)
    if messages and messages[0].get("role") == "user":
        messages.pop(0)
    return messages


def group_protocol_batches(messages: list[dict[str, Any]]) -> list[ProtocolBatch]:
    """按 assistant tool_calls 分组，保证压缩时不留下孤立 tool 消息。"""

    batches: list[ProtocolBatch] = []
    index = 0
    while index < len(messages):
        assistant = deepcopy(messages[index])
        tool_calls = assistant.get("tool_calls") if assistant.get("role") == "assistant" else None
        if isinstance(tool_calls, list) and tool_calls:
            tools: list[dict[str, Any]] = []
            cursor = index + 1
            while cursor < len(messages) and messages[cursor].get("role") == "tool":
                tools.append(deepcopy(messages[cursor]))
                cursor += 1
            batches.append(ProtocolBatch(assistant=assistant, tool_messages=tuple(tools)))
            index = cursor
            continue
        batches.append(ProtocolBatch(assistant=assistant))
        index += 1
    return batches


def compact_protocol_messages(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    max_batches: int = 4,
) -> list[dict[str, Any]]:
    """保留最近 batch，并从最老 batch 开始整批删除直到预算满足。"""

    batches = group_protocol_batches(messages)
    if max_batches > 0:
        batches = batches[-max_batches:]
    compacted = [_compact_batch(batch) for batch in batches]
    budget = max(1, max_tokens)
    while len(compacted) > 1 and estimate_tokens(_flatten(compacted)) > budget:
        compacted.pop(0)
    if compacted and estimate_tokens(_flatten(compacted)) > budget:
        compacted = [_compact_batch(batch, text_limit=320) for batch in compacted[-1:]]
    return _flatten(compacted)


def protocol_tool_result_view(result: ToolResult | dict[str, Any]) -> dict[str, Any]:
    """构造发送给模型的轻量 ToolResult View，不改变原始结果。"""

    raw = result.model_dump(mode="json") if isinstance(result, ToolResult) else dict(result)
    view = {
        "call_id": raw.get("call_id"),
        "status": raw.get("status"),
        "error": _compact_value(raw.get("error")),
        "warnings": _compact_value(raw.get("warnings", [])),
        "datasets": list(raw.get("datasets") or [])[:32],
        "artifacts": list(raw.get("artifacts") or [])[:32],
        "retryable": raw.get("retryable", False),
        "output": _compact_value(raw.get("output")),
    }
    return {key: value for key, value in view.items() if value not in (None, "", [], {})}


def protocol_tool_message(result: ToolResult) -> dict[str, Any]:
    """把 ToolResult 变成合法且紧凑的 role=tool 消息。"""

    return {
        "role": "tool",
        "tool_call_id": result.call_id,
        "content": json.dumps(protocol_tool_result_view(result), ensure_ascii=False, separators=(",", ":")),
    }


def _compact_batch(batch: ProtocolBatch, *, text_limit: int = 1200) -> ProtocolBatch:
    tools = []
    for item in batch.tool_messages:
        message = deepcopy(item)
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = _compact_tool_content(content, text_limit=text_limit)
        else:
            message["content"] = json.dumps(_compact_value(content), ensure_ascii=False, separators=(",", ":"))
        tools.append(message)
    return ProtocolBatch(assistant=deepcopy(batch.assistant), tool_messages=tuple(tools))


def _compact_tool_content(content: str, *, text_limit: int) -> str:
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError, json.JSONDecodeError):
        return content if len(content) <= text_limit else content[: text_limit - 1] + "…"
    compacted = _compact_value(parsed, text_limit=text_limit)
    serialized = json.dumps(compacted, ensure_ascii=False, separators=(",", ":"))
    return serialized if len(serialized) <= text_limit * 2 else serialized[: text_limit * 2 - 1] + "…"


def _compact_value(value: Any, *, depth: int = 0, text_limit: int = 1200) -> Any:
    if depth >= 3:
        return _clip(str(value), 240)
    if isinstance(value, str):
        return _clip(value, text_limit)
    if isinstance(value, dict):
        return {str(key): _compact_value(item, depth=depth + 1, text_limit=text_limit) for key, item in list(value.items())[:12]}
    if isinstance(value, (list, tuple)):
        return [_compact_value(item, depth=depth + 1, text_limit=text_limit) for item in list(value)[:16]]
    return value


def _flatten(batches: list[ProtocolBatch]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for batch in batches:
        result.extend(batch.messages())
    return result


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)] + "…"


__all__ = [
    "ProtocolBatch",
    "compact_protocol_messages",
    "extract_protocol_messages",
    "group_protocol_batches",
    "protocol_tool_message",
    "protocol_tool_result_view",
]

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
from app.runtime.tool_execution_cycle import ExecutionOutcome


@dataclass(frozen=True, slots=True)
class ProtocolBatch:
    """一个 assistant tool-call 消息及其全部 tool 响应。"""

    assistant: dict[str, Any]
    tool_messages: tuple[dict[str, Any], ...] = ()
    complete: bool = True

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
    """按 assistant tool_calls 分组，并校验 tool_call_id 基本对应关系。"""

    batches: list[ProtocolBatch] = []
    index = 0
    while index < len(messages):
        assistant = deepcopy(messages[index])
        tool_calls = assistant.get("tool_calls") if assistant.get("role") == "assistant" else None
        if isinstance(tool_calls, list) and tool_calls:
            expected_ids = {
                str(item.get("id"))
                for item in tool_calls
                if isinstance(item, dict) and item.get("id")
            }
            tools: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            cursor = index + 1
            while cursor < len(messages) and messages[cursor].get("role") == "tool":
                tool_message = messages[cursor]
                tool_id = str(tool_message.get("tool_call_id", ""))
                if not tool_id or tool_id not in expected_ids or tool_id in seen_ids:
                    break
                tools.append(deepcopy(tool_message))
                seen_ids.add(tool_id)
                cursor += 1
            complete = bool(expected_ids) and seen_ids == expected_ids
            batches.append(ProtocolBatch(assistant=assistant, tool_messages=tuple(tools), complete=complete))
            index = cursor
            continue
        if assistant.get("role") == "tool":
            # 没有紧邻合法 assistant tool_calls 的 tool 消息不能进入协议历史。
            index += 1
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
    # 不完整批次整体丢弃，避免 assistant + 部分 tool 或孤立 tool 进入 Provider。
    compacted = [_compact_batch(batch) for batch in batches if batch.complete]
    budget = max(1, max_tokens)
    while len(compacted) > 1 and estimate_tokens(_flatten(compacted)) > budget:
        compacted.pop(0)
    if compacted and estimate_tokens(_flatten(compacted)) > budget:
        compacted = [_compact_batch(batch, text_limit=320) for batch in compacted[-1:]]
    return _flatten(compacted)


def protocol_tool_result_view(result: ToolResult | ExecutionOutcome | dict[str, Any]) -> dict[str, Any]:
    """构造发送给模型的轻量 ToolResult View，不改变原始结果。"""

    if isinstance(result, ExecutionOutcome):
        raw = result.result.model_dump(mode="json")
        raw.update(
            {
                "accepted": result.accepted,
                "verified": result.verified,
                "verification_problems": list(result.verification_problems),
                "recovery_action": result.recovery_action.value if result.recovery_action else None,
                "directive": result.directive.value,
                "attempts": result.attempts,
            }
        )
    elif isinstance(result, ToolResult):
        raw = result.model_dump(mode="json")
    else:
        raw = dict(result)
    view = {
        "call_id": raw.get("call_id"),
        "status": raw.get("status"),
        "error": _compact_value(raw.get("error")),
        "warnings": _compact_value(raw.get("warnings", [])),
        "datasets": list(raw.get("datasets") or [])[:32],
        "artifacts": list(raw.get("artifacts") or [])[:32],
        "retryable": raw.get("retryable", False),
        "output": _compact_value(raw.get("output")),
        "accepted": raw.get("accepted"),
        "verified": raw.get("verified"),
        "verification_problems": _compact_value(raw.get("verification_problems", [])),
        "recovery_action": raw.get("recovery_action"),
        "directive": raw.get("directive"),
        "attempts": raw.get("attempts"),
        "rationale": _compact_value(raw.get("rationale")),
    }
    return {key: value for key, value in view.items() if value not in (None, "", [], {})}


def protocol_tool_message(result: ToolResult | ExecutionOutcome | dict[str, Any]) -> dict[str, Any]:
    """把执行结论变成合法且紧凑的 role=tool 消息。"""

    view = protocol_tool_result_view(result)

    return {
        "role": "tool",
        "tool_call_id": view.get("call_id"),
        "content": json.dumps(view, ensure_ascii=False, separators=(",", ":")),
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

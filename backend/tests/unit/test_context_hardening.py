import json

from app.core.models import AgentRequest, RunBudget, ToolResult, ToolStatus
from app.runtime.context_manager import ContextManager
from app.runtime.model_input_budget import ModelInputBudget
from app.runtime.protocol_history import compact_protocol_messages, extract_protocol_messages


def _tool_batches(count: int, output_size: int = 40) -> list[dict]:
    messages = []
    for index in range(count):
        call_id = f"call-{index}"
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "inspect", "arguments": "{}"}}],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps({"status": "SUCCESS", "output": "x" * output_size}),
            }
        )
    return messages


def test_protocol_history_keeps_recent_complete_batches():
    compacted = compact_protocol_messages(_tool_batches(6), max_tokens=100_000)

    assert len(compacted) == 8
    assert [item["tool_call_id"] for item in compacted if item["role"] == "tool"] == ["call-2", "call-3", "call-4", "call-5"]
    for index, item in enumerate(compacted):
        if item["role"] == "tool":
            assert compacted[index - 1]["role"] == "assistant"
            assert compacted[index - 1]["tool_calls"]


def test_protocol_history_compacts_tool_output_without_mutating_source():
    source = _tool_batches(2, output_size=10_000)
    original = json.dumps(source, ensure_ascii=False, sort_keys=True)

    compacted = compact_protocol_messages(source, max_tokens=260)

    assert json.dumps(source, ensure_ascii=False, sort_keys=True) == original
    assert compacted
    assert all(item["role"] != "tool" or len(item["content"]) < 10_000 for item in compacted)


def test_old_checkpoint_messages_are_converted_to_protocol_history():
    messages = [
        {"role": "system", "content": "系统"},
        {"role": "user", "content": "动态上下文"},
        *_tool_batches(1),
    ]

    restored = extract_protocol_messages(legacy_messages=messages)

    assert restored == messages[2:]


def test_model_input_budget_is_separate_from_output_budget():
    run_budget = RunBudget(max_tokens=900, model_input_tokens=8_000, model_context_tokens=4_000, protocol_history_tokens=2_000)
    input_budget = ModelInputBudget(
        input_tokens=run_budget.model_input_tokens,
        context_tokens=run_budget.model_context_tokens,
        protocol_tokens=run_budget.protocol_history_tokens,
    )

    assert run_budget.max_tokens == 900
    assert input_budget.input_tokens == 8_000
    assert input_budget.context_tokens == 4_000
    assert input_budget.protocol_tokens == 2_000


def test_context_overflow_metadata_keeps_required_sections():
    context = ContextManager(max_tokens=128).main_context(
        AgentRequest(user_input="请分析" + "很长" * 500),
        [],
        None,
        [],
    )

    assert context["context_meta"]["over_budget"] is True
    assert context["context_meta"]["required_tokens"] > 0
    assert context["context_meta"]["overflow_tokens"] > 0
    assert {"user_request", "request_frame", "task_goal", "request_resources", "working_memory"} <= context.keys()


def test_tool_result_view_is_small_and_preserves_status():
    result = ToolResult(call_id="call-1", status=ToolStatus.SUCCESS, output={"text": "a" * 20_000})
    compacted = compact_protocol_messages(
        [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1"}]},
            {"role": "tool", "tool_call_id": "call-1", "content": json.dumps(result.model_dump(mode="json"))},
        ],
        max_tokens=2_000,
    )

    payload = json.loads(compacted[1]["content"])
    assert payload["status"] == ToolStatus.SUCCESS.value
    assert len(compacted[1]["content"]) < 5_000

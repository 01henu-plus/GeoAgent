import json

import pytest

from app.core.models import (
    AgentRequest,
    LoopDirective,
    RequestResources,
    Run,
    RunBudget,
    ToolResult,
    ToolStatus,
)
from app.runtime.budget import BudgetExceeded
from app.runtime.context_manager import ContextManager
from app.runtime.model_input_budget import ModelInputBudget
from app.runtime.protocol_history import (
    compact_protocol_messages,
    extract_protocol_messages,
    group_protocol_batches,
    protocol_tool_message,
)
from app.runtime.tool_execution_cycle import ExecutionOutcome


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


def test_fixed_input_cost_overflow_has_no_fake_context_space():
    budget = ModelInputBudget(input_tokens=10, context_tokens=6000, protocol_tokens=3000)

    allocation = budget.allocate("x" * 500, [{"schema": "y" * 500}], _tool_batches(1, output_size=500))

    assert allocation.over_budget is True
    assert allocation.overflow_tokens > 0
    assert allocation.available_context_tokens == 0


def test_context_meta_and_compatibility_truncated_flag_are_identical():
    context = ContextManager(max_tokens=128).main_context(AgentRequest(user_input="x" * 500), [], None, [])

    assert context["truncated"] == context["context_meta"]["truncated"]


def test_multi_tool_protocol_batch_is_kept_or_removed_as_a_whole():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "one", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "two", "arguments": "{}"}},
                {"id": "c", "type": "function", "function": {"name": "three", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": "{}"},
        {"role": "tool", "tool_call_id": "b", "content": "{}"},
        {"role": "tool", "tool_call_id": "c", "content": "{}"},
    ]

    batches = group_protocol_batches(messages)
    compacted = compact_protocol_messages(messages, max_tokens=10_000)

    assert batches[0].complete is True
    assert [item["tool_call_id"] for item in compacted[1:]] == ["a", "b", "c"]


def test_mismatched_tool_call_id_does_not_form_a_valid_batch():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "expected", "type": "function", "function": {"name": "one", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "other", "content": "{}"},
    ]

    batches = group_protocol_batches(messages)
    compacted = compact_protocol_messages(messages, max_tokens=10_000)

    assert batches[0].complete is False
    assert compacted == []


def test_protocol_tool_message_exposes_execution_acceptance_and_verification():
    outcome = ExecutionOutcome(
        result=ToolResult(call_id="call-bad", status=ToolStatus.SUCCESS, datasets=["missing"]),
        verified=False,
        verification_problems=["结果不可读"],
        recovery_action=None,
        attempts=1,
        accepted=False,
        directive=LoopDirective.ABORT,
    )

    message = protocol_tool_message(outcome)
    payload = json.loads(message["content"])

    assert message["tool_call_id"] == "call-bad"
    assert payload["status"] == ToolStatus.SUCCESS.value
    assert payload["accepted"] is False
    assert payload["verified"] is False
    assert payload["verification_problems"] == ["结果不可读"]
    assert payload["directive"] == LoopDirective.ABORT.value


def test_main_agent_rejects_fixed_cost_input_overflow(application):
    application.main_agent.budget = RunBudget(
        model_input_tokens=128,
        model_context_tokens=128,
        protocol_history_tokens=128,
    )
    task = application.task_service.create("预算测试", conversation_id="budget-test")
    request = AgentRequest(user_input="检查数据", conversation_id=task.conversation_id)

    current_run = Run(task_id=task.id, conversation_id=task.conversation_id, agent_id="main")
    with pytest.raises(BudgetExceeded, match="MODEL_INPUT_BUDGET_EXCEEDED"):
        application.main_agent._build_model_messages(
            request,
            current_run,
            task,
            [],
            None,
            None,
            None,
            [],
            working_memory=None,
            request_resources=RequestResources(),
            current_observation=None,
        )

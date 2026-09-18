import json

from fastapi.testclient import TestClient

from app.api import create_app
from app.core.models import (
    AgentRequest,
    AgentResult,
    AgentResultStatus,
    ConversationMemoryEntry,
    InteractionMode,
    RequestFrame,
    Run,
    RunStatus,
    Task,
    WorkingMemory,
)


def _register(client: TestClient, username: str) -> dict:
    response = client.post("/api/v1/auth/register", json={"username": username, "password": "password123", "display_name": username})
    assert response.status_code == 200, response.text
    return response.json()


def _frame(goal: str = "整理当前会话背景") -> RequestFrame:
    return RequestFrame(mode=InteractionMode.NEW_TASK, goal=goal)


def test_conversation_memory_survives_task_boundary_and_isolated_by_conversation(application):
    with TestClient(create_app(application)) as client_a, TestClient(create_app(application)) as client_b:
        user_a = _register(client_a, "conversation-a")
        user_b = _register(client_b, "conversation-b")
        conversation_a = application.conversations.create("会话 A", user_id=user_a["id"])
        conversation_b = application.conversations.create("会话 B", user_id=user_b["id"])
        task_a = Task(goal="任务 A", conversation_id=conversation_a.id)
        application.store.save_task(task_a)
        run_a = Run(conversation_id=conversation_a.id, task_id=task_a.id, agent_id="main", status=RunStatus.COMPLETED)
        request = AgentRequest(user_id=user_a["id"], conversation_id=conversation_a.id, user_input="这次会话后续都以 2020 年为基准")
        memory = application.conversation_memory.apply_request(request, _frame(), task_a, run_a)
        assert memory is not None
        application.conversation_memory.apply_result(request, run_a, AgentResult(agent_id="main", task_id=task_a.id, status=AgentResultStatus.SUCCESS, summary="完成", trace_id=run_a.id))
        task_b = Task(goal="任务 B", conversation_id=conversation_a.id)
        application.store.save_task(task_b)
        run_b = Run(conversation_id=conversation_a.id, task_id=task_b.id, agent_id="main", status=RunStatus.COMPLETED)
        continued = application.conversation_memory.get(conversation_a.id, user_a["id"])
        assert continued is not None
        assert any("2020" in item.content for item in continued.key_facts)
        assert application.store.get_conversation_memory_for_user(conversation_b.id, user_b["id"]) is None
        assert application.store.get_working_memory(task_a.id) is None
        application.store.save_working_memory(WorkingMemory(task_id=task_b.id, conversation_id=conversation_a.id))
        assert application.store.get_working_memory(task_b.id) is not None
        assert application.store.get_working_memory(task_a.id) is None
        assert run_b.task_id == task_b.id


def test_conversation_memory_deduplicates_results_and_resolves_blocked_topic(application):
    with TestClient(create_app(application)) as client:
        user = _register(client, "conversation-dedupe")
        conversation = application.conversations.create("去重", user_id=user["id"])
        task = Task(goal="字段选择", conversation_id=conversation.id)
        application.store.save_task(task)
        run = Run(conversation_id=conversation.id, task_id=task.id, agent_id="main", status=RunStatus.WAITING_USER)
        request = AgentRequest(user_id=user["id"], conversation_id=conversation.id, user_input="请选择人口字段")
        blocked = AgentResult(agent_id="main", task_id=task.id, status=AgentResultStatus.BLOCKED, summary="请选择人口字段", error="WAITING_USER", trace_id=run.id)
        application.conversation_memory.apply_result(request, run, blocked)
        application.conversation_memory.apply_result(request, run, blocked)
        first = application.conversation_memory.get(conversation.id, user["id"])
        assert first is not None
        assert len(first.unresolved_topics) == 1
        success_run = run.model_copy(update={"id": "run-resolved", "status": RunStatus.COMPLETED})
        success = AgentResult(agent_id="main", task_id=task.id, status=AgentResultStatus.SUCCESS, summary="已完成", datasets=["ds-final"], trace_id=success_run.id)
        application.conversation_memory.apply_result(request, success_run, success)
        application.conversation_memory.apply_result(request, success_run, success)
        resolved = application.conversation_memory.get(conversation.id, user["id"])
        assert resolved is not None
        assert resolved.unresolved_topics == []
        assert len([item for item in resolved.important_references if item.reference_id == "ds-final"]) == 1

        error_run = run.model_copy(update={"id": "run-crs", "status": RunStatus.FAILED})
        error = AgentResult(agent_id="main", task_id=task.id, status=AgentResultStatus.FAILED, summary="CRS 错误", error="CRS_MISSING", trace_id=error_run.id)
        application.conversation_memory.apply_result(request, error_run, error)
        after_error = application.conversation_memory.get(conversation.id, user["id"])
        assert after_error is not None
        assert after_error.unresolved_topics == []


def test_delete_conversation_removes_conversation_memory(application):
    with TestClient(create_app(application)) as client:
        user = _register(client, "conversation-delete")
        conversation = application.conversations.create("删除", user_id=user["id"])
        run = Run(conversation_id=conversation.id, agent_id="main", status=RunStatus.COMPLETED)
        request = AgentRequest(user_id=user["id"], conversation_id=conversation.id, user_input="这次会话后续都使用 GeoJSON")
        application.conversation_memory.apply_request(request, _frame(), None, run)
        assert application.store.get_conversation_memory(conversation.id) is not None
        assert application.conversations.delete(conversation.id, user_id=user["id"])
        assert application.store.get_conversation_memory(conversation.id) is None


def test_model_context_contains_profile_conversation_project_and_working_memory(application):
    with TestClient(create_app(application)) as client:
        user = _register(client, "context-owner")
        conversation = application.conversations.create("上下文", user_id=user["id"])
        task = Task(goal="当前任务", conversation_id=conversation.id)
        application.store.save_task(task)
        run = Run(conversation_id=conversation.id, task_id=task.id, agent_id="main", status=RunStatus.RUNNING)
        application.store.save_run(run)
        application.profile.update(user["id"], {"response_style": "concise"})
        application.memory.set("project_default_crs", "EPSG:32651", user_id=user["id"])
        memory = application.conversation_memory.get_or_create(conversation.id, user["id"]).model_copy(update={"key_facts": [ConversationMemoryEntry(content="研究区域=上海")]})
        application.store.save_conversation_memory(memory)
        working_memory = WorkingMemory(task_id=task.id, conversation_id=conversation.id, active_dataset_ids=["dem-a"])
        application.store.save_working_memory(working_memory)
        payload_text = application.main_agent._model_user_message(
            AgentRequest(user_id=user["id"], conversation_id=conversation.id, user_input="默认 CRS 是什么"),
            run,
            [],
            None,
            None,
            working_memory=working_memory,
        )
        payload = json.loads(payload_text.split("\n", 1)[1])
        assert payload["user_profile"]["response_style"] == "concise"
        assert payload["conversation_memory"]["key_facts"][0]["content"] == "研究区域=上海"
        assert payload["project_memory"][0]["key"] == "project_default_crs"
        assert payload["working_memory"]["active_dataset_ids"] == ["dem-a"]
        assert "messages" not in payload["conversation_memory"]

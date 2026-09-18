"""ConversationMemory 的确定性更新策略。

Messages 仍然是会话原始事实来源；本服务只维护有界、可去重的结构化便利视图。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import UTC, datetime

from app.core.models import (
    AgentRequest,
    AgentResult,
    AgentResultStatus,
    ConversationMemory,
    ConversationMemoryEntry,
    RequestFrame,
    Run,
    Task,
)
from app.state import StateStore


class ConversationMemoryService:
    MAX_ENTRIES = 20
    MAX_CONTENT_LENGTH = 500
    MAX_SUMMARY_LENGTH = 1800
    _CONVERSATION_MARKERS = ("这次会话", "本次会话", "当前会话", "这个对话", "会话后续", "本次分析")
    _SENSITIVE_MARKERS = ("密码", "password", "api_key", "api key", "token", "session", "authorization", "bearer")

    def __init__(self, store: StateStore) -> None:
        self.store = store

    def get(self, conversation_id: str, user_id: str) -> ConversationMemory | None:
        return self.store.get_conversation_memory_for_user(conversation_id, user_id)

    def get_or_create(self, conversation_id: str, user_id: str) -> ConversationMemory:
        memory = self.get(conversation_id, user_id)
        if memory is not None:
            return memory
        if self.store.get_conversation_for_user(conversation_id, user_id) is None:
            raise PermissionError("当前用户无权访问该会话")
        memory = ConversationMemory(conversation_id=conversation_id, user_id=user_id)
        self.store.save_conversation_memory(memory)
        return memory

    def apply_request(self, request: AgentRequest, frame: RequestFrame | None, task: Task | None, run: Run) -> ConversationMemory | None:
        if not request.user_id:
            return None
        memory = self.get_or_create(request.conversation_id, request.user_id)
        source_message_id = self._source_message_id(request)
        additions = self._request_entries(request.user_input, frame, task, run, source_message_id)
        if not additions:
            return memory
        updated = self._add_entries(memory, key_facts=additions[0], decisions=additions[1])
        self.store.save_conversation_memory(updated)
        return updated

    def apply_result(self, request: AgentRequest, run: Run, result: AgentResult) -> ConversationMemory | None:
        if not request.user_id:
            return None
        memory = self.get_or_create(request.conversation_id, request.user_id)
        references = [
            ConversationMemoryEntry(
                content=f"本次任务生成数据集：{dataset_id}",
                source_task_id=run.task_id,
                source_run_id=run.id,
                reference_type="dataset",
                reference_id=dataset_id,
            )
            for dataset_id in result.datasets
        ] + [
            ConversationMemoryEntry(
                content=f"本次任务生成结果文件：{artifact_id}",
                source_task_id=run.task_id,
                source_run_id=run.id,
                reference_type="artifact",
                reference_id=artifact_id,
            )
            for artifact_id in result.artifacts
        ]
        unresolved = []
        if result.status is AgentResultStatus.BLOCKED and result.error in {"WAITING_USER", "NEEDS_CLARIFICATION"}:
            unresolved.append(
                ConversationMemoryEntry(
                    content=_clip(result.summary),
                    source_task_id=run.task_id,
                    source_run_id=run.id,
                )
            )
        unresolved_topics = memory.unresolved_topics
        if result.status in {AgentResultStatus.SUCCESS, AgentResultStatus.PARTIAL} and run.task_id:
            unresolved_topics = [item for item in unresolved_topics if item.source_task_id != run.task_id]
        updated = self._add_entries(memory, important_references=references, unresolved_topics=unresolved, unresolved_topics_override=unresolved_topics)
        self.store.save_conversation_memory(updated)
        return updated

    def _request_entries(
        self,
        text: str,
        frame: RequestFrame | None,
        task: Task | None,
        run: Run,
        source_message_id: str | None,
    ) -> tuple[list[ConversationMemoryEntry], list[ConversationMemoryEntry]]:
        normalized = _clip(text)
        folded = normalized.casefold()
        if not normalized or any(marker in folded for marker in self._SENSITIVE_MARKERS):
            return [], []
        if not any(marker in normalized for marker in self._CONVERSATION_MARKERS):
            return [], []
        entry = ConversationMemoryEntry(content=normalized, source_message_id=source_message_id, source_task_id=task.id if task else run.task_id, source_run_id=run.id)
        if any(marker in normalized for marker in ("决定", "选择", "改为", "改成")):
            return [], [entry]
        return [entry], []

    def _source_message_id(self, request: AgentRequest) -> str | None:
        messages = self.store.list_messages(request.conversation_id, limit=1000)
        for message in reversed(messages):
            if message.role == "user" and message.content == request.user_input:
                return message.id
        return None

    def _add_entries(
        self,
        memory: ConversationMemory,
        *,
        key_facts: Iterable[ConversationMemoryEntry] = (),
        decisions: Iterable[ConversationMemoryEntry] = (),
        important_references: Iterable[ConversationMemoryEntry] = (),
        unresolved_topics: Iterable[ConversationMemoryEntry] = (),
        unresolved_topics_override: list[ConversationMemoryEntry] | None = None,
    ) -> ConversationMemory:
        facts = _append_entries(memory.key_facts, key_facts)
        decisions_list = _append_entries(memory.decisions, decisions)
        references = _append_entries(memory.important_references, important_references)
        unresolved_base = unresolved_topics_override if unresolved_topics_override is not None else memory.unresolved_topics
        unresolved = _append_entries(unresolved_base, unresolved_topics)
        updated = memory.model_copy(update={
            "key_facts": facts[-self.MAX_ENTRIES :],
            "decisions": decisions_list[-self.MAX_ENTRIES :],
            "important_references": references[-self.MAX_ENTRIES :],
            "unresolved_topics": unresolved[-self.MAX_ENTRIES :],
            "updated_at": datetime.now(UTC),
        })
        return updated.model_copy(update={"summary": _summary(updated)})


def _append_entries(current: Iterable[ConversationMemoryEntry], incoming: Iterable[ConversationMemoryEntry]) -> list[ConversationMemoryEntry]:
    result = list(current)
    for item in incoming:
        if not item.content.strip():
            continue
        if any(_same_entry(existing, item) for existing in result):
            continue
        result.append(item.model_copy(update={"content": _clip(item.content)}))
    return result


def _same_entry(left: ConversationMemoryEntry, right: ConversationMemoryEntry) -> bool:
    if left.content.casefold() == right.content.casefold():
        return True
    if left.reference_type and right.reference_type and left.reference_id and left.reference_id == right.reference_id and left.reference_type == right.reference_type:
        return True
    return bool(left.source_run_id and left.source_run_id == right.source_run_id and left.content.casefold() == right.content.casefold())


def _summary(memory: ConversationMemory) -> str:
    sections = []
    if memory.key_facts:
        sections.append("关键事实：" + "；".join(item.content for item in memory.key_facts[-5:]))
    if memory.decisions:
        sections.append("已作决定：" + "；".join(item.content for item in memory.decisions[-5:]))
    if memory.unresolved_topics:
        sections.append("待确认：" + "；".join(item.content for item in memory.unresolved_topics[-5:]))
    return _clip("。".join(sections), 1800)


def _clip(value: str, limit: int = ConversationMemoryService.MAX_CONTENT_LENGTH) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return value[:limit]


__all__ = ["ConversationMemoryService"]

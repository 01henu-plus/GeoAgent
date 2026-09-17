"""Conversation 与 Main Agent 的业务入口。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from app.core.models import AgentRequest, AgentResult, Conversation, Message, Run, new_id
from app.run import RunManager
from app.state import StateStore


class ConversationService:
    def __init__(self, store: StateStore, run_manager: RunManager) -> None:
        self.store = store
        self.run_manager = run_manager

    def ensure(self, conversation_id: str, title: str) -> None:
        self.store.upsert_conversation(conversation_id, title, datetime.now(UTC).isoformat())

    def create(self, title: str = "新对话") -> Conversation:
        return self.store.create_conversation(title)

    def list(self, limit: int = 50) -> list[Conversation]:
        return self.store.list_conversations(limit)

    def delete(self, conversation_id: str) -> bool:
        return self.store.delete_conversation(conversation_id)

    async def ask(self, request: AgentRequest) -> AgentResult:
        run = await self.submit(request)
        return await self.wait(run.id)

    async def submit(self, request: AgentRequest, *, on_model_delta: Callable[[str], Awaitable[None]] | None = None) -> Run:
        self._save_user_message(request)
        return await self.run_manager.submit(request, on_model_delta=on_model_delta)

    async def wait(self, run_id: str) -> AgentResult:
        result = await self.run_manager.wait(run_id)
        run = self.store.get_run(run_id)
        if run and run.conversation_id:
            messages = self.store.list_messages(run.conversation_id, limit=1000)
            if not any(message.role == "assistant" and message.run_id == result.trace_id for message in messages):
                self.ensure(run.conversation_id, "GeoAgent resumed run")
                self.store.save_message(Message(id=new_id("msg"), conversation_id=run.conversation_id, role="assistant", content=_assistant_text(result), run_id=result.trace_id))
        return result

    def _save_user_message(self, request: AgentRequest) -> None:
        self.ensure(request.conversation_id, request.user_input[:40])
        self.store.save_message(Message(id=new_id("msg"), conversation_id=request.conversation_id, role="user", content=request.user_input))

def _assistant_text(result: AgentResult) -> str:
    return result.summary

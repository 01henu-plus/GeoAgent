"""进程内异步事件总线。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from app.core.models import TraceEvent

EventHandler = Callable[[TraceEvent], Awaitable[None]]


class EventBus:
    def __init__(self) -> None:
        self._handlers: list[EventHandler] = []

    def subscribe(self, handler: EventHandler) -> None:
        self._handlers.append(handler)

    def unsubscribe(self, handler: EventHandler) -> None:
        self._handlers.remove(handler)

    async def publish(self, event: TraceEvent) -> None:
        for handler in tuple(self._handlers):
            await handler(event)

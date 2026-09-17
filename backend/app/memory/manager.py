"""Working / Project Memory 的轻量持久化门面。

Memory 与当前 Run 的 state 分开保存。离线模式不做未经确认的自动写入，
但会按当前问题对已有记忆做有限召回，避免把所有历史 key/value 无差别放进
模型上下文。
"""

from __future__ import annotations

import re

from app.core.models import MemoryItem
from app.state import StateStore


class MemoryManager:
    def __init__(self, store: StateStore) -> None:
        self.store = store

    def set(self, key: str, value: str, *, scope: str = "project", metadata: dict | None = None) -> MemoryItem:
        item = MemoryItem(scope=scope, key=key, value=value, metadata=metadata or {})
        self.store.save_memory(item)
        return item

    def get(self, key: str, *, scope: str = "project") -> MemoryItem | None:
        return next((item for item in self.store.list_memories(scope) if item.key == key), None)

    def list(self, scope: str = "project") -> list[MemoryItem]:
        return self.store.list_memories(scope)

    def recall(self, query: str, *, scope: str = "project", limit: int = 5) -> list[MemoryItem]:
        """按简单词项重合召回相关记忆；没有命中时不返回全部记忆。"""

        if limit < 1:
            return []
        query_terms = _terms(query)
        scored: list[tuple[int, MemoryItem]] = []
        for item in self.list(scope):
            terms = _terms(f"{item.key} {item.value} {item.metadata}")
            score = len(query_terms.intersection(terms))
            if item.key.casefold() in query.casefold():
                score += 3
            if score:
                scored.append((score, item))
        scored.sort(key=lambda pair: (-pair[0], -pair[1].updated_at.timestamp()))
        return [item for _, item in scored[:limit]]


def _terms(value: str) -> set[str]:
    normalized = value.casefold()
    tokens = set(re.findall(r"[a-z0-9_:.+-]+", normalized))
    for sequence in re.findall(r"[\u3400-\u9fff]+", normalized):
        tokens.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
        tokens.update(sequence)
    return {token for token in tokens if token}


__all__ = ["MemoryManager"]

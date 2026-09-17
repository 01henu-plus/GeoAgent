"""可选模型适配器注册表。"""

from __future__ import annotations

from .adapter import ModelAdapter


class ModelRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, ModelAdapter] = {}

    def register(self, name: str, adapter: ModelAdapter) -> None:
        self._adapters[name] = adapter

    def get(self, name: str) -> ModelAdapter | None:
        return self._adapters.get(name)


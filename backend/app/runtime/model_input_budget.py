"""ModelRequest 输入预算的轻量计算。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.runtime.context_assembler import estimate_tokens


@dataclass(frozen=True, slots=True)
class ModelInputBudget:
    """区分模型输入、动态 Context、协议历史和模型输出预算。"""

    input_tokens: int = 12000
    context_tokens: int = 6000
    protocol_tokens: int = 3000

    def available_context_tokens(self, system: Any, tools: Any, protocol: Any, *, overhead: Any = None) -> int:
        fixed = estimate_tokens(system) + estimate_tokens(tools) + estimate_tokens(protocol) + estimate_tokens(overhead)
        return max(128, min(self.context_tokens, self.input_tokens - fixed))

    def estimate_request(self, system: Any, dynamic_context: Any, protocol: Any, tools: Any) -> int:
        return estimate_tokens({"system": system, "dynamic_context": dynamic_context, "protocol": protocol, "tools": tools})


__all__ = ["ModelInputBudget"]

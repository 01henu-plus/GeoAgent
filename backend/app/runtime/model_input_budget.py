"""ModelRequest 输入预算的轻量计算。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.runtime.context_assembler import estimate_tokens


@dataclass(frozen=True, slots=True)
class ModelInputAllocation:
    """一次 ModelRequest 的输入预算分配结果。"""

    input_tokens: int
    fixed_tokens: int
    available_context_tokens: int
    over_budget: bool
    overflow_tokens: int


@dataclass(frozen=True, slots=True)
class ModelInputBudget:
    """区分模型输入、动态 Context、协议历史和模型输出预算。"""

    input_tokens: int = 12000
    context_tokens: int = 6000
    protocol_tokens: int = 3000

    def allocate(self, system: Any, tools: Any, protocol: Any, *, overhead: Any = None) -> ModelInputAllocation:
        fixed = self.fixed_tokens(system, tools, protocol, overhead=overhead)
        overflow = max(0, fixed - self.input_tokens)
        return ModelInputAllocation(
            input_tokens=self.input_tokens,
            fixed_tokens=fixed,
            available_context_tokens=max(0, min(self.context_tokens, self.input_tokens - fixed)),
            over_budget=overflow > 0,
            overflow_tokens=overflow,
        )

    def fixed_tokens(self, system: Any, tools: Any, protocol: Any, *, overhead: Any = None) -> int:
        return estimate_tokens(system) + estimate_tokens(tools) + estimate_tokens(protocol) + estimate_tokens(overhead)

    def available_context_tokens(self, system: Any, tools: Any, protocol: Any, *, overhead: Any = None) -> int:
        """兼容旧调用；固定成本超限时返回 0，不伪造最小 Context 空间。"""

        return self.allocate(system, tools, protocol, overhead=overhead).available_context_tokens

    def estimate_request(self, system: Any, dynamic_context: Any, protocol: Any, tools: Any) -> int:
        return estimate_tokens({"system": system, "dynamic_context": dynamic_context, "protocol": protocol, "tools": tools})


__all__ = ["ModelInputAllocation", "ModelInputBudget"]

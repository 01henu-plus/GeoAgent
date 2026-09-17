"""Agent 局部 Context。"""

from dataclasses import dataclass, field
from typing import Any

from app.core.models import Dataset


@dataclass
class SubAgentContext:
    subtask: dict[str, Any]
    datasets: list[Dataset]
    parent_findings: list[Any] = field(default_factory=list)
    allowed_tools: tuple[str, ...] = ("dataset.inspect", "vector.validate", "analysis.distance", "raster.inspect", "raster.slope")


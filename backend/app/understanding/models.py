"""请求理解过程中的中间模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.models import ResolvedReference


class RequestUnderstanding(BaseModel):
    """引用解析结果，引用本身必须由状态仓库中的对象支持。"""

    references: list[ResolvedReference] = Field(default_factory=list)
    unresolved_references: list[str] = Field(default_factory=list)


"""把 API/CLI 输入规整成 AgentRequest。"""

from __future__ import annotations

from collections.abc import Iterable

from app.core.models import AgentRequest, new_id


def normalize_request(
    user_input: str | AgentRequest,
    *,
    conversation_id: str | None = None,
    dataset_ids: Iterable[str] = (),
    model_profile: str | None = None,
) -> AgentRequest:
    if isinstance(user_input, AgentRequest):
        return user_input
    return AgentRequest(
        user_input=user_input,
        conversation_id=conversation_id or new_id("conv"),
        dataset_ids=[item for item in dataset_ids if item],
        model_profile=model_profile,
    )

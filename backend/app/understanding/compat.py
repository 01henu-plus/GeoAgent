"""把 RequestFrame 适配到尚未迁移的旧 Planner/Router。"""

from __future__ import annotations

from app.core.models import AgentRequest, IntentResult, IntentType, InteractionMode, RequestFrame
from app.decision.intent import IntentResolver


class LegacyIntentAdapter:
    """旧 IntentResult 仅作为 Planner/Router 的兼容输入，不再作为主理解结果。"""

    def __init__(self, resolver: IntentResolver | None = None) -> None:
        self.resolver = resolver or IntentResolver()

    def to_intent(self, frame: RequestFrame, request: AgentRequest, datasets) -> IntentResult:
        if frame.mode in {InteractionMode.CHAT, InteractionMode.CANCEL_TASK}:
            intent = IntentType.UNKNOWN
            confidence = frame.confidence
            entities: dict[str, object] = {}
            rationale = "RequestFrame 已判定当前回合不进入 GIS 计划执行。"
        else:
            legacy = self.resolver.resolve(request.model_copy(update={"user_input": frame.goal}), datasets)
            intent = legacy.intent
            confidence = min(frame.confidence, legacy.confidence)
            entities = dict(legacy.entities)
            rationale = legacy.rationale
            if frame.mode is InteractionMode.QUERY:
                if legacy.intent is IntentType.RUN_DIAGNOSIS:
                    intent = IntentType.RUN_DIAGNOSIS
                else:
                    intent = IntentType.KNOWLEDGE_QUERY if "knowledge_lookup" in frame.capabilities else IntentType.RESULT_INTERPRETATION
            if intent is IntentType.UNKNOWN and "dataset_inspection" in frame.capabilities:
                intent = IntentType.DATA_INSPECTION
        entities.update(
            {
                "interaction_mode": frame.mode.value,
                "capabilities": list(frame.capabilities),
                "references": [item.model_dump(mode="json") for item in frame.references],
                "target_task_id": frame.target_task_id,
                "target_run_id": frame.target_run_id,
                "unresolved_references": list(frame.unresolved_references),
            }
        )
        return IntentResult(intent=intent, confidence=confidence, entities=entities, rationale=rationale)

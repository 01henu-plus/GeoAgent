"""把状态、引用和用户文本解释成结构化 RequestFrame。"""

from __future__ import annotations

import json
import re

from app.core.models import AgentRequest, Dataset, InteractionMode, RequestFrame, StateSnapshot
from app.decision.intent import IntentResolver
from app.models import ModelAdapter, ModelRequest
from app.understanding.models import ReferenceResolution
from app.understanding.rule_gate import infer_capabilities

_SYSTEM_PROMPT = """
你是 GeoAgent 的请求理解器，只负责理解用户请求，不负责规划、选工具或执行任务。
请严格输出一个 JSON 对象，字段必须符合 RequestFrame：
mode 只能是 new_task、continue_task、modify_task、retry_task、query、chat、cancel_task；
goal 是用户当前要完成的目标；references 只能引用输入状态中真实存在的对象；
capabilities 是完成目标所需能力，不要把能力写成 interaction mode；
target_task_id 和 target_run_id 只能从状态中选择，不能编造；
无法解析的上下文指代放入 unresolved_references；不要生成执行步骤。
""".strip()


class RequestInterpreter:
    """优先使用模型结构化输出，失败或离线时使用有限确定性兼容解析。"""

    def __init__(self, legacy_resolver: IntentResolver | None = None) -> None:
        self.legacy_resolver = legacy_resolver or IntentResolver()

    async def interpret(
        self,
        *,
        message: str,
        state: StateSnapshot,
        resolution: ReferenceResolution,
        model_adapter: ModelAdapter | None = None,
        datasets: list[Dataset] | None = None,
    ) -> RequestFrame:
        if model_adapter is not None and model_adapter.supports_structured_output:
            try:
                return await self._interpret_with_model(message, state, resolution, model_adapter)
            except Exception:
                # 结构化输出失败时回退到确定性解析，不能让模型异常中断请求入口。
                pass
        return self._interpret_without_model(message, state, resolution, datasets or [])

    async def _interpret_with_model(
        self,
        message: str,
        state: StateSnapshot,
        resolution: ReferenceResolution,
        model_adapter: ModelAdapter,
    ) -> RequestFrame:
        payload = {
            "message": message,
            "state": state.model_dump(mode="json"),
            "resolved_references": resolution.model_dump(mode="json"),
            "allowed_modes": [item.value for item in InteractionMode],
            "capability_vocabulary": [
                "raster_analysis",
                "vector_analysis",
                "artifact_read",
                "artifact_write",
                "dataset_inspection",
                "crs_transform",
                "python_execution",
                "knowledge_lookup",
                "result_query",
            ],
        }
        response = await model_adapter.complete(
            ModelRequest(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
                ],
                temperature=0,
                max_tokens=700,
                response_format={"type": "json_object"},
            )
        )
        return RequestFrame.model_validate_json(_strip_code_fence(response.content))

    def _interpret_without_model(
        self,
        message: str,
        state: StateSnapshot,
        resolution: ReferenceResolution,
        datasets: list[Dataset],
    ) -> RequestFrame:
        request = AgentRequest(user_input=message, conversation_id=state.conversation_id)
        legacy = self.legacy_resolver.resolve(request, datasets)
        lowered = message.casefold()
        mode = InteractionMode.NEW_TASK
        confidence = max(0.35, legacy.confidence)
        target_task_id = state.active_task_id
        target_run_id = state.active_run_id
        goal = message

        if legacy.entities.get("is_greeting"):
            mode = InteractionMode.CHAT
            target_task_id = None
            target_run_id = None
            confidence = 0.99
        elif _looks_like_cancel(lowered):
            mode = InteractionMode.CANCEL_TASK
            confidence = 0.85 if state.active_task_id else 0.35
        elif _looks_like_retry(lowered):
            mode = InteractionMode.RETRY_TASK
            target_run_id = _failed_run_id(state)
            confidence = 0.85 if target_run_id else 0.35
        elif _looks_like_continue(lowered):
            mode = InteractionMode.CONTINUE_TASK
            goal = message if message.strip() != "继续" else (state.task_goal or message)
            confidence = 0.85 if state.active_task_id else 0.35
        elif _looks_like_modify(lowered) and state.active_task_id:
            mode = InteractionMode.MODIFY_TASK
            confidence = 0.78
        elif legacy.intent.value in {"KNOWLEDGE_QUERY", "RESULT_INTERPRETATION", "RUN_DIAGNOSIS"} or legacy.entities.get("is_question"):
            mode = InteractionMode.QUERY
            target_task_id = state.active_task_id
            target_run_id = state.active_run_id
            confidence = max(confidence, 0.7)

        capabilities = infer_capabilities(message, legacy=legacy, references=resolution.references)
        if mode is InteractionMode.CHAT:
            capabilities = ["conversation"]
        if mode is InteractionMode.QUERY and not capabilities:
            capabilities = ["result_query"]
        constraints = [message] if mode is InteractionMode.MODIFY_TASK else []
        return RequestFrame(
            mode=mode,
            goal=goal,
            references=resolution.references,
            constraints=constraints,
            capabilities=capabilities,
            target_task_id=target_task_id,
            target_run_id=target_run_id,
            needs_planning=mode in {InteractionMode.NEW_TASK, InteractionMode.CONTINUE_TASK, InteractionMode.MODIFY_TASK, InteractionMode.RETRY_TASK},
            needs_tool=bool(set(capabilities) - {"conversation", "knowledge_lookup", "result_query"}),
            unresolved_references=resolution.unresolved_references,
            confidence=confidence,
        )


def _strip_code_fence(content: str) -> str:
    value = content.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.IGNORECASE | re.DOTALL)
    return value.strip()


def _looks_like_cancel(text: str) -> bool:
    return text.startswith(("停止", "取消", "算了", "结束任务", "终止"))


def _looks_like_retry(text: str) -> bool:
    return text.startswith(("再试一次", "重试", "重新来", "重新执行", "再跑一次"))


def _looks_like_continue(text: str) -> bool:
    return text == "继续" or text.startswith(("继续", "接着", "沿用刚才"))


def _looks_like_modify(text: str) -> bool:
    return text.startswith(("不对", "把", "改成", "换成", "调整")) and any(term in text for term in ("改", "换", "调整", "范围", "条件", "参数"))


def _failed_run_id(state: StateSnapshot) -> str | None:
    for run in state.recent_runs:
        if run.status.value == "FAILED" or run.error:
            return run.id
    return None

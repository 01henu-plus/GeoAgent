"""把状态、引用和用户文本解释成结构化 RequestFrame。"""

from __future__ import annotations

import json
import logging
import re

from app.core.models import Dataset, InteractionMode, RequestFrame, StateSnapshot
from app.models import ModelAdapter, ModelRequest
from app.understanding.deterministic import extract_request_hints
from app.understanding.models import ReferenceResolution
from app.understanding.patterns import (
    is_cancel_request,
    is_continue_request,
    is_modify_request,
    is_retry_request,
)
from app.understanding.rule_gate import infer_capabilities

logger = logging.getLogger(__name__)

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
            except Exception as exc:
                # 保留离线回退，但让调用方能够在日志中发现结构化理解失败。
                logger.warning(
                    "结构化 RequestInterpreter 调用失败，回退确定性解析: exception_type=%s fallback_to=deterministic",
                    type(exc).__name__,
                )
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
        hints = extract_request_hints(message, datasets)
        lowered = message.casefold()
        mode = InteractionMode.NEW_TASK
        confidence = 0.9 if hints.operations else 0.55
        target_task_id = None
        target_run_id = None
        goal = message

        if hints.is_greeting:
            mode = InteractionMode.CHAT
            target_task_id = None
            target_run_id = None
            confidence = 0.99
        elif is_cancel_request(lowered):
            mode = InteractionMode.CANCEL_TASK
            target_task_id = state.active_task_id
            target_run_id = state.active_run_id
            confidence = 0.85 if state.active_task_id else 0.35
        elif is_retry_request(lowered):
            mode = InteractionMode.RETRY_TASK
            target_run_id = _failed_run_id(state)
            confidence = 0.85 if target_run_id else 0.35
        elif is_continue_request(lowered):
            mode = InteractionMode.CONTINUE_TASK
            target_task_id = state.active_task_id
            target_run_id = state.active_run_id
            goal = message if message.strip() != "继续" else (state.task_goal or message)
            confidence = 0.85 if state.active_task_id else 0.35
        elif is_modify_request(lowered) and state.active_task_id:
            mode = InteractionMode.MODIFY_TASK
            target_task_id = state.active_task_id
            target_run_id = state.active_run_id
            confidence = 0.78
        elif not hints.operations and (hints.is_knowledge_query or hints.is_diagnosis or hints.result_reference_requested or hints.is_question):
            mode = InteractionMode.QUERY
            target_task_id = state.active_task_id
            target_run_id = state.active_run_id
            confidence = max(confidence, 0.7)

        capabilities = infer_capabilities(message, operations=hints.operations, references=resolution.references)
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


def _failed_run_id(state: StateSnapshot) -> str | None:
    for run in state.recent_runs:
        if run.status.value == "FAILED" or run.error:
            return run.id
    return None

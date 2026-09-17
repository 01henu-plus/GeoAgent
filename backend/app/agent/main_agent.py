"""GeoAgent 的 Main Agent Loop。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from app.agent.manager import AgentManager
from app.checkpoint.context import make_checkpoint
from app.checkpoint.store import CheckpointStore
from app.core.models import (
    AgentRequest,
    AgentResult,
    AgentResultStatus,
    Checkpoint,
    ErrorCategory,
    IntentResult,
    IntentType,
    InteractionMode,
    Plan,
    RequestFrame,
    RequestResources,
    Run,
    RunBudget,
    RunStatus,
    Task,
    TaskStatus,
    ToolCall,
    ToolError,
    ToolResult,
    ToolStatus,
    WorkingMemory,
    new_id,
)
from app.decision import (
    AgentRouter,
    FailureAnalyzer,
    IntentResolver,
    Planner,
    ResultVerifier,
    TaskDecomposer,
)
from app.events import EventType
from app.execution.tools import ToolExecutor
from app.gis.crs.service import CRSService
from app.knowledge import KnowledgeRetriever
from app.memory import MemoryManager
from app.memory.extractor import MemoryExtractor
from app.models import ModelAdapter, ModelRequest, ModelResponse
from app.observability import TraceRecorder
from app.run.lifecycle import PreparedRequest, RequestLifecycleBinder
from app.runtime.agent_loop import AgentLoop
from app.runtime.budget import BudgetExceeded, BudgetGuard
from app.runtime.context_manager import ContextManager
from app.runtime.lifecycle import finish_run
from app.state import StateStore, WorkingMemoryUpdater
from app.task.service import TaskService
from app.understanding.compat import LegacyIntentAdapter
from app.understanding.interpreter import RequestInterpreter
from app.understanding.pipeline import RequestUnderstandingPipeline

_MODEL_SYSTEM_PROMPT = """
你是 GeoAgent，一个面向 GIS 的中文智能助手。你必须先判断用户的真实目标，再决定本轮动作：
1. 普通问候、闲聊、概念解释或询问当前界面时，直接用中文回答，不调用工具，也不要自动跳转到结果。
2. 用户明确要求检查、查询、分析、转换或生成空间数据时，才调用工具；工具调用必须使用真实的结构化 tool_calls。
3. 先利用会话历史、用户上传的数据、数据集属性和历史运行结果理解指代关系，不要因为列表中的第一个数据集就擅自选用它。
4. 输入不完整、数据角色不明确或关键参数缺失时，先用中文追问；不要猜测距离、字段、坐标系或数据集。
5. 每轮工具返回后重新判断下一步。可以先检查数据，再根据检查结果选择后续工具，也可以停止并直接回答。
6. 只使用提供的工具和工具返回的事实，不能编造数据、文件、统计值或已完成的操作。最终回答简洁、具体、中文化。
""".strip()


class MainAgent:
    """负责用户目标、策略循环和结果汇总；确定性 GIS 计算委托给 Tools。"""

    def __init__(self, *, store: StateStore, trace: TraceRecorder, executor: ToolExecutor, registry, task_service: TaskService, agent_manager: AgentManager, settings, budget: RunBudget | None = None, checkpoint_store: CheckpointStore | None = None, memory: MemoryManager | None = None, knowledge: KnowledgeRetriever | None = None, model_adapter: ModelAdapter | None = None, model_adapters: dict[str, ModelAdapter] | None = None, default_model_profile: str | None = None, context_manager: ContextManager | None = None) -> None:
        self.store = store
        self.trace = trace
        self.executor = executor
        self.registry = registry
        self.task_service = task_service
        self.agent_manager = agent_manager
        self.settings = settings
        self.budget = budget or RunBudget()
        self.guard = BudgetGuard(self.budget)
        self.intent_resolver = IntentResolver()
        self.request_understanding = RequestUnderstandingPipeline(
            store,
            interpreter=RequestInterpreter(self.intent_resolver),
        )
        self.legacy_intent_adapter = LegacyIntentAdapter(self.intent_resolver)
        self.planner = Planner()
        self.router = AgentRouter()
        self.decomposer = TaskDecomposer()
        self.failure_analyzer = FailureAnalyzer()
        self.verifier = ResultVerifier()
        self.checkpoint_store = checkpoint_store
        self.memory = memory
        self.memory_extractor = MemoryExtractor()
        self.working_memory_updater = WorkingMemoryUpdater(store)
        self.knowledge = knowledge or KnowledgeRetriever()
        self.model_adapter = model_adapter
        self.model_adapters = model_adapters if model_adapters is not None else {}
        self.default_model_profile = default_model_profile
        self.context_manager = context_manager or ContextManager()
        self.loop = AgentLoop()
        self.lifecycle_binder = RequestLifecycleBinder(store, task_service)

    def prepare(self, request: AgentRequest, *, metadata: dict[str, object] | None = None) -> tuple[Task, Run]:
        """旧同步 API 兼容入口；正常请求必须走异步 prepare_request。"""
        frame = RequestFrame(mode=InteractionMode.NEW_TASK, goal=request.user_input, needs_planning=True)
        prepared = self.lifecycle_binder.bind(request, frame, metadata=metadata)
        if prepared.task is None or prepared.run is None:
            raise RuntimeError("无法为兼容调用创建新任务")
        return prepared.task, prepared.run

    async def prepare_request(
        self,
        request: AgentRequest,
        *,
        metadata: dict[str, object] | None = None,
        resume_from: Checkpoint | None = None,
    ) -> PreparedRequest:
        """先完成 Request Understanding，再绑定 Task/Run 生命周期。"""
        resume_state = resume_from.state if resume_from else {}
        if isinstance(resume_state.get("request"), dict):
            request = AgentRequest.model_validate(resume_state["request"])
        datasets = self._resolve_datasets(request)
        request_resources = self._resolve_request_resources(request)
        saved_frame = resume_state.get("request_frame")
        if isinstance(saved_frame, dict):
            frame = RequestFrame.model_validate(saved_frame)
        else:
            frame = await self.request_understanding.understand(
                request.conversation_id,
                request.user_input,
                request=request,
                request_resources=request_resources,
                datasets=datasets,
                model_adapter=self._model_adapter_for(request),
            )
        previous_run = self.store.get_run(resume_from.run_id) if resume_from else None
        if previous_run and previous_run.task_id and frame.mode is InteractionMode.NEW_TASK:
            frame = frame.model_copy(
                update={
                    "mode": InteractionMode.CONTINUE_TASK,
                    "target_task_id": previous_run.task_id,
                    "target_run_id": previous_run.id,
                    "needs_planning": True,
                }
            )
        return self.lifecycle_binder.bind(request, frame, request_resources=request_resources, metadata=metadata)

    async def run(
        self,
        request: AgentRequest,
        *,
        prepared: PreparedRequest | None = None,
        resume_from: Checkpoint | None = None,
        on_model_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> AgentResult:
        resume_state = resume_from.state if resume_from else {}
        prepared_request = prepared or await self.prepare_request(request, resume_from=resume_from)
        request = prepared_request.request
        task, run = prepared_request.task, prepared_request.run
        working_memory = prepared_request.working_memory
        saved_working_memory = resume_state.get("working_memory")
        if task is not None and isinstance(saved_working_memory, dict):
            restored = WorkingMemory.model_validate(saved_working_memory)
            if restored.task_id == task.id:
                self.store.save_working_memory(restored)
                working_memory = restored
        if run is None:
            raise RuntimeError("请求没有可执行的 Run")
        intent: IntentResult | None = None
        plan: Plan | None = None
        request_frame: RequestFrame | None = prepared_request.frame
        datasets = []
        phase = "created"
        await self.trace.emit(
            run.id,
            EventType.RUN_CREATED,
            "Main Agent 开始处理请求",
            payload={"request_id": request.request_id, "resumed_from": resume_from.run_id if resume_from else None},
            agent_id="main",
        )
        try:
            datasets = self._resolve_datasets(request)
            if resume_from and resume_state.get("intent") and resume_state.get("plan"):
                intent = IntentResult.model_validate(resume_state["intent"])
                plan = Plan.model_validate(resume_state["plan"])
                phase = resume_from.phase
                await self.trace.emit(run.id, EventType.RESUME_STARTED, f"从 Checkpoint 继续：{resume_from.phase}", payload={"checkpoint_id": resume_from.id, "phase": resume_from.phase}, agent_id="main")
            phase = "request_understood"
            if intent is None:
                intent = self.legacy_intent_adapter.to_intent(request_frame, request, datasets)
            await self.trace.emit(
                run.id,
                EventType.INTENT_RESOLVED,
                "请求理解完成",
                payload={
                    "request_frame": request_frame.model_dump(mode="json"),
                    "legacy_intent": intent.model_dump(mode="json"),
                    "source": "request_understanding",
                },
                agent_id="main",
            )
            if request_frame.resolution_status.value != "resolved":
                result = AgentResult(
                    agent_id="main",
                    task_id=task.id if task else run.task_id,
                    status=AgentResultStatus.BLOCKED,
                    summary="当前请求需要补充信息后才能继续。" + ("；".join(request_frame.blocking_issues) if request_frame.blocking_issues else ""),
                    error="NEEDS_CLARIFICATION",
                    trace_id=run.id,
                )
                run = finish_run(run, RunStatus.WAITING_USER, error=result.error)
                self.store.save_run(run.model_copy(update={"metadata": {**run.metadata, "result": result.model_dump(mode="json")}}))
                await self._checkpoint(run.id, "run_completed", {"status": run.status.value, "result": result.model_dump(mode="json")})
                await self.trace.emit(run.id, EventType.RUN_FAILED, result.summary, payload={"status": run.status.value, "result": result.model_dump(mode="json")}, agent_id="main")
                return result
            run = run.model_copy(update={"status": RunStatus.PLANNING})
            self.store.save_run(run)
            self.guard.check_execution_time(run)
            if isinstance(resume_state.get("delegation_result"), dict):
                result = AgentResult.model_validate(resume_state["delegation_result"]).model_copy(update={"task_id": task.id if task else run.task_id, "trace_id": run.id})
            else:
                # 参考项目的核心：模型先基于完整会话和工具结果自行判断。
                # 规则 IntentResolver/Planner 只作为无模型时的离线兜底，不能
                # 提前阻断模型，也不能把固定计划当成模型已经作出的决定。
                model_result = await self._model_loop(
                    request,
                    run,
                    task,
                    datasets,
                    intent,
                    plan,
                    request_frame=request_frame,
                    initial_messages=resume_state.get("messages") if resume_from and resume_state.get("messages") else None,
                    initial_findings=resume_state.get("model_findings") if resume_from and resume_state.get("model_findings") else None,
                    initial_dataset_ids=resume_state.get("model_dataset_ids") if resume_from and resume_state.get("model_dataset_ids") else None,
                    initial_artifact_ids=resume_state.get("model_artifact_ids") if resume_from and resume_state.get("model_artifact_ids") else None,
                    working_memory=working_memory,
                    on_model_delta=on_model_delta,
                )
                if model_result is not None:
                    result = model_result
                else:
                    if intent is None or plan is None:
                        plan = self.planner.build(request_frame.goal, intent, datasets)
                        phase = "plan_created"
                        await self.trace.emit(
                            run.id,
                            EventType.PLAN_CREATED,
                            f"生成离线兜底计划：{len(plan.steps)} 步",
                            payload={**plan.model_dump(mode="json"), "source": "offline_fallback"},
                            agent_id="main",
                        )
                        await self._checkpoint(run.id, "plan_created", self._checkpoint_state(request, intent, plan, datasets, request_frame))
                    if request_frame.mode is InteractionMode.CANCEL_TASK:
                        result = AgentResult(agent_id="main", task_id=task.id if task else run.task_id, status=AgentResultStatus.SUCCESS, summary="已识别为取消当前任务的请求。", trace_id=run.id)
                        decision = None
                    else:
                        decision = self.router.route(intent, plan, datasets)
                    if decision is None:
                        pass
                    elif decision.type.value == "DELEGATE":
                        decision = decision.model_copy(update={"subtasks": self.decomposer.decompose(request, datasets)})
                    if decision is not None:
                        await self.trace.emit(
                            run.id,
                            EventType.DECISION_MADE,
                            decision.reasoning_summary,
                            payload={**decision.model_dump(mode="json"), "source": "offline_fallback"},
                            agent_id="main",
                        )
                    if decision is None:
                        pass
                    elif decision.type.value == "ASK_USER":
                        result = AgentResult(agent_id="main", task_id=task.id if task else run.task_id, status=AgentResultStatus.BLOCKED, summary=decision.final_response or decision.reasoning_summary, error="WAITING_USER", trace_id=run.id)
                    elif decision.type.value == "DELEGATE":
                        result = await self._delegate(request, run, task, datasets, decision.subtasks, intent=intent, plan=plan, request_frame=request_frame, working_memory=working_memory)
                    elif decision.type.value == "TOOL":
                        result = await self._execute_plan(request, run, task, datasets, intent, plan, resume_state, request_frame=request_frame)
                    elif intent.intent.value == "RUN_DIAGNOSIS":
                        result = self._diagnose_runs(request, run)
                    elif intent.intent.value == "RESULT_INTERPRETATION":
                        result = self._interpret_result(request, run)
                    elif intent.intent.value == "UNKNOWN":
                        result = AgentResult(agent_id="main", task_id=task.id if task else run.task_id, status=AgentResultStatus.SUCCESS, summary=_conversation_reply(request.user_input), trace_id=run.id)
                    else:
                        findings = self.knowledge.retrieve(request.user_input) if intent.intent.value == "KNOWLEDGE_QUERY" else []
                        result = AgentResult(agent_id="main", task_id=task.id if task else run.task_id, status=AgentResultStatus.SUCCESS, summary="已整理 GIS 知识上下文。" if findings else "已理解请求，但当前计划没有需要执行的 GIS 操作。", findings=findings, trace_id=run.id, warnings=[] if findings else ["当前未配置大模型，离线模式只支持有限的 GIS 操作。"])
            final_status = _run_status_for_result(result.status, result.error)
            run = finish_run(self.store.get_run(run.id) or run, final_status, error=result.error)
            self.store.save_run(run.model_copy(update={"metadata": {**run.metadata, "result": result.model_dump(mode="json")}}))
            if self.memory:
                candidates = self.memory_extractor.extract(request, request_frame, run, result)
                self.memory.write_candidates(candidates)
            await self._checkpoint(run.id, "run_completed", {"status": final_status.value, "result": result.model_dump(mode="json")})
            if task is not None and prepared_request.action in {"create_task", "bind_task", "retry_run"}:
                self.task_service.update(task, status=_task_status_for_result(result.status), result=result.summary)
            await self.trace.emit(run.id, EventType.RUN_COMPLETED if result.status in {AgentResultStatus.SUCCESS, AgentResultStatus.PARTIAL} else EventType.RUN_FAILED, result.summary, payload={"status": result.status.value, "result": result.model_dump(mode="json")}, agent_id="main")
            return result.model_copy(update={"trace_id": run.id})
        except asyncio.CancelledError:
            run = finish_run(self.store.get_run(run.id) or run, RunStatus.CANCELLED, error="CANCELLED")
            self.store.save_run(run)
            previous_checkpoint = self.checkpoint_store.latest(run.id) if self.checkpoint_store else None
            state = {**(previous_checkpoint.state if previous_checkpoint else {}), **self._checkpoint_state(request, intent, plan, datasets, request_frame)}
            state["phase"] = phase
            await self._checkpoint(run.id, "run_cancelled", state)
            if task is not None and prepared_request.action in {"create_task", "bind_task", "retry_run"}:
                self.task_service.update(task, status=TaskStatus.CANCELLED, result="运行已取消")
            await self.trace.emit(run.id, EventType.RUN_CANCELLED, "运行已取消", payload={"status": RunStatus.CANCELLED.value}, agent_id="main")
            return AgentResult(agent_id="main", task_id=task.id if task else run.task_id, status=AgentResultStatus.CANCELLED, summary="运行已取消。", error="CANCELLED", trace_id=run.id)
        except BudgetExceeded as exc:
            run = finish_run(self.store.get_run(run.id) or run, RunStatus.BUDGET_EXCEEDED, error=str(exc))
            self.store.save_run(run)
            if task is not None and prepared_request.action in {"create_task", "bind_task", "retry_run"}:
                self.task_service.update(task, status=TaskStatus.BLOCKED, result=str(exc))
            await self.trace.emit(run.id, EventType.RUN_FAILED, str(exc), payload={"status": RunStatus.BUDGET_EXCEEDED.value}, agent_id="main")
            return AgentResult(agent_id="main", task_id=task.id if task else run.task_id, status=AgentResultStatus.BLOCKED, summary="运行因预算限制停止。", error=str(exc), trace_id=run.id)
        except Exception as exc:
            run = finish_run(self.store.get_run(run.id) or run, RunStatus.FAILED, error=str(exc))
            self.store.save_run(run)
            if task is not None and prepared_request.action in {"create_task", "bind_task", "retry_run"}:
                self.task_service.update(task, status=TaskStatus.FAILED, result=str(exc))
            await self.trace.emit(run.id, EventType.RUN_FAILED, str(exc), payload={"error_type": exc.__class__.__name__}, agent_id="main")
            return AgentResult(agent_id="main", task_id=task.id if task else run.task_id, status=AgentResultStatus.FAILED, summary="任务执行失败。", error=str(exc), trace_id=run.id)

    async def _model_loop(
        self,
        request: AgentRequest,
        run: Run,
        task: Task | None,
        datasets,
        intent: IntentResult | None,
        plan: Plan | None,
        *,
        request_frame: RequestFrame | None = None,
        initial_messages: list[dict[str, Any]] | None = None,
        initial_findings: list[Any] | None = None,
        initial_dataset_ids: list[str] | None = None,
        initial_artifact_ids: list[str] | None = None,
        working_memory: WorkingMemory | None = None,
        on_model_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> AgentResult | None:
        model_adapter = self._model_adapter_for(request)
        if model_adapter is None:
            return None
        messages = initial_messages or [
            {"role": "system", "content": _MODEL_SYSTEM_PROMPT},
            {"role": "user", "content": self._model_user_message(request, run, datasets, intent, plan, request_frame, working_memory=working_memory)},
        ]
        findings: list[Any] = list(initial_findings or [])
        dataset_ids: set[str] = set(initial_dataset_ids or [])
        artifact_ids: set[str] = set(initial_artifact_ids or [])
        for turn in range(self.budget.max_agent_turns):
            current = self.store.get_run(run.id) or run
            self.guard.check_turn(current)
            self.guard.check_execution_time(current)
            self.store.save_run(current.model_copy(update={"turn_count": current.turn_count + 1, "status": RunStatus.RUNNING}))
            content_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            input_tokens = 0
            output_tokens = 0
            model_name: str | None = None
            async for chunk in model_adapter.stream(ModelRequest(messages=messages, tools=self._model_tools(), max_tokens=self.budget.max_tokens)):
                if chunk.content:
                    content_parts.append(chunk.content)
                    if on_model_delta is not None:
                        await on_model_delta(chunk.content)
                if chunk.tool_calls:
                    tool_calls = chunk.tool_calls
                input_tokens = chunk.input_tokens or input_tokens
                output_tokens = chunk.output_tokens or output_tokens
                model_name = chunk.model or model_name
            response = ModelResponse(content="".join(content_parts), tool_calls=tool_calls, input_tokens=input_tokens, output_tokens=output_tokens, model=model_name)
            if not response.tool_calls:
                if not response.content.strip():
                    return None
                await self.trace.emit(run.id, EventType.DECISION_MADE, "模型运行时决定直接回复，不执行空间工具。", payload={"source": "model_runtime", "action": "final", "model": response.model}, agent_id="main")
                findings.append({"model": response.model, "content": response.content})
                return AgentResult(agent_id="main", task_id=task.id if task else run.task_id, status=AgentResultStatus.SUCCESS, summary=response.content.strip(), findings=findings, datasets=sorted(dataset_ids), artifacts=sorted(artifact_ids), trace_id=run.id)

            await self.trace.emit(run.id, EventType.DECISION_MADE, f"模型运行时决定调用 {len(response.tool_calls)} 个工具。", payload={"source": "model_runtime", "action": "tool", "tools": [_model_tool_name(item) for item in response.tool_calls]}, agent_id="main")
            messages.append({"role": "assistant", "content": response.content or "", "tool_calls": response.tool_calls})
            for index, raw_call in enumerate(response.tool_calls):
                call_id = str(raw_call.get("id") or f"model_call_{turn}_{index}")
                try:
                    name, arguments = _parse_model_tool_call(raw_call)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    tool_result = ToolResult(
                        call_id=call_id,
                        status=ToolStatus.FAILED,
                        error=ToolError(
                            code="INVALID_TOOL_ARGUMENTS",
                            category=ErrorCategory.INPUT,
                            message=f"模型工具参数不是有效 JSON：{exc}",
                        ),
                    )
                    name = "model.tool_call"
                else:
                    tool_result = await self._tool(run, name, arguments, call_id=call_id)
                findings.append({"tool": name, "status": tool_result.status.value, "output": tool_result.output, "error": tool_result.error.model_dump(mode="json") if tool_result.error else None})
                dataset_ids.update(tool_result.datasets)
                artifact_ids.update(tool_result.artifacts)
                messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(tool_result.model_dump(mode="json"), ensure_ascii=False, default=str)})
            await self._checkpoint(
                run.id,
                "model_tool_completed",
                {
                    **self._checkpoint_state(request, intent, plan, datasets, request_frame),
                    "messages": messages,
                    "model_findings": findings,
                    "model_dataset_ids": sorted(dataset_ids),
                    "model_artifact_ids": sorted(artifact_ids),
                },
            )
        raise BudgetExceeded("Agent turn budget exceeded")

    def _model_adapter_for(self, request: AgentRequest) -> ModelAdapter | None:
        profile_id = request.model_profile or self.default_model_profile
        if profile_id:
            adapter = self.model_adapters.get(profile_id)
            if adapter is not None:
                return adapter
        return self.model_adapter

    def _model_user_message(self, request: AgentRequest, run: Run, datasets, intent: IntentResult | None, plan: Plan | None, request_frame: RequestFrame | None = None, *, working_memory: WorkingMemory | None = None) -> str:
        from app.entry.reference_resolver import ReferenceResolver

        history = self.store.list_messages(request.conversation_id, limit=20)
        if history and history[-1].role == "user" and history[-1].content == request.user_input:
            history = history[:-1]
        conversation = [
            {"role": message.role, "content": message.content}
            for message in history
            if message.role in {"user", "assistant", "system"}
        ]
        tool_definitions = []
        for item in self.executor.registry.definitions():
            properties = item.input_schema.get("properties", {})
            tool_definitions.append(
                {
                    "name": item.name,
                    "description": item.description,
                    "parameters": list(properties) if isinstance(properties, dict) else [],
                }
            )
        memories = self.memory.recall(request.user_input, scope="project", limit=5) if self.memory else []
        referenced_runs = [
            item.model_dump(mode="json")
            for item in ReferenceResolver(self.store).resolve_runs(request.referenced_run_ids, conversation_id=request.conversation_id, exclude_run_id=run.id)
        ]
        working_memory = working_memory or (self.store.get_working_memory(run.task_id) if run.task_id else None)
        context = self.context_manager.main_context(
            request,
            datasets,
            plan,
            memories,
            conversation=conversation,
            tool_definitions=tool_definitions,
            working_memory=working_memory,
            budget=self.budget.model_dump(mode="json"),
            referenced_runs=referenced_runs,
            intent_hint=intent,
            request_frame=request_frame,
        )
        return "请先理解用户真正想完成的事情，再决定下一步。以下上下文中的 deterministic_hint 只是离线规则生成的提示，可能不准确，不能当作已经确认的意图或固定流水线。你可以直接用中文回答、询问缺失信息、调用一个或多个工具，并在每次工具返回后重新判断是否继续。只有用户明确需要数据处理或检查时才调用工具；问候、闲聊、解释概念不要调用工具。不要自行挑选不明确的数据集，不要编造工具结果。\n" + json.dumps(context, ensure_ascii=False, default=str)

    def _model_tools(self) -> list[dict[str, Any]]:
        common_schema = {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "left_dataset_id": {"type": "string"},
                "right_dataset_id": {"type": "string"},
                "mask_dataset_id": {"type": "string"},
                "distance": {"type": "number"},
                "target_crs": {"type": "string"},
                "output_path": {"type": "string"},
                "title": {"type": "string"},
                "by": {"type": "string"},
                "predicate": {"type": "string"},
                "path": {"type": "string"},
                "name": {"type": "string"},
            },
            "additionalProperties": True,
        }
        return [
            {"type": "function", "function": {"name": item.name, "description": item.description, "parameters": item.input_schema or common_schema}}
            for item in self.executor.registry.definitions()
        ]

    @staticmethod
    def _checkpoint_state(request: AgentRequest, intent: IntentResult | None, plan: Plan | None, datasets, request_frame: RequestFrame | None = None) -> dict[str, Any]:
        return {
            "request": request.model_dump(mode="json"),
            "request_frame": request_frame.model_dump(mode="json") if request_frame else None,
            "intent": intent.model_dump(mode="json") if intent else None,
            "plan": plan.model_dump(mode="json") if plan else None,
            "dataset_ids": [item.id for item in datasets],
        }

    async def _checkpoint(self, run_id: str, phase: str, state: dict[str, Any]) -> None:
        if not self.checkpoint_store:
            return
        checkpoint = make_checkpoint(run_id, phase, state)
        self.checkpoint_store.save(checkpoint)
        await self.trace.emit(run_id, EventType.CHECKPOINT_SAVED, f"保存 Checkpoint：{phase}", payload={"checkpoint_id": checkpoint.id, "phase": phase}, agent_id="main")

    async def _step_checkpoint(self, run_id: str, request: AgentRequest, intent: IntentResult, plan: Plan, datasets, completed_steps: set[str], *, request_frame: RequestFrame | None = None, **state: Any) -> None:
        await self._checkpoint(
            run_id,
            "step_completed",
            {
                **self._checkpoint_state(request, intent, plan, datasets, request_frame),
                "completed_steps": sorted(completed_steps),
                **state,
            },
        )

    async def _execute_plan(self, request: AgentRequest, run: Run, task: Task, datasets, intent: IntentResult, plan: Plan, resume_state: dict[str, Any], *, request_frame: RequestFrame | None = None) -> AgentResult:
        completed = set(resume_state.get("completed_steps", []))
        findings = list(resume_state.get("findings", []))
        output_ids = list(resume_state.get("output_ids", []))
        artifacts = list(resume_state.get("artifacts", []))
        errors = list(resume_state.get("errors", []))
        step_outputs = dict(resume_state.get("step_outputs", {})) if isinstance(resume_state.get("step_outputs"), dict) else {}

        async def execute_step(step, arguments):
            completed_arguments = self._complete_plan_arguments(step.tool_name or "", arguments)
            result = await self._tool(run, step.tool_name or "", completed_arguments)
            return await self._recover_plan_failure(run, step.tool_name or "", completed_arguments, result)

        async def verify_step(step, result):
            if result.status in {ToolStatus.SUCCESS, ToolStatus.PARTIAL_SUCCESS} and result.datasets and self._tool_produces_dataset(step.tool_name or ""):
                await self.trace.emit(run.id, EventType.VERIFICATION_STARTED, f"开始验证 {step.title} 的输出", payload={"tool": step.tool_name, "dataset_ids": result.datasets}, agent_id="main")
            problems = self._verification_problems(step.tool_name or "", result)
            if problems:
                await self.trace.emit(run.id, EventType.VERIFICATION_FAILED, "；".join(problems), payload={"tool": step.tool_name, "problems": problems}, agent_id="main")
            return problems

        async def checkpoint(completed_steps, state):
            await self._step_checkpoint(run.id, request, intent, plan, datasets, completed_steps, request_frame=request_frame, **state)

        outcome = await self.loop.execute_plan(
            plan,
            completed_steps=completed,
            findings=findings,
            output_ids=output_ids,
            artifacts=artifacts,
            errors=errors,
            step_outputs=step_outputs,
            execute_step=execute_step,
            verify_step=verify_step,
            checkpoint=checkpoint,
        )
        if outcome.failed_step is not None:
            result = outcome.failed_result
            message = outcome.errors[-1] if outcome.errors else f"{outcome.failed_step.title}未完成。"
            blocked = result is not None and result.error is not None and result.error.code in {"MISSING_DATASET", "MISSING_FIELD", "CRS_MISSING", "UNSUPPORTED_FORMAT", "NO_OVERLAP", "APPROVAL_REQUIRED"}
            summary = f"{outcome.failed_step.title}未完成：{message}" if not outcome.verification_problems else f"{outcome.failed_step.title}结果未通过验证。"
            return AgentResult(agent_id="main", task_id=task.id, status=AgentResultStatus.BLOCKED if blocked else AgentResultStatus.FAILED, summary=summary, findings=list(outcome.findings), datasets=list(outcome.output_ids), artifacts=list(outcome.artifacts), warnings=list(outcome.errors[:-1]), error=message, trace_id=run.id)

        operation = str(plan.metadata.get("operation") or intent.entities.get("operation") or "")
        status = AgentResultStatus.PARTIAL if outcome.errors else AgentResultStatus.SUCCESS
        return AgentResult(agent_id="main", task_id=task.id, status=status, summary=_plan_result_summary(operation, plan, list(outcome.findings), list(outcome.output_ids), list(outcome.artifacts)), findings=list(outcome.findings), datasets=list(outcome.output_ids), artifacts=list(outcome.artifacts), warnings=list(outcome.errors), error=outcome.errors[0] if outcome.errors and not outcome.findings else None, trace_id=run.id)

    def _verification_problems(self, tool_name: str, result: ToolResult) -> list[str]:
        if result.status not in {ToolStatus.SUCCESS, ToolStatus.PARTIAL_SUCCESS} or not result.datasets:
            return []
        if not self._tool_produces_dataset(tool_name):
            return []
        verified, problems = self.verifier.verify(result, {item.id: item for item in self.registry.list()})
        if verified:
            return []
        return problems

    def _tool_produces_dataset(self, tool_name: str) -> bool:
        try:
            return bool(self.executor.registry.get(tool_name).metadata.produces_dataset)
        except KeyError:
            return False

    def _complete_plan_arguments(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """补齐只有运行时才能确定的参数，例如自动选择投影 CRS。"""

        completed = dict(arguments)
        if tool_name in {"crs.reproject", "raster.reproject"} and completed.get("target_crs") == "auto":
            dataset = self.registry.resolve(str(completed.get("dataset_id", "")))
            if dataset is not None:
                completed["target_crs"] = CRSService(default_crs=self.settings.default_crs).choose_projected_crs(dataset)
        return completed

    async def _recover_plan_failure(self, run: Run, tool_name: str, arguments: dict[str, Any], result: ToolResult) -> ToolResult:
        """只对已知且可证明安全的 GIS 条件做一次修复后重试。"""

        if result.status is not ToolStatus.FAILED or result.error is None:
            return result
        action, rationale = self.failure_analyzer.analyze(result)
        event_type = {
            "REPAIR": EventType.REPAIR_SELECTED,
            "REPLAN": EventType.REPLAN_STARTED,
            "ASK_USER": EventType.DECISION_MADE,
            "ABORT": EventType.DECISION_MADE,
        }.get(action.value, EventType.DECISION_MADE)
        await self.trace.emit(
            run.id,
            event_type,
            rationale,
            payload={"action": action.value, "tool": tool_name, "error": result.error.model_dump(mode="json")},
            agent_id="main",
        )
        if action.value == "RETRY" and result.retryable:
            for _ in range(self.budget.max_retry_per_action):
                await self.trace.emit(run.id, EventType.RETRY_STARTED, f"重试 {tool_name}", payload={"tool": tool_name}, agent_id="main")
                retry = await self._tool(run, tool_name, arguments)
                if retry.status in {ToolStatus.SUCCESS, ToolStatus.PARTIAL_SUCCESS}:
                    return retry
                result = retry
            return result
        if action.value != "REPAIR":
            return result

        repaired_arguments = await self._repair_arguments(run, tool_name, arguments, result.error.code)
        if repaired_arguments is None:
            return result
        await self.trace.emit(run.id, EventType.RETRY_STARTED, f"修复输入后重试 {tool_name}", payload={"tool": tool_name, "arguments": repaired_arguments}, agent_id="main")
        return await self._tool(run, tool_name, repaired_arguments)

    async def _repair_arguments(self, run: Run, tool_name: str, arguments: dict[str, Any], error_code: str) -> dict[str, Any] | None:
        if error_code in {"CRS_UNIT_MISMATCH", "CRS_MISSING"} and "dataset_id" in arguments:
            dataset = self.registry.resolve(str(arguments["dataset_id"]))
            if dataset is None or dataset.crs is None:
                return None
            target_crs = CRSService(default_crs=self.settings.default_crs).choose_projected_crs(dataset)
            reprojection_tool = "raster.reproject" if dataset.kind.value == "RASTER" else "crs.reproject"
            repaired = await self._tool(run, reprojection_tool, {"dataset_id": dataset.id, "target_crs": target_crs})
            if repaired.status is not ToolStatus.SUCCESS or not repaired.datasets:
                return None
            updated = dict(arguments)
            updated["dataset_id"] = repaired.datasets[-1]
            return updated
        if error_code == "CRS_UNIT_MISMATCH" and "source_dataset_id" in arguments:
            source = self.registry.resolve(str(arguments["source_dataset_id"]))
            if source is None or source.crs is None:
                return None
            target_crs = CRSService(default_crs=self.settings.default_crs).choose_projected_crs(source)
            updated = dict(arguments)
            for key in ("source_dataset_id", "target_dataset_id"):
                identifier = updated.get(key)
                dataset = self.registry.resolve(str(identifier)) if identifier else None
                if dataset is None:
                    continue
                reprojection_tool = "raster.reproject" if dataset.kind.value == "RASTER" else "crs.reproject"
                repaired = await self._tool(run, reprojection_tool, {"dataset_id": dataset.id, "target_crs": target_crs})
                if repaired.status is not ToolStatus.SUCCESS or not repaired.datasets:
                    return None
                updated[key] = repaired.datasets[-1]
            return updated
        if error_code == "CRS_MISMATCH":
            left_id = arguments.get("left_dataset_id") or arguments.get("source_dataset_id")
            right_key = "right_dataset_id" if arguments.get("right_dataset_id") else "mask_dataset_id" if arguments.get("mask_dataset_id") else "target_dataset_id"
            right_id = arguments.get(right_key)
            left = self.registry.resolve(str(left_id)) if left_id else None
            right = self.registry.resolve(str(right_id)) if right_id else None
            if left is None or right is None or left.crs is None:
                return None
            target_crs = left.crs.authority
            if not target_crs:
                return None
            reprojection_tool = "raster.reproject" if right.kind.value == "RASTER" else "crs.reproject"
            repaired = await self._tool(run, reprojection_tool, {"dataset_id": right.id, "target_crs": target_crs})
            if repaired.status is not ToolStatus.SUCCESS or not repaired.datasets:
                return None
            updated = dict(arguments)
            updated[right_key] = repaired.datasets[-1]
            return updated
        if error_code == "INVALID_GEOMETRY":
            updated = dict(arguments)
            input_keys = [key for key in ("dataset_id", "left_dataset_id", "right_dataset_id", "mask_dataset_id") if key in updated]
            changed = False
            for key in input_keys:
                dataset = self.registry.resolve(str(updated[key]))
                if dataset is None or dataset.kind.value != "VECTOR":
                    continue
                repaired = await self._tool(run, "vector.repair", {"dataset_id": dataset.id})
                if repaired.status is ToolStatus.SUCCESS and repaired.datasets:
                    updated[key] = repaired.datasets[-1]
                    changed = True
            return updated if changed else None
        return None

    def _interpret_result(self, request: AgentRequest, run: Run) -> AgentResult:
        from app.entry.reference_resolver import ReferenceResolver

        identifiers = list(request.referenced_run_ids)
        if not identifiers and any(term in request.user_input.casefold() for term in ("刚才", "上一轮", "上一次", "结果", "产物")):
            identifiers = ["latest"]
        runs = ReferenceResolver(self.store).resolve_runs(identifiers, conversation_id=request.conversation_id, exclude_run_id=run.id)
        if not runs:
            return AgentResult(agent_id="main", task_id=run.task_id, status=AgentResultStatus.BLOCKED, summary="没有找到可以解读的历史结果。", error="RUN_NOT_FOUND", trace_id=run.id)
        latest = runs[0]
        saved = latest.metadata.get("result") if isinstance(latest.metadata, dict) else None
        if not isinstance(saved, dict):
            return AgentResult(agent_id="main", task_id=run.task_id, status=AgentResultStatus.BLOCKED, summary="找到历史运行，但其中没有保存可解读的结果。", error="RESULT_NOT_FOUND", trace_id=run.id)
        findings = saved.get("findings") if isinstance(saved.get("findings"), list) else []
        datasets = saved.get("datasets") if isinstance(saved.get("datasets"), list) else []
        artifacts = saved.get("artifacts") if isinstance(saved.get("artifacts"), list) else []
        status = str(saved.get("status", "UNKNOWN"))
        return AgentResult(agent_id="main", task_id=run.task_id, status=AgentResultStatus.SUCCESS, summary=f"上一轮运行状态为 {status}：{saved.get('summary', '未提供摘要')}", findings=findings, datasets=[str(item) for item in datasets], artifacts=[str(item) for item in artifacts], trace_id=run.id)

    def _resolve_datasets(self, request: AgentRequest):
        from app.entry.dataset_resolver import DatasetResolver

        return DatasetResolver().resolve(request, self.registry, store=self.store)

    def _resolve_request_resources(self, request: AgentRequest) -> RequestResources:
        from app.entry.dataset_resolver import DatasetResolver

        return DatasetResolver.request_resources(request, self.registry, self.store)

    def _diagnose_runs(self, request: AgentRequest, run: Run) -> AgentResult:
        from app.entry.reference_resolver import ReferenceResolver

        identifiers = list(request.referenced_run_ids)
        if not identifiers and any(word in request.user_input.casefold() for word in ("刚才", "上一轮", "上一次", "失败", "错误", "trace")):
            identifiers = ["latest"]
        runs = ReferenceResolver(self.store).resolve_runs(identifiers, conversation_id=request.conversation_id, exclude_run_id=run.id)
        if not runs:
            return AgentResult(
                agent_id="main",
                task_id=run.task_id,
                status=AgentResultStatus.BLOCKED,
                summary="没有找到可诊断的历史运行。",
                error="RUN_NOT_FOUND",
                trace_id=run.id,
            )
        findings = [
            {
                "run_id": item.id,
                "status": item.status.value,
                "error": item.error,
                "result": item.metadata.get("result"),
            }
            for item in runs
        ]
        return AgentResult(
            agent_id="main",
            task_id=run.task_id,
            status=AgentResultStatus.SUCCESS,
            summary=f"已整理 {len(findings)} 个历史运行的状态和错误信息。",
            findings=findings,
            trace_id=run.id,
        )

    async def _delegate(self, request: AgentRequest, run: Run, task: Task, datasets, tasks=None, *, intent: IntentResult | None = None, plan: Plan | None = None, request_frame: RequestFrame | None = None, working_memory: WorkingMemory | None = None) -> AgentResult:
        tasks = tasks or self.decomposer.decompose(request, datasets)
        self.guard.check_subagents(len(tasks))
        attached = self.task_service.attach_subtasks(task, tasks)
        # MainAgent 后面还会更新整体 Task；把拆解结果同步回当前对象，避免
        # 最终状态写回时把 subtasks 覆盖为空。
        task.subtasks = attached.subtasks
        task.updated_at = attached.updated_at
        for subtask in tasks:
            await self.trace.emit(run.id, EventType.SUBTASK_CREATED, subtask.goal, payload=subtask.model_dump(mode="json"), agent_id="main")
        if plan is not None and intent is not None:
            _set_plan_step_status(plan, "decompose", TaskStatus.SUCCEEDED)
            _set_plan_step_status(plan, "parallel", TaskStatus.RUNNING)
            await self._checkpoint(
                run.id,
                "delegation_started",
                {
                    **self._checkpoint_state(request, intent, plan, datasets, request_frame),
                    "completed_steps": ["decompose"],
                    "subtask_ids": [item.id for item in tasks],
                },
            )
        self.store.save_run(run.model_copy(update={"status": RunStatus.WAITING_SUBAGENT}))
        parent_memory = working_memory or self.store.get_working_memory(task.id)
        executions = await self.agent_manager.run(
            request,
            tasks,
            datasets,
            parent_task_id=task.id,
            parent_run_id=run.id,
            working_memory_snapshot=parent_memory,
        )
        results = [item.result for item in executions]
        deltas = [item.working_memory_delta for item in executions]
        if parent_memory is not None:
            parent_memory = self.working_memory_updater.merge_deltas(parent_memory, deltas)
            self.store.save_working_memory(parent_memory)
        findings = [{"agent_id": item.agent_id, "summary": item.summary, "status": item.status.value, "findings": item.findings} for item in results]
        output_ids = sorted({dataset_id for item in results for dataset_id in item.datasets})
        artifacts = sorted({artifact_id for item in results for artifact_id in item.artifacts})
        failed_required = [item for item, subtask in zip(results, tasks, strict=True) if subtask.required and item.status in {AgentResultStatus.FAILED, AgentResultStatus.BLOCKED}]
        failed_any = [item for item in results if item.status is not AgentResultStatus.SUCCESS]
        status = AgentResultStatus.FAILED if failed_required and len(failed_any) == len(tasks) else AgentResultStatus.PARTIAL if failed_any else AgentResultStatus.SUCCESS
        summary = f"已并行完成 {len(results) - len(failed_any)}/{len(results)} 个主题分析，并汇总结果。"
        result = AgentResult(agent_id="main", task_id=run.task_id, status=status, summary=summary, findings=findings, datasets=output_ids, artifacts=artifacts, warnings=[item.error for item in failed_any if item.error], trace_id=run.id)
        if plan is not None and intent is not None:
            _set_plan_step_status(plan, "parallel", TaskStatus.SUCCEEDED)
            _set_plan_step_status(plan, "synthesize", TaskStatus.SUCCEEDED)
            await self._checkpoint(
                run.id,
                "delegation_completed",
                {
                    **self._checkpoint_state(request, intent, plan, datasets, request_frame),
                    "completed_steps": [item.id for item in plan.steps],
                    "subtask_ids": [item.id for item in tasks],
                    "delegation_result": result.model_dump(mode="json"),
                    "working_memory": parent_memory.model_dump(mode="json") if parent_memory else None,
                    "working_memory_deltas": [item.model_dump(mode="json") for item in deltas],
                },
            )
        return result

    async def _tool(self, run: Run, name: str, arguments: dict[str, Any], *, call_id: str | None = None) -> ToolResult:
        current = self.store.get_run(run.id) or run
        self.guard.check_turn(current)
        self.guard.check_execution_time(current)
        self.guard.check_tool(current)
        next_run = current.model_copy(update={"tool_call_count": current.tool_call_count + 1, "status": RunStatus.WAITING_TOOL})
        self.store.save_run(next_run)
        call = ToolCall(id=call_id or new_id("call"), name=name, arguments=arguments, run_id=run.id, agent_id="main")
        result = await self.executor.execute(call, agent_id="main", services=self.executor.services)
        self.working_memory_updater.update_from_tool_result(current.task_id, result, run_id=current.id)
        return result


def _set_plan_step_status(plan: Plan, step_id: str, status: TaskStatus) -> None:
    step = next((item for item in plan.steps if item.id == step_id), None)
    if step is not None:
        step.status = status


def _plan_result_summary(operation: str, plan: Plan, findings: list[Any], datasets: list[str], artifacts: list[str]) -> str:
    if operation == "buffer":
        distance = plan.metadata.get("distance")
        distance_text = f"{float(distance):g} 米" if isinstance(distance, (int, float)) else "指定距离"
        return f"已生成 {distance_text} 缓冲区，并完成输出验证。"
    if operation == "distance":
        return "已完成矢量数据之间的距离分析。"
    if operation == "zonal_statistics":
        return "已完成栅格分区统计，并整理了各分区的统计结果。"
    if operation == "slope":
        return "已完成 DEM 坡度分析，并生成坡度栅格。"
    if operation == "reproject":
        return "已完成数据重投影，并登记了新的派生数据集。"
    if operation == "render":
        return "已生成可查看的地图结果。"
    if operation in {"clip", "intersection", "spatial_join", "dissolve", "repair"}:
        return f"已完成{ {'clip': '裁剪', 'intersection': '相交分析', 'spatial_join': '空间连接', 'dissolve': '要素融合', 'repair': '几何修复'}[operation] }，并完成输出验证。"
    if operation == "validate" or plan.intent is IntentType.DATA_INSPECTION:
        return f"已检查 {len(findings)} 个数据集，并整理了数据质量摘要。"
    return "已按计划完成 GIS 处理。"


def _parse_model_tool_call(raw_call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    function = raw_call.get("function") or raw_call
    name = str(function.get("name") or raw_call.get("name") or "")
    arguments = function.get("arguments", raw_call.get("arguments", {}))
    if isinstance(arguments, str):
        arguments = json.loads(arguments or "{}")
    if not isinstance(arguments, dict):
        raise TypeError("模型工具参数必须是 JSON 对象")
    return name, arguments


def _model_tool_name(raw_call: Any) -> str:
    """读取工具名只用于追踪；协议异常不应覆盖后续的错误处理。"""

    if not isinstance(raw_call, dict):
        return "未知工具"
    try:
        name, _ = _parse_model_tool_call(raw_call)
    except (TypeError, ValueError, json.JSONDecodeError):
        return "无效工具调用"
    return name or "未知工具"


def _run_status_for_result(status: AgentResultStatus, error: str | None = None) -> RunStatus:
    if status is AgentResultStatus.SUCCESS:
        return RunStatus.COMPLETED
    if status is AgentResultStatus.PARTIAL:
        return RunStatus.PARTIAL_COMPLETED
    if status is AgentResultStatus.CANCELLED:
        return RunStatus.CANCELLED
    if status is AgentResultStatus.BLOCKED:
        return RunStatus.WAITING_APPROVAL if error == "APPROVAL_REQUIRED" else RunStatus.WAITING_USER
    return RunStatus.FAILED


def _task_status_for_result(status: AgentResultStatus) -> TaskStatus:
    if status is AgentResultStatus.SUCCESS:
        return TaskStatus.SUCCEEDED
    if status is AgentResultStatus.PARTIAL:
        return TaskStatus.PARTIAL
    if status is AgentResultStatus.CANCELLED:
        return TaskStatus.CANCELLED
    if status is AgentResultStatus.BLOCKED:
        return TaskStatus.BLOCKED
    return TaskStatus.FAILED


def _conversation_reply(user_input: str) -> str:
    text = user_input.casefold()
    if any(word in text for word in ("你好", "您好", "嗨", "哈喽", "hello", "hi")):
        return "你好！我是 GeoAgent，可以帮你检查空间数据、处理坐标系、执行空间分析，也可以解释已有的分析结果。"
    return "我可以帮你检查数据集、处理坐标系、执行空间分析，或解释已有的运行结果。你可以直接描述目标，也可以先添加一个空间数据文件。"

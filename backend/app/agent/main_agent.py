"""GeoAgent 的 Main Agent Loop。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.agent.manager import AgentManager
from app.checkpoint.context import make_checkpoint
from app.checkpoint.store import CheckpointStore
from app.conversation_memory import ConversationMemoryService
from app.core.models import (
    AgentDecision,
    AgentRequest,
    AgentResult,
    AgentResultStatus,
    Checkpoint,
    DecisionType,
    ErrorCategory,
    FailureAction,
    IntentResult,
    IntentType,
    InteractionMode,
    LoopDirective,
    Plan,
    ReplanContext,
    RequestFrame,
    RequestResources,
    Run,
    RunBudget,
    RunStatus,
    SubAgentExecutionResult,
    SubTask,
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
    CONTROL_CAPABILITY_DEFINITIONS,
    AgentRouter,
    DecisionEngine,
    FailureAnalyzer,
    IntentResolver,
    Planner,
    Replanner,
    ReplanNotPossible,
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
from app.profile import ProfilePreferenceExtractor, UserProfileService
from app.run.lifecycle import PreparedRequest, RequestLifecycleBinder
from app.runtime.agent_loop import AgentLoop, PlanLoopOutcome, resolve_plan_arguments
from app.runtime.agent_runtime import AgentRuntime, RuntimeTransition
from app.runtime.agent_state import AgentStateBuilder
from app.runtime.budget import BudgetExceeded, BudgetGuard
from app.runtime.context_manager import ContextManager
from app.runtime.lifecycle import finish_run
from app.runtime.model_input_budget import ModelInputBudget
from app.runtime.protocol_history import (
    compact_protocol_messages,
    extract_protocol_messages,
    protocol_tool_message,
)
from app.runtime.tool_execution_cycle import ExecutionOutcome, ToolExecutionCycle
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
4. 输入不完整、数据角色不明确或关键参数缺失时，调用 agent.ask_user；不要猜测距离、字段、坐标系或数据集。
5. 简单且直接可执行时调用 GIS 工具；存在明确多步依赖时调用 agent.plan；存在相对独立的多个主题且确有并行价值时调用 agent.delegate。
6. 当前计划因失败或新信息不再适用时调用 agent.replan；已有清晰 current_plan 时优先让运行时继续执行，不要每轮重新规划。
7. 不要为了规划而规划，也不要为了委派而委派。每一批只能选择一个内部控制动作，不能把内部控制动作和 GIS 工具混在一起。
8. 每轮工具返回后重新判断下一步。只使用提供的工具和工具返回的事实，不能编造数据、文件、统计值或已完成的操作。最终回答简洁、具体、中文化。
""".strip()

_MODEL_CONTEXT_INSTRUCTION = "请优先依据当前请求、RequestFrame、WorkingMemory 和最新工具观察回答；只在确有必要时调用提供的空间工具。"


@dataclass(frozen=True, slots=True)
class DelegationExecutionResult:
    """一次委派动作的观察结果，不等同于 MainAgent 最终 AgentResult。"""

    tasks: tuple[SubTask, ...]
    executions: tuple[SubAgentExecutionResult, ...]
    directive: LoopDirective
    findings: tuple[Any, ...]
    dataset_ids: tuple[str, ...]
    artifact_ids: tuple[str, ...]
    observation: dict[str, Any]
    subagent_results: tuple[dict[str, Any], ...]
    latest_failure: dict[str, Any] | None
    fingerprint: str


class MainAgent:
    """负责用户目标、策略循环和结果汇总；确定性 GIS 计算委托给 Tools。"""

    def __init__(self, *, store: StateStore, trace: TraceRecorder, executor: ToolExecutor, registry, task_service: TaskService, agent_manager: AgentManager, settings, budget: RunBudget | None = None, checkpoint_store: CheckpointStore | None = None, memory: MemoryManager | None = None, knowledge: KnowledgeRetriever | None = None, model_adapter: ModelAdapter | None = None, model_adapters: dict[str, ModelAdapter] | None = None, default_model_profile: str | None = None, context_manager: ContextManager | None = None, services_factory=None, profile_service: UserProfileService | None = None, profile_extractor: ProfilePreferenceExtractor | None = None, conversation_memory: ConversationMemoryService | None = None) -> None:
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
        self.replanner = Replanner(self.planner)
        self.router = AgentRouter()
        self.decomposer = TaskDecomposer()
        self.failure_analyzer = FailureAnalyzer()
        self.verifier = ResultVerifier()
        self.checkpoint_store = checkpoint_store
        self.memory = memory
        self.memory_extractor = MemoryExtractor()
        self.profile_service = profile_service
        self.profile_extractor = profile_extractor or ProfilePreferenceExtractor()
        self.conversation_memory = conversation_memory
        self.working_memory_updater = WorkingMemoryUpdater(store)
        self.tool_execution_cycle = ToolExecutionCycle(
            raw_executor=self._raw_tool_for_cycle,
            tool_registry=self.executor.registry,
            registry=self.registry,
            trace=self.trace,
            failure_analyzer=self.failure_analyzer,
            verifier=self.verifier,
            budget=self.budget,
            default_crs=self.settings.default_crs,
        )
        self.knowledge = knowledge or KnowledgeRetriever()
        self.model_adapter = model_adapter
        self.model_adapters = model_adapters if model_adapters is not None else {}
        self.default_model_profile = default_model_profile
        self.context_manager = context_manager or ContextManager()
        self.services_factory = services_factory
        self.loop = AgentLoop()
        self.state_builder = AgentStateBuilder(store)
        self.decision_engine = DecisionEngine()
        self.agent_runtime = AgentRuntime(max_runtime_transitions=self.budget.max_runtime_transitions)
        self.lifecycle_binder = RequestLifecycleBinder(store, task_service)

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
            current_memory = self.store.get_working_memory(task.id)
            if current_memory is not None and not prepared_request.working_memory_created:
                working_memory = current_memory
            else:
                restored = WorkingMemory.model_validate(saved_working_memory)
                if restored.task_id == task.id:
                    self.store.save_working_memory(restored)
                    working_memory = restored
        if run is None:
            raise RuntimeError("请求没有可执行的 Run")
        saved_replan_count = resume_state.get("replan_count")
        if isinstance(saved_replan_count, int) and saved_replan_count > 0 and run.replan_count < saved_replan_count:
            run = run.model_copy(update={"replan_count": saved_replan_count})
            self.store.save_run(run)
        intent: IntentResult | None = None
        plan: Plan | None = None
        request_frame: RequestFrame | None = prepared_request.frame
        if self.conversation_memory is not None:
            self.conversation_memory.apply_request(request, request_frame, task, run)
        self._apply_profile_preference(request)
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
                if task is not None:
                    self.working_memory_updater.add_unresolved_questions(task.id, request_frame.blocking_issues, run_id=run.id)
                run = finish_run(run, RunStatus.WAITING_USER, error=result.error)
                self.store.save_run(run.model_copy(update={"metadata": {**run.metadata, "result": result.model_dump(mode="json")}}))
                if self.conversation_memory is not None:
                    self.conversation_memory.apply_result(request, run, result)
                await self._checkpoint(run.id, "run_completed", {"status": run.status.value, "result": result.model_dump(mode="json")})
                await self.trace.emit(run.id, EventType.RUN_FAILED, result.summary, payload={"status": run.status.value, "result": result.model_dump(mode="json")}, agent_id="main")
                return result
            run = run.model_copy(update={"status": RunStatus.PLANNING})
            self.store.save_run(run)
            self.guard.check_execution_time(run)
            # Model 与 Offline 只选择不同的 Decision Provider；执行控制统一进入 AgentRuntime。
            phase = "runtime_started"
            result = await self._model_loop(
                request,
                run,
                task,
                datasets,
                intent,
                plan,
                request_frame=request_frame,
                initial_messages=resume_state.get("messages") if resume_from and resume_state.get("messages") else None,
                initial_protocol_messages=resume_state.get("protocol_messages") if resume_from and isinstance(resume_state.get("protocol_messages"), list) else None,
                initial_latest_observation=resume_state.get("latest_observation") if resume_from else None,
                initial_latest_failure=resume_state.get("latest_failure") if resume_from and isinstance(resume_state.get("latest_failure"), dict) else None,
                initial_findings=resume_state.get("model_findings") or resume_state.get("findings") if resume_from else None,
                initial_dataset_ids=resume_state.get("model_dataset_ids") or resume_state.get("dataset_ids") if resume_from else None,
                initial_artifact_ids=resume_state.get("model_artifact_ids") or resume_state.get("artifact_ids") if resume_from else None,
                initial_subagent_results=resume_state.get("subagent_results") if resume_from and isinstance(resume_state.get("subagent_results"), list) else None,
                initial_delegation_fingerprints=resume_state.get("completed_delegation_fingerprints") if resume_from and isinstance(resume_state.get("completed_delegation_fingerprints"), list) else None,
                initial_legacy_delegation_result=resume_state.get("delegation_result") if resume_from and isinstance(resume_state.get("delegation_result"), dict) else None,
                initial_original_plan=resume_state.get("original_plan") if resume_from and isinstance(resume_state.get("original_plan"), dict) else None,
                working_memory=working_memory,
                initial_plan_completed_steps=resume_state.get("completed_steps") if resume_from and isinstance(resume_state.get("completed_steps"), list) else None,
                initial_plan_step_outputs=resume_state.get("step_outputs") if resume_from and isinstance(resume_state.get("step_outputs"), dict) else None,
                initial_previous_replan_reasons=resume_state.get("previous_replan_reasons") if resume_from and isinstance(resume_state.get("previous_replan_reasons"), list) else None,
                on_model_delta=on_model_delta,
            )
            final_status = _run_status_for_result(result.status, result.error)
            run = finish_run(self.store.get_run(run.id) or run, final_status, error=result.error)
            self.store.save_run(run.model_copy(update={"metadata": {**run.metadata, "result": result.model_dump(mode="json")}}))
            if task is not None and result.status is AgentResultStatus.BLOCKED and result.error in {"NEEDS_CLARIFICATION", "WAITING_USER"}:
                questions = request_frame.blocking_issues if request_frame and request_frame.blocking_issues else [result.summary]
                self.working_memory_updater.add_unresolved_questions(task.id, questions, run_id=run.id)
            if self.memory:
                candidates = self.memory_extractor.extract(request, request_frame, run, result, user_id=request.user_id)
                self.memory.write_candidates(candidates)
            if self.conversation_memory is not None:
                self.conversation_memory.apply_result(request, run, result)
            completion_state = {"status": final_status.value, "result": result.model_dump(mode="json")}
            previous_checkpoint = self.checkpoint_store.latest(run.id) if self.checkpoint_store else None
            if previous_checkpoint is not None:
                for key in (
                    "request",
                    "request_frame",
                    "intent",
                    "plan",
                    "current_plan",
                    "original_plan",
                    "completed_steps",
                    "step_outputs",
                    "protocol_messages",
                    "latest_observation",
                    "latest_failure",
                    "findings",
                    "dataset_ids",
                    "artifact_ids",
                    "subagent_results",
                    "completed_delegation_fingerprints",
                    "replan_count",
                    "previous_replan_reasons",
                    "runtime_mode",
                    "model_findings",
                    "model_dataset_ids",
                    "model_artifact_ids",
                ):
                    if key in previous_checkpoint.state:
                        completion_state[key] = previous_checkpoint.state[key]
            await self._checkpoint(run.id, "run_completed", completion_state)
            if task is not None and prepared_request.action in {"create_task", "bind_task", "retry_run"}:
                self.task_service.update(task, status=_task_status_for_result(result.status), result=result.summary)
            await self.trace.emit(run.id, EventType.RUN_COMPLETED if result.status in {AgentResultStatus.SUCCESS, AgentResultStatus.PARTIAL} else EventType.RUN_FAILED, result.summary, payload={"status": result.status.value, "result": result.model_dump(mode="json")}, agent_id="main")
            return result.model_copy(update={"trace_id": run.id})
        except asyncio.CancelledError:
            run = finish_run(self.store.get_run(run.id) or run, RunStatus.CANCELLED, error="CANCELLED")
            self.store.save_run(run)
            previous_checkpoint = self.checkpoint_store.latest(run.id) if self.checkpoint_store else None
            state = dict(previous_checkpoint.state) if previous_checkpoint else {}
            state.update(
                {
                    "request": request.model_dump(mode="json"),
                    "request_frame": request_frame.model_dump(mode="json") if request_frame else None,
                    "intent": intent.model_dump(mode="json") if intent else None,
                }
            )
            if plan is not None or "plan" not in state:
                state["plan"] = plan.model_dump(mode="json") if plan is not None else None
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
        initial_protocol_messages: list[dict[str, Any]] | None = None,
        initial_latest_observation: dict[str, Any] | None = None,
        initial_latest_failure: dict[str, Any] | None = None,
        initial_findings: list[Any] | None = None,
        initial_dataset_ids: list[str] | None = None,
        initial_artifact_ids: list[str] | None = None,
        initial_subagent_results: list[Any] | None = None,
        initial_delegation_fingerprints: list[str] | None = None,
        initial_legacy_delegation_result: dict[str, Any] | None = None,
        initial_original_plan: dict[str, Any] | None = None,
        working_memory: WorkingMemory | None = None,
        initial_plan_completed_steps: list[str] | None = None,
        initial_plan_step_outputs: dict[str, Any] | None = None,
        initial_previous_replan_reasons: list[str] | None = None,
        on_model_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> AgentResult:
        """兼容入口；模型和无模型请求都交给同一个 AgentRuntime。"""

        return await self._run_agent_runtime(
            request,
            run,
            task,
            datasets,
            intent,
            plan,
            request_frame=request_frame,
            initial_messages=initial_messages,
            initial_protocol_messages=initial_protocol_messages,
            initial_latest_observation=initial_latest_observation,
            initial_latest_failure=initial_latest_failure,
            initial_findings=initial_findings,
            initial_dataset_ids=initial_dataset_ids,
            initial_artifact_ids=initial_artifact_ids,
            initial_subagent_results=initial_subagent_results,
            initial_delegation_fingerprints=initial_delegation_fingerprints,
            initial_legacy_delegation_result=initial_legacy_delegation_result,
            initial_original_plan=initial_original_plan,
            working_memory=working_memory,
            initial_plan_completed_steps=initial_plan_completed_steps,
            initial_plan_step_outputs=initial_plan_step_outputs,
            initial_previous_replan_reasons=initial_previous_replan_reasons,
            on_model_delta=on_model_delta,
        )

    async def _run_agent_runtime(
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
        initial_protocol_messages: list[dict[str, Any]] | None = None,
        initial_latest_observation: dict[str, Any] | None = None,
        initial_latest_failure: dict[str, Any] | None = None,
        initial_findings: list[Any] | None = None,
        initial_dataset_ids: list[str] | None = None,
        initial_artifact_ids: list[str] | None = None,
        initial_subagent_results: list[Any] | None = None,
        initial_delegation_fingerprints: list[str] | None = None,
        initial_legacy_delegation_result: dict[str, Any] | None = None,
        initial_original_plan: dict[str, Any] | None = None,
        working_memory: WorkingMemory | None = None,
        initial_plan_completed_steps: list[str] | None = None,
        initial_plan_step_outputs: dict[str, Any] | None = None,
        initial_previous_replan_reasons: list[str] | None = None,
        on_model_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> AgentResult:
        model_adapter = self._model_adapter_for(request)
        session: dict[str, Any] = {
            "protocol_messages": extract_protocol_messages(protocol_messages=initial_protocol_messages, legacy_messages=initial_messages),
            "findings": list(initial_findings or []),
            "dataset_ids": set(initial_dataset_ids or []),
            "artifact_ids": set(initial_artifact_ids or []),
            "latest_observation": _observation_from_checkpoint(initial_latest_observation),
            "latest_failure": dict(initial_latest_failure or {}) or None,
            "datasets": list(datasets),
            "plan": plan,
            "original_plan": Plan.model_validate(initial_original_plan) if initial_original_plan else plan.model_copy(deep=True) if plan is not None else None,
            "completed_steps": set(initial_plan_completed_steps or []),
            "step_outputs": dict(initial_plan_step_outputs or {}),
            "previous_replan_reasons": list(initial_previous_replan_reasons or []),
            "subagent_results": list(initial_subagent_results or []),
            "completed_delegation_fingerprints": set(initial_delegation_fingerprints or []),
            "legacy_delegation_result": dict(initial_legacy_delegation_result) if initial_legacy_delegation_result else None,
            "run": run,
            "working_memory": working_memory,
            "fast_path_enabled": plan is not None,
            "decision_provider": "model" if model_adapter is not None else "offline",
        }

        await self._checkpoint(
            run.id,
            "runtime_started",
            self._runtime_checkpoint_state(request, intent, request_frame, session),
        )

        initial_state = self.state_builder.build(
            request,
            request_frame,
            run,
            task=task,
            current_plan=plan,
            plan_completed_steps=session["completed_steps"],
            plan_step_outputs=session["step_outputs"],
            latest_observation=session["latest_observation"],
            active_dataset_ids=sorted(session["dataset_ids"]),
            active_artifact_ids=sorted(session["artifact_ids"]),
            subagent_results=session["subagent_results"],
            working_memory=working_memory,
        )

        async def refresh_state(_state):
            current = self.store.get_run(run.id) or session["run"]
            return self.state_builder.build(
                request,
                request_frame,
                current,
                task=task,
                current_plan=session["plan"],
                plan_completed_steps=session["completed_steps"],
                plan_step_outputs=session["step_outputs"],
                latest_observation=session["latest_observation"],
                latest_failure=session["latest_failure"],
                active_dataset_ids=sorted(session["dataset_ids"]),
                active_artifact_ids=sorted(session["artifact_ids"]),
                subagent_results=session["subagent_results"],
                working_memory=session["working_memory"],
            )

        async def decide(state):
            state_decision = self.decision_engine.decide_from_state(state)
            if state_decision is not None:
                await self._trace_runtime_decision(state_decision, state)
                return state_decision
            current = self.store.get_run(run.id) or session["run"]
            self.guard.check_turn(current)
            self.guard.check_execution_time(current)
            current = current.model_copy(update={"turn_count": current.turn_count + 1, "status": RunStatus.RUNNING})
            self.store.save_run(current)
            session["run"] = current
            current_memory = self.store.get_working_memory(current.task_id) if current.task_id else None
            session["working_memory"] = current_memory or session["working_memory"]
            if model_adapter is not None:
                decision = await self._model_decision(
                    request,
                    current,
                    task,
                    datasets,
                    intent,
                    request_frame,
                    session,
                    model_adapter=model_adapter,
                    on_model_delta=on_model_delta,
                )
            else:
                decision = self._offline_decision(
                    state,
                    request=request,
                    task=task,
                    intent=intent,
                    request_frame=request_frame,
                    session=session,
                )
            await self._trace_runtime_decision(decision, state)
            return decision

        async def dispatch(decision, state):
            return await self._dispatch_runtime_decision(
                decision,
                state,
                request=request,
                run=session["run"],
                task=task,
                intent=intent,
                request_frame=request_frame,
                session=session,
            )

        async def fast_path(state):
            if not session["fast_path_enabled"] or session["plan"] is None:
                return None
            if session["decision_provider"] == "offline" and session["plan"].metadata.get("delegated_roles") and not session["subagent_results"]:
                # 委派计划的控制动作由 Offline Decision Provider 产生，不能被
                # 旧的无工具 PlanStep 直接消费。
                session["fast_path_enabled"] = False
                return None
            step = self.loop.next_executable_step(session["plan"], session["completed_steps"])
            if step is None:
                session["fast_path_enabled"] = False
                return None
            await self._trace_runtime_decision(
                AgentDecision(
                    type=DecisionType.TOOL,
                    reasoning_summary=f"按当前计划直接执行下一步：{step.title}。",
                    source="plan_fast_path",
                ),
                state,
            )
            return await self._execute_runtime_plan_step(
                step,
                request=request,
                run=session["run"],
                task=task,
                intent=intent,
                request_frame=request_frame,
                session=session,
            )

        outcome = await self.agent_runtime.run(
            initial_state,
            decide=decide,
            dispatch=dispatch,
            refresh_state=refresh_state,
            fast_path=fast_path,
            max_runtime_transitions=self.budget.max_runtime_transitions,
        )
        return self._finalize_runtime_outcome(outcome, task=task, run=run)

    async def _model_decision(
        self,
        request: AgentRequest,
        run: Run,
        task: Task | None,
        datasets,
        intent: IntentResult | None,
        request_frame: RequestFrame | None,
        session: dict[str, Any],
        *,
        model_adapter: ModelAdapter,
        on_model_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> AgentDecision:
        """Model Decision Provider：只构造模型输入并返回 AgentDecision。"""

        refreshed_datasets = self._refresh_model_datasets(
            request,
            datasets,
            session["dataset_ids"],
            session["working_memory"],
            session["latest_observation"],
        )
        session["datasets"] = refreshed_datasets
        request_resources = self._resolve_request_resources(request)
        messages, bounded_protocol, tools = self._build_model_messages(
            request,
            run,
            task,
            refreshed_datasets,
            intent,
            session["plan"],
            request_frame,
            session["protocol_messages"],
            working_memory=session["working_memory"],
            request_resources=request_resources,
            current_observation=session["latest_observation"],
            plan_progress={"completed_steps": sorted(session["completed_steps"]), "step_outputs": session["step_outputs"]},
            latest_failure=session["latest_failure"],
        )
        session["protocol_messages"] = bounded_protocol
        content_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        input_tokens = output_tokens = 0
        model_name: str | None = None
        async for chunk in model_adapter.stream(ModelRequest(messages=messages, tools=tools, max_tokens=self.budget.max_tokens)):
            if chunk.content:
                content_parts.append(chunk.content)
                if on_model_delta is not None:
                    await on_model_delta(chunk.content)
            if chunk.tool_calls:
                tool_calls = chunk.tool_calls
            input_tokens = chunk.input_tokens or input_tokens
            output_tokens = chunk.output_tokens or output_tokens
            model_name = chunk.model or model_name
        response = ModelResponse(
            content="".join(content_parts),
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model_name,
        )
        session["last_response"] = response
        return self.decision_engine.from_model_response(response, source="model")

    def _offline_decision(
        self,
        state,
        *,
        request: AgentRequest,
        task: Task | None,
        intent: IntentResult | None,
        request_frame: RequestFrame | None,
        session: dict[str, Any],
    ) -> AgentDecision:
        """Offline Decision Provider：只复用旧 Planner/Router 生成 AgentDecision。"""

        if session.get("legacy_delegation_result"):
            return _decision_from_agent_result(AgentResult.model_validate(session["legacy_delegation_result"]), source="offline_compat")
        if intent is None:
            intent = self.legacy_intent_adapter.to_intent(request_frame, request, session["datasets"])

        current_plan = session.get("plan")
        if current_plan is not None:
            if current_plan.clarification:
                return AgentDecision(
                    type=DecisionType.ASK_USER,
                    reasoning_summary=current_plan.clarification,
                    final_response=current_plan.clarification,
                    source="offline",
                )
            if current_plan.metadata.get("delegated_roles") and not session["subagent_results"]:
                return self.router.route(
                    intent,
                    current_plan,
                    session["datasets"],
                    subtasks=self.decomposer.decompose(request, session["datasets"]),
                ).model_copy(update={"source": "offline"})
            if current_plan.metadata.get("delegated_roles") and session["subagent_results"]:
                completed = sum(1 for item in session["subagent_results"] if item.get("status") == AgentResultStatus.SUCCESS.value)
                total = len(session["subagent_results"])
                result = AgentResult(
                    agent_id="main",
                    task_id=task.id if task else state.task_id,
                    status=AgentResultStatus.SUCCESS if completed == total else AgentResultStatus.PARTIAL,
                    summary=f"已并行完成 {completed}/{total} 个主题分析，并汇总结果。",
                    findings=list(session["findings"]),
                    datasets=sorted(session["dataset_ids"]),
                    artifacts=sorted(session["artifact_ids"]),
                    trace_id=state.run_id,
                )
                return _decision_from_agent_result(result, source="offline")
            if self.loop.next_executable_step(current_plan, session["completed_steps"]) is None:
                operation = str(current_plan.metadata.get("operation") or intent.entities.get("operation") or "")
                result = AgentResult(
                    agent_id="main",
                    task_id=task.id if task else state.task_id,
                    status=AgentResultStatus.SUCCESS,
                    summary=_plan_result_summary(operation, current_plan, list(session["findings"]), sorted(session["dataset_ids"]), sorted(session["artifact_ids"])),
                    findings=list(session["findings"]),
                    datasets=sorted(session["dataset_ids"]),
                    artifacts=sorted(session["artifact_ids"]),
                    trace_id=state.run_id,
                )
                return _decision_from_agent_result(result, source="offline")

        if intent.intent.value == "RUN_DIAGNOSIS":
            return _decision_from_agent_result(self._diagnose_runs(request, session["run"]), source="offline")
        if intent.intent.value == "RESULT_INTERPRETATION":
            return _decision_from_agent_result(self._interpret_result(request, session["run"]), source="offline")
        if intent.intent.value == "UNKNOWN":
            return AgentDecision(type=DecisionType.FINAL, reasoning_summary="当前回合不需要 GIS 执行。", final_response=_conversation_reply(request.user_input), source="offline")
        findings = self.knowledge.retrieve(request.user_input) if intent.intent.value == "KNOWLEDGE_QUERY" else []
        result = AgentResult(
            agent_id="main",
            task_id=task.id if task else state.task_id,
            status=AgentResultStatus.SUCCESS,
            summary="已整理 GIS 知识上下文。" if findings else "已理解请求，但当前计划没有需要执行的 GIS 操作。",
            findings=findings,
            trace_id=state.run_id,
            warnings=[] if findings else ["当前未配置大模型，离线模式只支持有限的 GIS 操作。"],
        )
        return _decision_from_agent_result(result, source="offline")

    @staticmethod
    def _finalize_runtime_outcome(outcome, *, task: Task | None, run: Run) -> AgentResult:
        """RuntimeOutcome 到 AgentResult 的唯一转换出口。"""

        if outcome.error == "AGENT_RUNTIME_BUDGET_EXCEEDED":
            raise BudgetExceeded("Agent turn budget exceeded")
        status = outcome.status or (AgentResultStatus.FAILED if outcome.error else AgentResultStatus.SUCCESS)
        summary = outcome.final_response or outcome.error or "当前运行已结束。"
        if outcome.error == "EMPTY_MODEL_RESPONSE":
            summary = "模型未返回可执行的工具调用或文本回答。"
        return AgentResult(
            agent_id="main",
            task_id=task.id if task else run.task_id,
            status=status,
            summary=summary,
            findings=list(outcome.findings),
            datasets=sorted(set(outcome.dataset_ids)),
            artifacts=sorted(set(outcome.artifact_ids)),
            error=outcome.error,
            trace_id=run.id,
        )

    async def _trace_runtime_decision(self, decision: AgentDecision, state) -> None:
        """统一记录运行时决策摘要，不记录模型私有推理过程。"""

        current = self.store.get_run(state.run_id)
        plan = state.current_plan
        failure = state.latest_failure or {}
        await self.trace.emit(
            state.run_id,
            EventType.DECISION_MADE,
            decision.reasoning_summary,
            payload={
                "decision_type": decision.type.value,
                "source": decision.source,
                "turn_count": current.turn_count if current else state.turn_count,
                "has_plan": plan is not None,
                "plan_revision": plan.revision if plan else None,
                "latest_failure_action": failure.get("directive") or failure.get("action"),
                "tool_call_count": current.tool_call_count if current else state.tool_call_count,
                "replan_count": current.replan_count if current else state.replan_count,
            },
            agent_id="main",
        )

    async def _dispatch_runtime_decision(
        self,
        decision,
        state,
        *,
        request: AgentRequest,
        run: Run,
        task: Task | None,
        intent: IntentResult | None,
        request_frame: RequestFrame | None,
        session: dict[str, Any],
    ) -> RuntimeTransition:
        """把 AgentDecision 分派到已有执行能力，不在此处重复实现工具语义。"""

        if decision.type.value == "FINAL":
            response = (decision.final_response or "").strip()
            if not response:
                return RuntimeTransition(terminal=True, error="EMPTY_MODEL_RESPONSE")
            metadata = decision.metadata
            for finding in metadata.get("findings") or []:
                if finding not in session["findings"]:
                    session["findings"].append(finding)
            session["dataset_ids"].update(str(item) for item in metadata.get("datasets") or [])
            session["artifact_ids"].update(str(item) for item in metadata.get("artifacts") or [])
            if decision.source == "model" or metadata.get("model"):
                session["findings"].append({"model": metadata.get("model"), "content": response})
            return RuntimeTransition(
                terminal=True,
                status=_agent_result_status(metadata.get("status"), default=AgentResultStatus.SUCCESS),
                final_response=response,
                error=metadata.get("error"),
                findings=tuple(session["findings"]),
                dataset_ids=tuple(sorted(session["dataset_ids"])),
                artifact_ids=tuple(sorted(session["artifact_ids"])),
            )
        if decision.type.value == "ASK_USER":
            return RuntimeTransition(
                terminal=True,
                status=AgentResultStatus.BLOCKED,
                final_response=decision.final_response or decision.reasoning_summary,
                error="WAITING_USER",
            )
        if decision.type.value == "ABORT":
            if decision.metadata.get("empty_response"):
                return RuntimeTransition(terminal=True, error="EMPTY_MODEL_RESPONSE")
            return RuntimeTransition(
                terminal=True,
                status=AgentResultStatus.FAILED,
                error=decision.metadata.get("error_code") or decision.reasoning_summary,
            )
        if decision.type.value == "PLAN":
            if intent is None:
                intent = self.legacy_intent_adapter.to_intent(request_frame, request, session["datasets"])
            new_plan = self.planner.build(decision.plan_goal or state.goal, intent, session["datasets"])
            session["plan"] = new_plan
            session["original_plan"] = new_plan.model_copy(deep=True)
            session["fast_path_enabled"] = True
            session["completed_steps"] = set()
            session["step_outputs"] = {}
            session["latest_failure"] = None
            await self.trace.emit(run.id, EventType.PLAN_CREATED, f"生成运行时计划：{len(new_plan.steps)} 步", payload={**new_plan.model_dump(mode="json"), "source": "agent_runtime"}, agent_id="main")
            await self._checkpoint(run.id, "plan_created", self._runtime_checkpoint_state(request, intent, request_frame, session))
            return RuntimeTransition(current_plan=new_plan)
        if decision.type.value == "DELEGATE":
            if task is None:
                return RuntimeTransition(terminal=True, status=AgentResultStatus.BLOCKED, error="WAITING_USER", final_response="当前请求没有可委派的业务任务。")
            delegation_tasks = decision.subtasks or self.decomposer.decompose(request, session["datasets"])
            fingerprint = _delegation_fingerprint(delegation_tasks)
            if fingerprint in session["completed_delegation_fingerprints"]:
                observation = {
                    "type": "delegation",
                    "code": "DELEGATION_NO_PROGRESS",
                    "completed": 0,
                    "total": len(delegation_tasks),
                    "directive": LoopDirective.CONTINUE.value,
                    "fingerprint": fingerprint,
                    "message": "相同委派已经执行过，本轮没有新的资源或失败条件变化。",
                }
                session["latest_observation"] = observation
                return RuntimeTransition(
                    terminal=False,
                    observation=observation,
                    directive=LoopDirective.CONTINUE,
                    subagent_results=tuple(session["subagent_results"]),
                )
            delegation = await self._execute_delegation(
                request,
                run,
                task,
                session["datasets"],
                delegation_tasks,
                intent=intent,
                plan=session["plan"],
                request_frame=request_frame,
                working_memory=session["working_memory"],
                fingerprint=fingerprint,
            )
            session["fast_path_enabled"] = False
            session["completed_delegation_fingerprints"].add(delegation.fingerprint)
            session["subagent_results"] = _merge_subagent_views(session["subagent_results"], delegation.subagent_results)
            session["findings"].extend(delegation.findings)
            session["dataset_ids"].update(delegation.dataset_ids)
            session["artifact_ids"].update(delegation.artifact_ids)
            session["latest_observation"] = delegation.observation
            session["latest_failure"] = delegation.latest_failure
            return RuntimeTransition(
                terminal=False,
                observation=delegation.observation,
                directive=delegation.directive,
                latest_failure=delegation.latest_failure,
                clear_failure=delegation.latest_failure is None,
                findings=tuple(session["findings"]),
                dataset_ids=tuple(sorted(session["dataset_ids"])),
                artifact_ids=tuple(sorted(session["artifact_ids"])),
                subagent_results=tuple(session["subagent_results"]),
            )
        if decision.type.value == "REPLAN":
            return await self._dispatch_runtime_replan(
                decision,
                state,
                request=request,
                run=run,
                task=task,
                intent=intent,
                request_frame=request_frame,
                session=session,
            )

        if decision.type.value != "TOOL":
            return RuntimeTransition(terminal=True, status=AgentResultStatus.FAILED, error=f"不支持的运行时动作：{decision.type.value}")

        raw_calls = decision.metadata.get("raw_tool_calls") or []
        session["protocol_messages"].append({"role": "assistant", "content": decision.metadata.get("content", ""), "tool_calls": raw_calls})
        invalid_calls = {item.get("id"): item for item in decision.metadata.get("invalid_tool_calls", []) if isinstance(item, dict)}
        batch_observations: list[dict[str, Any]] = []
        batch_failure: dict[str, Any] | None = None
        for call in decision.normalized_tool_calls():
            invalid = invalid_calls.get(call.id)
            if invalid is not None:
                tool_result = ToolResult(
                    call_id=call.id,
                    status=ToolStatus.FAILED,
                    error=ToolError(code="INVALID_TOOL_ARGUMENTS", category=ErrorCategory.INPUT, message=f"模型工具参数不是有效 JSON：{invalid.get('error', '未知错误')}"),
                )
                execution_outcome = None
                observation = _invalid_tool_call_observation(call.id, tool_result, str(invalid.get("error", "未知错误")))
                name = call.name
            else:
                execution_outcome = await self.tool_execution_cycle.execute(
                    session["run"],
                    call.name,
                    call.arguments,
                    user_id=self.store.user_id_for_run(session["run"].id),
                    call_id=call.id,
                )
                tool_result = execution_outcome.result
                observation = _execution_observation(execution_outcome)
                name = call.name
            batch_observations.append(observation)
            session["findings"].append(
                {
                    "tool": name,
                    "status": tool_result.status.value,
                    "accepted": execution_outcome.accepted if execution_outcome else False,
                    "verification_problems": execution_outcome.verification_problems if execution_outcome else [],
                    "recovery_action": execution_outcome.recovery_action.value if execution_outcome and execution_outcome.recovery_action else None,
                    "directive": execution_outcome.directive.value if execution_outcome else LoopDirective.ABORT.value,
                    "verified": execution_outcome.verified if execution_outcome else False,
                    "attempts": execution_outcome.attempts if execution_outcome else 1,
                    "output": tool_result.output,
                    "error": tool_result.error.model_dump(mode="json") if tool_result.error else None,
                }
            )
            if execution_outcome is not None and execution_outcome.accepted:
                self._accept_main_tool_result(session["run"], tool_result)
                session["dataset_ids"].update(tool_result.datasets)
                session["artifact_ids"].update(tool_result.artifacts)
            elif execution_outcome is not None:
                batch_failure = {
                    "action": execution_outcome.directive.value,
                    "directive": execution_outcome.directive.value,
                    "deterministic": False,
                    "tool_name": name,
                    "error_code": tool_result.error.code if tool_result.error else None,
                    "error": tool_result.error.message if tool_result.error else execution_outcome.rationale,
                    "verification_problems": list(execution_outcome.verification_problems),
                    "recovery_action": execution_outcome.recovery_action.value if execution_outcome.recovery_action else None,
                    "attempts": execution_outcome.attempts,
                }
            session["protocol_messages"].append(protocol_tool_message(execution_outcome or observation))
        session["latest_failure"] = batch_failure
        latest_observation = _batch_observation(batch_observations)
        session["latest_observation"] = latest_observation
        bounded_protocol = compact_protocol_messages(session["protocol_messages"], max_tokens=self.budget.protocol_history_tokens)
        session["protocol_messages"] = bounded_protocol
        await self._checkpoint(
            session["run"].id,
            "model_tool_completed",
            self._runtime_checkpoint_state(request, intent, request_frame, session),
        )
        return RuntimeTransition(
            observation=latest_observation,
            latest_failure=session["latest_failure"],
            clear_failure=session["latest_failure"] is None,
            findings=tuple(session["findings"]),
            dataset_ids=tuple(sorted(session["dataset_ids"])),
            artifact_ids=tuple(sorted(session["artifact_ids"])),
        )

    async def _execute_runtime_plan_step(
        self,
        step,
        *,
        request: AgentRequest,
        run: Run,
        task: Task | None,
        intent: IntentResult | None,
        request_frame: RequestFrame | None,
        session: dict[str, Any],
    ) -> RuntimeTransition:
        if step.tool_name is None:
            session["completed_steps"].add(step.id)
            step.status = TaskStatus.SUCCEEDED
            return RuntimeTransition(completed_steps=(step.id,), current_plan=session["plan"])
        arguments = self._complete_plan_arguments(step.tool_name, resolve_plan_arguments(step.arguments, session["step_outputs"]), user_id=request.user_id)
        outcome = await self.tool_execution_cycle.execute(session["run"], step.tool_name, arguments, user_id=request.user_id)
        session["findings"].append(
            {
                "step_id": step.id,
                "tool": step.tool_name,
                "status": outcome.result.status.value,
                "accepted": outcome.accepted,
                "verified": outcome.verified,
                "verification_problems": list(outcome.verification_problems),
                "directive": outcome.directive.value,
                "attempts": outcome.attempts,
                "datasets": list(outcome.result.datasets),
                "artifacts": list(outcome.result.artifacts),
            }
        )
        if outcome.accepted:
            self._accept_main_tool_result(session["run"], outcome.result)
            session["dataset_ids"].update(outcome.result.datasets)
            session["artifact_ids"].update(outcome.result.artifacts)
            session["latest_failure"] = None
            session["completed_steps"].add(step.id)
            step.status = TaskStatus.SUCCEEDED
            session["step_outputs"][step.id] = {
                "dataset_id": outcome.result.datasets[-1] if outcome.result.datasets else None,
                "dataset_ids": list(outcome.result.datasets),
                "artifact_ids": list(outcome.result.artifacts),
                "output": outcome.result.output,
            }
        else:
            session["fast_path_enabled"] = False
            session["latest_failure"] = {
                "action": outcome.directive.value,
                "directive": outcome.directive.value,
                "deterministic": True,
                "step_id": step.id,
                "tool_name": step.tool_name,
                "error_code": outcome.result.error.code if outcome.result.error else None,
                "error": outcome.result.error.message if outcome.result.error else outcome.rationale,
                "verification_problems": list(outcome.verification_problems),
                "recovery_action": outcome.recovery_action.value if outcome.recovery_action else None,
                "attempts": outcome.attempts,
            }
        observation = _execution_observation(outcome)
        session["latest_observation"] = observation
        await self._step_checkpoint(
            session["run"].id,
            request,
            intent,
            session["plan"],
            session["datasets"],
            set(session["completed_steps"]),
            request_frame=request_frame,
            phase="plan_step_completed" if outcome.accepted else "plan_step_failed",
            runtime_session=session,
        )
        return RuntimeTransition(
            observation=observation,
            latest_failure=session["latest_failure"],
            clear_failure=session["latest_failure"] is None,
            directive=outcome.directive,
            findings=tuple(session["findings"]),
            dataset_ids=tuple(sorted(session["dataset_ids"])),
            artifact_ids=tuple(sorted(session["artifact_ids"])),
            current_plan=session["plan"],
            completed_steps=(step.id,) if outcome.accepted else (),
            step_outputs={step.id: session["step_outputs"][step.id]} if outcome.accepted else {},
        )

    async def _dispatch_runtime_replan(
        self,
        decision,
        state,
        *,
        request: AgentRequest,
        run: Run,
        task: Task | None,
        intent: IntentResult | None,
        request_frame: RequestFrame | None,
        session: dict[str, Any],
    ) -> RuntimeTransition:
        """把运行时 REPLAN 请求接到现有 Replanner，不在 Decision 层重建计划。"""

        current_plan = session.get("plan")
        if current_plan is None:
            return RuntimeTransition(
                terminal=True,
                status=AgentResultStatus.BLOCKED,
                error="REPLAN_REQUIRED",
                final_response="当前没有可供重新规划的执行计划。",
                directive=LoopDirective.REPLAN,
            )
        failure = session.get("latest_failure") or {}
        failed_step = next(
            (item for item in current_plan.steps if item.id == failure.get("step_id")),
            None,
        )
        error_message = str(failure.get("error") or decision.reasoning_summary or "当前执行失败，需要重新规划。")
        error_code = str(failure.get("error_code") or "REPLAN_REQUIRED")
        verification_problems = [str(item) for item in failure.get("verification_problems", [])]
        failed_result = ToolResult(
            call_id=f"replan_{failed_step.id if failed_step else 'runtime'}",
            status=ToolStatus.SUCCESS if verification_problems and not failure.get("error_code") else ToolStatus.FAILED,
            error=None if verification_problems and not failure.get("error_code") else ToolError(
                code=error_code,
                category=ErrorCategory.EXECUTION,
                message=error_message,
            ),
        )
        failed_outcome = ExecutionOutcome(
            result=failed_result,
            verified=False,
            verification_problems=verification_problems,
            recovery_action=FailureAction.REPLAN,
            attempts=int(failure.get("attempts") or 1),
            accepted=False,
            directive=LoopDirective.REPLAN,
            rationale=error_message,
        )
        outcome = PlanLoopOutcome(
            completed_steps=frozenset(session["completed_steps"]),
            findings=tuple(session["findings"]),
            output_ids=tuple(sorted(session["dataset_ids"])),
            artifacts=tuple(sorted(session["artifact_ids"])),
            errors=(error_message,),
            step_outputs=dict(session["step_outputs"]),
            failed_step=failed_step,
            failed_outcome=failed_outcome,
            directive=LoopDirective.REPLAN,
        )
        if intent is None:
            intent = self.legacy_intent_adapter.to_intent(request_frame, request, session["datasets"])
        state_payload = {
            "completed_steps": sorted(session["completed_steps"]),
            "findings": list(session["findings"]),
            "output_ids": sorted(session["dataset_ids"]),
            "artifacts": sorted(session["artifact_ids"]),
            "errors": [error_message],
            "step_outputs": dict(session["step_outputs"]),
            "previous_replan_reasons": list(session.get("previous_replan_reasons", [])),
        }
        original_plan = session.get("original_plan") or current_plan.model_copy(deep=True)
        try:
            revised, next_state = await self._replan_plan(
                request,
                run,
                session["datasets"],
                intent,
                original_plan,
                current_plan,
                outcome,
                state_payload,
                request_frame=request_frame,
            )
        except ReplanNotPossible as exc:
            running = self.store.get_run(run.id) or run
            if running.status is RunStatus.REPLANNING:
                self.store.save_run(running.model_copy(update={"status": RunStatus.RUNNING}))
            return RuntimeTransition(
                terminal=True,
                status=AgentResultStatus.BLOCKED,
                error="REPLAN_REQUIRED",
                final_response=str(exc),
                directive=LoopDirective.REPLAN,
                latest_failure=failure,
            )
        except BudgetExceeded as exc:
            return RuntimeTransition(
                terminal=True,
                status=AgentResultStatus.BLOCKED,
                error=str(exc),
                directive=LoopDirective.ABORT,
                latest_failure=failure,
            )
        session["plan"] = revised
        session["original_plan"] = original_plan
        session["completed_steps"] = set(next_state["completed_steps"])
        session["step_outputs"] = dict(next_state["step_outputs"])
        session["findings"] = list(next_state["findings"])
        session["latest_failure"] = None
        session["fast_path_enabled"] = True
        session["run"] = self.store.get_run(run.id) or run
        return RuntimeTransition(
            current_plan=revised,
            clear_failure=True,
            directive=LoopDirective.CONTINUE,
            observation={"replanned": True, "revision": revised.revision},
            findings=tuple(session["findings"]),
            dataset_ids=tuple(sorted(session["dataset_ids"])),
            artifact_ids=tuple(sorted(session["artifact_ids"])),
        )

    def _build_model_messages(
        self,
        request: AgentRequest,
        run: Run,
        task: Task | None,
        datasets,
        intent: IntentResult | None,
        plan: Plan | None,
        request_frame: RequestFrame | None,
        protocol_messages: list[dict[str, Any]],
        *,
        working_memory: WorkingMemory | None,
        request_resources: RequestResources,
        current_observation: ToolResult | dict[str, Any] | None,
        plan_progress: dict[str, Any] | None = None,
        latest_failure: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """为当前 turn 计算完整输入预算，并生成动态 Context + 协议历史。"""

        tools = self._model_tools()
        bounded_protocol = compact_protocol_messages(protocol_messages, max_tokens=self.budget.protocol_history_tokens)
        budget = ModelInputBudget(
            input_tokens=self.budget.model_input_tokens,
            context_tokens=self.budget.model_context_tokens,
            protocol_tokens=self.budget.protocol_history_tokens,
        )
        allocation = budget.allocate(
            _MODEL_SYSTEM_PROMPT,
            tools,
            bounded_protocol,
            overhead=_MODEL_CONTEXT_INSTRUCTION,
        )
        if allocation.over_budget:
            raise BudgetExceeded(
                "MODEL_INPUT_BUDGET_EXCEEDED: "
                f"固定输入成本 {allocation.fixed_tokens} 超过模型输入预算 {allocation.input_tokens}。"
            )
        dynamic_tokens = allocation.available_context_tokens
        content = self._model_user_message(
            request,
            run,
            datasets,
            intent,
            plan,
            request_frame,
            working_memory=working_memory,
            request_resources=request_resources,
            current_observation=current_observation,
            plan_progress=plan_progress,
            latest_failure=latest_failure,
            task_goal=task.goal if task else None,
            context_tokens=dynamic_tokens,
        )
        messages = [
            {"role": "system", "content": _MODEL_SYSTEM_PROMPT},
            {"role": "user", "content": content},
            *bounded_protocol,
        ]
        total = budget.estimate_request(_MODEL_SYSTEM_PROMPT, content, bounded_protocol, tools)
        if total > self.budget.model_input_tokens:
            dynamic_tokens = max(0, dynamic_tokens - (total - self.budget.model_input_tokens))
            content = self._model_user_message(
                request,
                run,
                datasets,
                intent,
                plan,
                request_frame,
                working_memory=working_memory,
                request_resources=request_resources,
                current_observation=current_observation,
                plan_progress=plan_progress,
                latest_failure=latest_failure,
                task_goal=task.goal if task else None,
                context_tokens=dynamic_tokens,
            )
            messages[1] = {"role": "user", "content": content}
            total = budget.estimate_request(_MODEL_SYSTEM_PROMPT, content, bounded_protocol, tools)
        if total > self.budget.model_input_tokens:
            raise BudgetExceeded(
                "MODEL_INPUT_BUDGET_EXCEEDED: "
                f"模型请求估算 {total} token，超过输入预算 {self.budget.model_input_tokens}。"
            )
        return messages, bounded_protocol, tools

    def _model_adapter_for(self, request: AgentRequest) -> ModelAdapter | None:
        profile_id = request.model_profile or self.default_model_profile
        if profile_id:
            adapter = self.model_adapters.get(profile_id)
            if adapter is not None:
                return adapter
        return self.model_adapter

    def _apply_profile_preference(self, request: AgentRequest) -> None:
        if self.profile_service is None or not request.user_id:
            return
        changes = self.profile_extractor.extract(request.user_input)
        if changes:
            self.profile_service.update(request.user_id, changes)

    def _model_user_message(
        self,
        request: AgentRequest,
        run: Run,
        datasets,
        intent: IntentResult | None,
        plan: Plan | None,
        request_frame: RequestFrame | None = None,
        *,
        working_memory: WorkingMemory | None = None,
        request_resources: RequestResources | None = None,
        current_observation: ToolResult | dict[str, Any] | None = None,
        task_goal: str | None = None,
        plan_progress: dict[str, Any] | None = None,
        latest_failure: dict[str, Any] | None = None,
        context_tokens: int | None = None,
    ) -> str:
        history = self.store.list_messages(request.conversation_id, limit=8)
        if history and history[-1].role == "user" and history[-1].content == request.user_input:
            history = history[:-1]
        conversation = [
            {"role": message.role, "content": message.content}
            for message in history
            if message.role in {"user", "assistant", "system"}
        ]
        memories = self.memory.recall(request.user_input, scope="project", user_id=request.user_id, limit=5) if self.memory else []
        user_profile = self.profile_service.get_or_create(request.user_id) if self.profile_service and request.user_id else None
        conversation_memory = self.conversation_memory.get(request.conversation_id, request.user_id) if self.conversation_memory and request.user_id else None
        referenced_runs = [
            item.model_dump(mode="json")
            for item in self.request_understanding.reference_resolver.resolve_runs(request.referenced_run_ids, conversation_id=request.conversation_id, exclude_run_id=run.id)
        ]
        working_memory = working_memory or (self.store.get_working_memory(run.task_id) if run.task_id else None)
        context = self.context_manager.main_context(
            request,
            datasets,
            plan,
            memories,
            conversation=conversation,
            working_memory=working_memory,
            budget=self.budget.model_dump(mode="json"),
            referenced_runs=referenced_runs,
            intent_hint=intent,
            request_frame=request_frame,
            user_profile=user_profile,
            conversation_memory=conversation_memory,
            request_resources=request_resources or self._resolve_request_resources(request),
            task_goal=task_goal,
            run_state=run,
            current_observation=current_observation,
            plan_progress=plan_progress,
            latest_failure=latest_failure,
            max_tokens=context_tokens,
        )
        return _MODEL_CONTEXT_INSTRUCTION + "\n" + json.dumps(context, ensure_ascii=False, default=str)

    def _refresh_model_datasets(
        self,
        request: AgentRequest,
        datasets,
        dataset_ids: set[str],
        working_memory: WorkingMemory | None,
        observation: ToolResult | dict[str, Any] | None,
    ) -> list[Any]:
        """每个模型 turn 重新按用户作用域读取当前可见数据集。"""

        identifiers: list[str] = [*request.dataset_ids, *request.attachment_ids, *[item.id for item in datasets]]
        if working_memory is not None:
            identifiers.extend(working_memory.active_dataset_ids)
        identifiers.extend(dataset_ids)
        if observation is not None:
            if isinstance(observation, ToolResult):
                identifiers.extend(observation.datasets)
            elif isinstance(observation, dict):
                for item in _observation_items(observation):
                    if item.get("accepted") is True:
                        identifiers.extend(item.get("datasets") or [])
        registry = self.registry.for_user(request.user_id)
        refreshed: list[Any] = []
        seen: set[str] = set()
        for identifier in identifiers:
            item = registry.resolve(identifier)
            if item is not None and item.id not in seen:
                refreshed.append(item)
                seen.add(item.id)
        return refreshed

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
        ] + CONTROL_CAPABILITY_DEFINITIONS

    @staticmethod
    def _checkpoint_state(request: AgentRequest, intent: IntentResult | None, plan: Plan | None, datasets, request_frame: RequestFrame | None = None) -> dict[str, Any]:
        return {
            "request": request.model_dump(mode="json"),
            "request_frame": request_frame.model_dump(mode="json") if request_frame else None,
            "intent": intent.model_dump(mode="json") if intent else None,
            "plan": plan.model_dump(mode="json") if plan else None,
            "dataset_ids": [item.id for item in datasets],
        }

    def _runtime_checkpoint_state(
        self,
        request: AgentRequest,
        intent: IntentResult | None,
        request_frame: RequestFrame | None,
        session: dict[str, Any],
    ) -> dict[str, Any]:
        """Runtime 的统一恢复视图；Context 和完整 Tool 输出不写入其中。"""

        current_run = self.store.get_run(session["run"].id) or session["run"]
        plan = session.get("plan")
        payload = self._checkpoint_state(request, intent, plan, session.get("datasets", []), request_frame)
        payload.update(
            {
                "current_plan": plan.model_dump(mode="json") if plan is not None else None,
                "original_plan": session["original_plan"].model_dump(mode="json") if session.get("original_plan") is not None else None,
                "completed_steps": sorted(session.get("completed_steps", set())),
                "step_outputs": dict(session.get("step_outputs", {})),
                "protocol_messages": list(session.get("protocol_messages", [])),
                "latest_observation": session.get("latest_observation"),
                "latest_failure": session.get("latest_failure"),
                "findings": list(session.get("findings", [])),
                "dataset_ids": sorted(session.get("dataset_ids", set())),
                "artifact_ids": sorted(session.get("artifact_ids", set())),
                "subagent_results": list(session.get("subagent_results", [])),
                "completed_delegation_fingerprints": sorted(session.get("completed_delegation_fingerprints", set())),
                "replan_count": current_run.replan_count,
                "previous_replan_reasons": list(session.get("previous_replan_reasons", [])),
                "runtime_mode": session.get("decision_provider"),
            }
        )
        # 旧模型恢复仍读取这些别名；它们与 canonical runtime state 同源。
        payload.update(
            {
                "model_findings": list(session.get("findings", [])),
                "model_dataset_ids": sorted(session.get("dataset_ids", set())),
                "model_artifact_ids": sorted(session.get("artifact_ids", set())),
            }
        )
        return payload

    async def _checkpoint(self, run_id: str, phase: str, state: dict[str, Any]) -> None:
        if not self.checkpoint_store:
            return
        checkpoint = make_checkpoint(run_id, phase, state)
        self.checkpoint_store.save(checkpoint)
        await self.trace.emit(run_id, EventType.CHECKPOINT_SAVED, f"保存 Checkpoint：{phase}", payload={"checkpoint_id": checkpoint.id, "phase": phase}, agent_id="main")

    async def _step_checkpoint(
        self,
        run_id: str,
        request: AgentRequest,
        intent: IntentResult,
        plan: Plan,
        datasets,
        completed_steps: set[str],
        *,
        request_frame: RequestFrame | None = None,
        phase: str = "step_completed",
        runtime_session: dict[str, Any] | None = None,
        **state: Any,
    ) -> None:
        if runtime_session is not None:
            await self._checkpoint(run_id, phase, self._runtime_checkpoint_state(request, intent, request_frame, runtime_session))
            return
        await self._checkpoint(
            run_id,
            phase,
            {
                **self._checkpoint_state(request, intent, plan, datasets, request_frame),
                "completed_steps": sorted(completed_steps),
                **state,
            },
        )

    async def _execute_plan(self, request: AgentRequest, run: Run, task: Task, datasets, intent: IntentResult, plan: Plan, resume_state: dict[str, Any], *, request_frame: RequestFrame | None = None) -> AgentResult:
        state = _plan_state_from_resume(resume_state)
        original_plan = Plan.model_validate(resume_state["original_plan"]) if isinstance(resume_state.get("original_plan"), dict) else plan.model_copy(deep=True)
        current_plan = plan

        while True:
            outcome = await self._execute_plan_once(request, run, task, datasets, intent, current_plan, state, original_plan=original_plan, request_frame=request_frame)
            if outcome.failed_step is not None and outcome.directive is LoopDirective.REPLAN and _replan_candidate(outcome):
                try:
                    current_plan, state = await self._replan_plan(
                        request,
                        run,
                        datasets,
                        intent,
                        original_plan,
                        current_plan,
                        outcome,
                        state,
                        request_frame=request_frame,
                    )
                except ReplanNotPossible as exc:
                    return _plan_failure_result(task, run, outcome, str(exc))
                continue
            return self._plan_result_from_outcome(task, run, intent, current_plan, outcome)

    async def _execute_plan_once(self, request: AgentRequest, run: Run, task: Task, datasets, intent: IntentResult, plan: Plan, state: dict[str, Any], *, original_plan: Plan, request_frame: RequestFrame | None = None):
        async def execute_step(step, arguments):
            completed_arguments = self._complete_plan_arguments(step.tool_name or "", arguments, user_id=request.user_id)
            outcome = await self.tool_execution_cycle.execute(run, step.tool_name or "", completed_arguments, user_id=request.user_id)
            if outcome.accepted:
                self._accept_main_tool_result(run, outcome.result)
            return outcome

        async def checkpoint(completed_steps, checkpoint_state):
            current_run = self.store.get_run(run.id) or run
            await self._step_checkpoint(
                run.id,
                request,
                intent,
                plan,
                datasets,
                completed_steps,
                request_frame=request_frame,
                original_plan=original_plan.model_dump(mode="json"),
                replan_count=current_run.replan_count,
                previous_replan_reasons=list(state.get("previous_replan_reasons", [])),
                **checkpoint_state,
            )

        return await self.loop.execute_plan(
            plan,
            completed_steps=set(state["completed_steps"]),
            findings=list(state["findings"]),
            output_ids=list(state["output_ids"]),
            artifacts=list(state["artifacts"]),
            errors=list(state["errors"]),
            step_outputs=dict(state["step_outputs"]),
            execute_step=execute_step,
            checkpoint=checkpoint,
        )

    async def _replan_plan(self, request, run, datasets, intent, original_plan, current_plan, outcome, state, *, request_frame):
        current_run = self.store.get_run(run.id) or run
        self.guard.check_replan(current_run)
        next_count = current_run.replan_count + 1
        failed = outcome.failed_outcome
        failed_step = outcome.failed_step
        reason = _replan_reason(failed_step, failed)
        reasons = list(state.get("previous_replan_reasons", []))
        reasons.append(reason)
        current_memory = self.store.get_working_memory(run.task_id) if run.task_id else None
        current_dataset_ids = list(dict.fromkeys([*(current_memory.active_dataset_ids if current_memory else []), *outcome.output_ids]))
        current_artifact_ids = list(dict.fromkeys([*(current_memory.active_artifact_ids if current_memory else []), *outcome.artifacts]))
        context = ReplanContext(
            goal=current_plan.goal,
            original_plan=original_plan,
            current_plan=current_plan,
            current_revision=current_plan.revision,
            completed_steps=sorted(outcome.completed_steps),
            step_outputs=_compact_replan_step_outputs(outcome.step_outputs),
            failed_step=failed_step,
            failed_tool_name=failed_step.tool_name if failed_step else None,
            failed_arguments=dict(failed_step.arguments) if failed_step else {},
            error_code=failed.result.error.code if failed and failed.result.error else None,
            error_message=failed.result.error.message if failed and failed.result.error else None,
            verification_problems=list(failed.verification_problems) if failed else [],
            recovery_action=failed.recovery_action if failed else None,
            directive=outcome.directive,
            attempts=failed.attempts if failed else 1,
            current_dataset_ids=current_dataset_ids,
            current_artifact_ids=current_artifact_ids,
            replan_count=next_count,
            previous_replan_reasons=reasons,
        )
        replanning_run = current_run.model_copy(update={"status": RunStatus.REPLANNING, "replan_count": next_count})
        self.store.save_run(replanning_run)
        await self._checkpoint(
            run.id,
            "replan_started",
            {
                "request": request.model_dump(mode="json"),
                "request_frame": request_frame.model_dump(mode="json") if request_frame else None,
                "intent": intent.model_dump(mode="json"),
                "plan": current_plan.model_dump(mode="json"),
                "original_plan": original_plan.model_dump(mode="json"),
                "replan_context": context.model_dump(mode="json"),
                "replan_count": next_count,
                **_plan_state_from_outcome(outcome, reasons),
            },
        )
        revised = self.replanner.replan(context, intent, datasets)
        running = replanning_run.model_copy(update={"status": RunStatus.RUNNING})
        self.store.save_run(running)
        next_state = _plan_state_from_outcome(outcome, reasons)
        # 失败尝试保留在 findings/replan reason 中，但不能让已成功修复的
        # 旧错误把新 revision 错判为 PARTIAL。
        next_state["errors"] = []
        await self._checkpoint(
            run.id,
            "replan_completed",
            {
                "request": request.model_dump(mode="json"),
                "request_frame": request_frame.model_dump(mode="json") if request_frame else None,
                "intent": intent.model_dump(mode="json"),
                "plan": revised.model_dump(mode="json"),
                "original_plan": original_plan.model_dump(mode="json"),
                "replan_count": next_count,
                **next_state,
            },
        )
        await self.trace.emit(run.id, EventType.PLAN_CREATED, f"生成 Replan revision {revised.revision}", payload={"revision": revised.revision, "source": "replan", "previous_revision": current_plan.revision}, agent_id="main")
        return revised, next_state

    def _plan_result_from_outcome(self, task, run, intent, plan, outcome):
        if outcome.failed_step is not None:
            failed = outcome.failed_outcome
            message = outcome.errors[-1] if outcome.errors else f"{outcome.failed_step.title}未完成。"
            if failed is not None and failed.rationale and not failed.verification_problems and failed.result.error is None:
                message = failed.rationale
            if outcome.directive is LoopDirective.ASK_USER:
                return _plan_failure_result(task, run, outcome, "WAITING_USER", f"{outcome.failed_step.title}需要补充信息：{message}")
            if outcome.directive is LoopDirective.REPLAN:
                return _plan_failure_result(task, run, outcome, "REPLAN_REQUIRED", f"{outcome.failed_step.title}当前执行策略不适用，需要重新规划。")
            return _plan_failure_result(task, run, outcome, message, f"{outcome.failed_step.title}未完成：{message}")
        operation = str(plan.metadata.get("operation") or intent.entities.get("operation") or "")
        status = AgentResultStatus.PARTIAL if outcome.errors else AgentResultStatus.SUCCESS
        return AgentResult(agent_id="main", task_id=task.id, status=status, summary=_plan_result_summary(operation, plan, list(outcome.findings), list(outcome.output_ids), list(outcome.artifacts)), findings=list(outcome.findings), datasets=list(outcome.output_ids), artifacts=list(outcome.artifacts), warnings=list(outcome.errors), error=outcome.errors[0] if outcome.errors and not outcome.findings else None, trace_id=run.id)

    def _accept_main_tool_result(self, run: Run, result: ToolResult) -> None:
        """MainAgent 作为 Task 状态所有者，显式接收已验证的 Tool 结果。"""

        self.working_memory_updater.update_from_tool_result(run.task_id, result, run_id=run.id)

    def _complete_plan_arguments(self, tool_name: str, arguments: dict[str, Any], *, user_id: str | None = None) -> dict[str, Any]:
        """补齐只有运行时才能确定的参数，例如自动选择投影 CRS。"""

        completed = dict(arguments)
        if tool_name in {"crs.reproject", "raster.reproject"} and completed.get("target_crs") == "auto":
            dataset = self.registry.for_user(user_id).resolve(str(completed.get("dataset_id", "")))
            if dataset is not None:
                completed["target_crs"] = CRSService(default_crs=self.settings.default_crs).choose_projected_crs(dataset)
        return completed

    def _interpret_result(self, request: AgentRequest, run: Run) -> AgentResult:
        identifiers = list(request.referenced_run_ids)
        if not identifiers and any(term in request.user_input.casefold() for term in ("刚才", "上一轮", "上一次", "结果", "产物")):
            identifiers = ["latest"]
        runs = self.request_understanding.reference_resolver.resolve_runs(identifiers, conversation_id=request.conversation_id, exclude_run_id=run.id)
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

        return DatasetResolver().resolve(request, self.registry.for_user(request.user_id), store=self.store)

    def _resolve_request_resources(self, request: AgentRequest) -> RequestResources:
        from app.entry.dataset_resolver import DatasetResolver

        return DatasetResolver.request_resources(request, self.registry.for_user(request.user_id), self.store)

    def _diagnose_runs(self, request: AgentRequest, run: Run) -> AgentResult:
        identifiers = list(request.referenced_run_ids)
        if not identifiers and any(word in request.user_input.casefold() for word in ("刚才", "上一轮", "上一次", "失败", "错误", "trace")):
            identifiers = ["latest"]
        runs = self.request_understanding.reference_resolver.resolve_runs(identifiers, conversation_id=request.conversation_id, exclude_run_id=run.id)
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

    async def _execute_delegation(
        self,
        request: AgentRequest,
        run: Run,
        task: Task,
        datasets,
        tasks=None,
        *,
        intent: IntentResult | None = None,
        plan: Plan | None = None,
        request_frame: RequestFrame | None = None,
        working_memory: WorkingMemory | None = None,
        fingerprint: str | None = None,
        completed_fingerprints: set[str] | None = None,
        legacy_plan_progress: bool = False,
    ) -> DelegationExecutionResult:
        """只执行一次委派并返回 Observation；不构造 MainAgent 最终结果。"""

        tasks = list(tasks or self.decomposer.decompose(request, datasets))
        self.guard.check_subagents(len(tasks))
        fingerprint = fingerprint or _delegation_fingerprint(tasks)
        attached = self.task_service.attach_subtasks(task, tasks)
        task.subtasks = attached.subtasks
        task.updated_at = attached.updated_at
        for subtask in tasks:
            await self.trace.emit(run.id, EventType.SUBTASK_CREATED, subtask.goal, payload=subtask.model_dump(mode="json"), agent_id="main")
        if legacy_plan_progress and plan is not None and intent is not None:
            _set_plan_step_status(plan, "decompose", TaskStatus.SUCCEEDED)
            _set_plan_step_status(plan, "parallel", TaskStatus.RUNNING)
        await self._checkpoint(
            run.id,
            "delegation_started",
            {
                **self._checkpoint_state(request, intent, plan, datasets, request_frame),
                "subtask_ids": [item.id for item in tasks],
                "delegation_fingerprint": fingerprint,
                "completed_delegation_fingerprints": sorted(completed_fingerprints or set()),
                "runtime_delegation": not legacy_plan_progress,
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
        deltas = [item.working_memory_delta for item in executions]
        if parent_memory is not None:
            parent_memory = self.working_memory_updater.merge_deltas(parent_memory, deltas)
            self.store.save_working_memory(parent_memory)
        current_run = self.store.get_run(run.id) or run
        self.store.save_run(current_run.model_copy(update={"status": RunStatus.RUNNING}))

        views = tuple(_subagent_result_view(subtask, execution) for subtask, execution in zip(tasks, executions, strict=True))
        findings = tuple(dict(item) for item in views)
        output_ids = tuple(sorted({dataset_id for delta in deltas for dataset_id in delta.added_dataset_ids}))
        artifact_ids = tuple(sorted({artifact_id for delta in deltas for artifact_id in delta.added_artifact_ids}))
        required_executions = [execution for execution, subtask in zip(executions, tasks, strict=True) if subtask.required]
        directive = _aggregate_subagent_directive(required_executions)
        latest_failure = _delegation_failure(executions, tasks)
        observation = {
            "type": "delegation",
            "completed": sum(1 for item in executions if item.result.status is AgentResultStatus.SUCCESS),
            "total": len(executions),
            "directive": directive.value,
            "fingerprint": fingerprint,
            "results": [dict(item) for item in views],
        }
        await self.trace.emit(
            run.id,
            EventType.DELEGATION_COMPLETED,
            f"委派完成：{observation['completed']}/{observation['total']}",
            payload={
                "subtask_count": len(tasks),
                "success_count": observation["completed"],
                "blocked_count": sum(1 for item in executions if item.result.status is AgentResultStatus.BLOCKED),
                "failed_count": sum(1 for item in executions if item.result.status is AgentResultStatus.FAILED),
                "directive": directive.value,
                "dataset_count": len(output_ids),
                "artifact_count": len(artifact_ids),
                "fingerprint": fingerprint,
            },
            agent_id="main",
        )
        await self._checkpoint(
            run.id,
            "delegation_completed",
            {
                **self._checkpoint_state(request, intent, plan, datasets, request_frame),
                "subtask_ids": [item.id for item in tasks],
                "delegation_fingerprint": fingerprint,
                "completed_delegation_fingerprints": sorted(completed_fingerprints or {fingerprint}),
                "subagent_results": [dict(item) for item in views],
                "dataset_ids": list(output_ids),
                "artifact_ids": list(artifact_ids),
                "directive": directive.value,
                "working_memory_refs": _working_memory_refs(parent_memory),
                "runtime_delegation": not legacy_plan_progress,
            },
        )
        return DelegationExecutionResult(
            tasks=tuple(tasks),
            executions=tuple(executions),
            directive=directive,
            findings=findings,
            dataset_ids=output_ids,
            artifact_ids=artifact_ids,
            observation=observation,
            subagent_results=views,
            latest_failure=latest_failure,
            fingerprint=fingerprint,
        )

    async def _delegate(self, request: AgentRequest, run: Run, task: Task, datasets, tasks=None, *, intent: IntentResult | None = None, plan: Plan | None = None, request_frame: RequestFrame | None = None, working_memory: WorkingMemory | None = None) -> AgentResult:
        """旧离线路径兼容包装器；Runtime 路径使用 _execute_delegation。"""

        delegation = await self._execute_delegation(
            request,
            run,
            task,
            datasets,
            tasks,
            intent=intent,
            plan=plan,
            request_frame=request_frame,
            working_memory=working_memory,
            legacy_plan_progress=True,
        )
        results = [item.result for item in delegation.executions]
        failed_any = [item for item in results if item.status is not AgentResultStatus.SUCCESS]
        required_executions = [execution for execution, subtask in zip(delegation.executions, delegation.tasks, strict=True) if subtask.required]
        status = AgentResultStatus.FAILED if delegation.directive is LoopDirective.ABORT and required_executions and all(item.directive is LoopDirective.ABORT for item in required_executions) else AgentResultStatus.PARTIAL if failed_any else AgentResultStatus.SUCCESS
        summary = f"已并行完成 {len(results) - len(failed_any)}/{len(results)} 个主题分析，并汇总结果。"
        if delegation.directive is LoopDirective.ASK_USER:
            status, summary, error = AgentResultStatus.BLOCKED, "部分子任务需要补充用户信息后才能继续。", "WAITING_USER"
        elif delegation.directive is LoopDirective.REPLAN:
            status, summary, error = AgentResultStatus.BLOCKED, "部分子任务当前执行策略不适用，需要重新规划。", "REPLAN_REQUIRED"
        elif status is AgentResultStatus.FAILED:
            error = next((item.result.error for item in required_executions if item.result.error), "SubAgent 执行失败")
            summary = "所有必需子任务均执行失败。"
        else:
            error = next((item.error for item in failed_any if item.error), None)
        result = AgentResult(
            agent_id="main",
            task_id=run.task_id,
            status=status,
            summary=summary,
            findings=list(delegation.findings),
            datasets=list(delegation.dataset_ids),
            artifacts=list(delegation.artifact_ids),
            warnings=[item.error for item in failed_any if item.error],
            error=error,
            trace_id=run.id,
        )
        if plan is not None and intent is not None:
            _set_plan_step_status(plan, "parallel", TaskStatus.SUCCEEDED)
            _set_plan_step_status(plan, "synthesize", TaskStatus.SUCCEEDED)
            await self._checkpoint(
                run.id,
                "delegation_completed",
                {
                    **self._checkpoint_state(request, intent, plan, datasets, request_frame),
                    "completed_steps": [item.id for item in plan.steps],
                    "subtask_ids": [item.id for item in delegation.tasks],
                    "legacy_result": result.model_dump(mode="json"),
                    "working_memory_refs": _working_memory_refs(self.store.get_working_memory(task.id)),
                    "runtime_delegation": False,
                },
            )
        return result

    async def _execute_tool_raw(self, run: Run, name: str, arguments: dict[str, Any], *, call_id: str | None = None, attempt: int = 1) -> ToolResult:
        current = self.store.get_run(run.id) or run
        self.guard.check_turn(current)
        self.guard.check_execution_time(current)
        self.guard.check_tool(current)
        next_run = current.model_copy(update={"tool_call_count": current.tool_call_count + 1, "status": RunStatus.WAITING_TOOL})
        self.store.save_run(next_run)
        call = ToolCall(id=call_id or new_id("call"), name=name, arguments=arguments, run_id=run.id, agent_id="main", attempt=attempt)
        user_id = self.store.user_id_for_run(current.id)
        services = self.services_factory(user_id) if self.services_factory else self.executor.services
        return await self.executor.execute(call, agent_id="main", services=services)

    async def _tool(self, run: Run, name: str, arguments: dict[str, Any], *, call_id: str | None = None, attempt: int = 1) -> ToolResult:
        """保留给旧测试/调用方的 Raw Tool hook；新路径由 Cycle 负责包裹。"""

        return await self._execute_tool_raw(run, name, arguments, call_id=call_id, attempt=attempt)

    async def _raw_tool_for_cycle(self, run: Run, name: str, arguments: dict[str, Any], *, call_id: str | None = None, attempt: int = 1) -> ToolResult:
        """通过兼容 hook 执行 Raw Tool，便于测试替换而不绕过 Cycle。"""
        try:
            return await self._tool(run, name, arguments, call_id=call_id, attempt=attempt)
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            return await self._tool(run, name, arguments, call_id=call_id)


def _set_plan_step_status(plan: Plan, step_id: str, status: TaskStatus) -> None:
    step = next((item for item in plan.steps if item.id == step_id), None)
    if step is not None:
        step.status = status


def _plan_state_from_resume(resume_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "completed_steps": list(resume_state.get("completed_steps", [])),
        "findings": list(resume_state.get("findings", [])),
        "output_ids": list(resume_state.get("output_ids", [])),
        "artifacts": list(resume_state.get("artifacts", [])),
        "errors": list(resume_state.get("errors", [])),
        "step_outputs": dict(resume_state.get("step_outputs", {})) if isinstance(resume_state.get("step_outputs"), dict) else {},
        "previous_replan_reasons": list(resume_state.get("previous_replan_reasons", [])),
    }


def _plan_state_from_outcome(outcome, reasons: list[str]) -> dict[str, Any]:
    return {
        "completed_steps": sorted(outcome.completed_steps),
        "findings": list(outcome.findings),
        "output_ids": list(outcome.output_ids),
        "artifacts": list(outcome.artifacts),
        "errors": list(outcome.errors),
        "step_outputs": dict(outcome.step_outputs),
        "previous_replan_reasons": reasons,
    }


def _replan_candidate(outcome) -> bool:
    failed = outcome.failed_outcome
    if failed is None:
        return False
    if failed.verification_problems:
        return True
    return bool(failed.result.error and failed.result.error.code == "ALGORITHM_NOT_APPLICABLE")


def _replan_reason(step, outcome: ExecutionOutcome | None) -> str:
    code = outcome.result.error.code if outcome and outcome.result.error else "VERIFICATION_FAILED" if outcome and outcome.verification_problems else "REPLAN"
    return f"{step.id if step else 'unknown_step'}:{code}"


def _compact_replan_step_outputs(step_outputs: dict[str, Any]) -> dict[str, Any]:
    """只保留 Replanner 所需的输出引用，不把完整 Tool output 带入失败上下文。"""

    compact: dict[str, Any] = {}
    for step_id, value in step_outputs.items():
        if not isinstance(value, dict):
            continue
        compact[step_id] = {
            key: value[key]
            for key in ("dataset_id", "dataset_ids", "artifact_ids", "status")
            if key in value
        }
    return compact


def _plan_failure_result(task, run, outcome, error: str, summary: str | None = None) -> AgentResult:
    return AgentResult(
        agent_id="main",
        task_id=task.id,
        status=AgentResultStatus.BLOCKED if error.startswith(("WAITING_USER", "REPLAN")) else AgentResultStatus.FAILED,
        summary=summary or error,
        findings=list(outcome.findings),
        datasets=list(outcome.output_ids),
        artifacts=list(outcome.artifacts),
        warnings=list(outcome.errors[:-1]),
        error=error,
        trace_id=run.id,
    )


def _aggregate_subagent_directive(executions: list[Any]) -> LoopDirective:
    """按必需子任务聚合控制信号：用户信息优先于重新规划，再优先于终止。"""

    directives = {item.directive for item in executions}
    if LoopDirective.ASK_USER in directives:
        return LoopDirective.ASK_USER
    if LoopDirective.REPLAN in directives:
        return LoopDirective.REPLAN
    if LoopDirective.ABORT in directives:
        return LoopDirective.ABORT
    return LoopDirective.CONTINUE


def _agent_result_status(value: Any, *, default: AgentResultStatus) -> AgentResultStatus:
    if isinstance(value, AgentResultStatus):
        return value
    try:
        return AgentResultStatus(str(value)) if value else default
    except ValueError:
        return default


def _decision_from_agent_result(result: AgentResult, *, source: str) -> AgentDecision:
    if result.status is AgentResultStatus.BLOCKED:
        return AgentDecision(
            type=DecisionType.ASK_USER,
            reasoning_summary=result.summary,
            final_response=result.summary,
            source=source,
            metadata={
                "status": result.status.value,
                "error": result.error,
                "findings": result.findings,
                "datasets": result.datasets,
                "artifacts": result.artifacts,
            },
        )
    if result.status in {AgentResultStatus.FAILED, AgentResultStatus.CANCELLED}:
        return AgentDecision(
            type=DecisionType.ABORT,
            reasoning_summary=result.summary,
            source=source,
            metadata={
                "status": result.status.value,
                "error": result.error,
                "findings": result.findings,
                "datasets": result.datasets,
                "artifacts": result.artifacts,
            },
        )
    return AgentDecision(
        type=DecisionType.FINAL,
        reasoning_summary=result.summary,
        final_response=result.summary,
        source=source,
        metadata={
            "status": result.status.value,
            "error": result.error,
            "findings": result.findings,
            "datasets": result.datasets,
            "artifacts": result.artifacts,
            "warnings": result.warnings,
        },
    )


def _subagent_result_view(subtask: SubTask, execution: SubAgentExecutionResult) -> dict[str, Any]:
    """把 SubAgent 结果裁成下一次 Decision 所需的轻量证据。"""

    result = execution.result
    return {
        "subtask_id": subtask.id,
        "agent_id": result.agent_id,
        "goal": subtask.goal,
        "operation": subtask.operation,
        "required": subtask.required,
        "status": result.status.value,
        "summary": result.summary,
        "directive": execution.directive.value,
        "dataset_ids": list(execution.working_memory_delta.added_dataset_ids),
        "artifact_ids": list(execution.working_memory_delta.added_artifact_ids),
        "key_findings": _compact_subagent_findings(result.findings),
        "failure_rationale": execution.failure_rationale or result.error,
    }


def _compact_subagent_findings(findings: list[Any]) -> list[Any]:
    compacted: list[Any] = []
    for item in findings[:6]:
        if isinstance(item, dict):
            compacted.append({str(key): _compact_subagent_value(value) for key, value in list(item.items())[:8]})
        else:
            compacted.append(_compact_subagent_value(item))
    return compacted


def _compact_subagent_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 2:
        return str(value)[:240]
    if isinstance(value, dict):
        return {str(key): _compact_subagent_value(item, depth=depth + 1) for key, item in list(value.items())[:8]}
    if isinstance(value, (list, tuple)):
        return [_compact_subagent_value(item, depth=depth + 1) for item in list(value)[:8]]
    if isinstance(value, str):
        return value if len(value) <= 500 else value[:499] + "…"
    return value


def _delegation_failure(executions: list[SubAgentExecutionResult], tasks: list[SubTask]) -> dict[str, Any] | None:
    for subtask, execution in zip(tasks, executions, strict=True):
        if not subtask.required or execution.directive is LoopDirective.CONTINUE:
            continue
        return {
            "directive": execution.directive.value,
            "action": execution.directive.value,
            "deterministic": True,
            "subtask_id": subtask.id,
            "tool_name": subtask.operation,
            "error": execution.failure_rationale or execution.result.error or execution.result.summary,
            "recovery_action": execution.directive.value,
            "attempts": 1,
        }
    return None


def _merge_subagent_views(current: list[Any], incoming: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    merged = [dict(item) for item in current if isinstance(item, dict)]
    positions = {(item.get("subtask_id"), item.get("agent_id")): index for index, item in enumerate(merged)}
    for item in incoming:
        key = (item.get("subtask_id"), item.get("agent_id"))
        if key in positions:
            merged[positions[key]] = dict(item)
        else:
            positions[key] = len(merged)
            merged.append(dict(item))
    return merged


def _delegation_fingerprint(tasks: list[SubTask]) -> str:
    payload = sorted(
        [
            {
                "goal": task.goal,
                "operation": task.operation,
                "dataset_ids": sorted(task.dataset_ids),
                "dependencies": sorted(task.dependencies),
                "required": task.required,
            }
            for task in tasks
        ],
        key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True),
    )
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def _working_memory_refs(memory: WorkingMemory | None) -> dict[str, Any] | None:
    if memory is None:
        return None
    return {
        "task_id": memory.task_id,
        "conversation_id": memory.conversation_id,
        "active_dataset_ids": list(memory.active_dataset_ids),
        "active_artifact_ids": list(memory.active_artifact_ids),
        "unresolved_questions": list(memory.unresolved_questions),
    }


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


def _observation_from_checkpoint(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("tool_observations"), list):
        return {"tool_observations": [dict(item) for item in value["tool_observations"] if isinstance(item, dict)]}
    if "call_id" in value or "status" in value:
        # 兼容早期只保存单个 ToolResult 的 checkpoint。
        item = dict(value)
        item.setdefault("accepted", item.get("status") in {ToolStatus.SUCCESS.value, ToolStatus.PARTIAL_SUCCESS.value})
        item.setdefault("verified", item.get("accepted", False))
        item.setdefault("verification_problems", [])
        item.setdefault("directive", LoopDirective.CONTINUE.value if item.get("accepted") else LoopDirective.ABORT.value)
        item.setdefault("attempts", 1)
        return _batch_observation([item])
    return dict(value)


def _execution_observation(outcome: ExecutionOutcome) -> dict[str, Any]:
    """给下一轮模型的轻量执行观察，不把 ExecutionOutcome 全量暴露出去。"""

    observation = outcome.result.model_dump(mode="json")
    observation.update(
        {
            "accepted": outcome.accepted,
            "verified": outcome.verified,
            "verification_problems": list(outcome.verification_problems),
            "recovery_action": outcome.recovery_action.value if outcome.recovery_action else None,
            "directive": outcome.directive.value,
            "attempts": outcome.attempts,
            "rationale": outcome.rationale,
        }
    )
    return {key: value for key, value in observation.items() if value not in (None, [], {})}


def _invalid_tool_call_observation(call_id: str, result: ToolResult, detail: str) -> dict[str, Any]:
    observation = result.model_dump(mode="json")
    observation.update(
        {
            "call_id": call_id,
            "accepted": False,
            "verified": False,
            "verification_problems": [f"模型工具参数无效：{detail}"],
            "recovery_action": None,
            "directive": LoopDirective.ABORT.value,
            "attempts": 1,
        }
    )
    return {key: value for key, value in observation.items() if value not in (None, [], {})}


def _batch_observation(items: list[dict[str, Any]]) -> dict[str, Any]:
    """统一构造本轮 Tool Batch Observation，并保留单 Tool checkpoint 兼容字段。"""

    batch = {"tool_observations": items}
    if len(items) == 1:
        # 旧前端和旧 checkpoint 检查仍可能直接读取 call_id；权威内容仍是 batch。
        batch["call_id"] = items[0].get("call_id")
    return batch


def _observation_items(observation: dict[str, Any]) -> list[dict[str, Any]]:
    batch = observation.get("tool_observations")
    if isinstance(batch, list):
        return [item for item in batch if isinstance(item, dict)]
    return [observation]


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

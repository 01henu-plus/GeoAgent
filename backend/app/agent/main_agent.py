"""GeoAgent 的 Main Agent Loop。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
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
    ToolResult,
    WorkingMemory,
    new_id,
)
from app.decision import (
    CONTROL_CAPABILITY_DEFINITIONS,
    DecisionEngine,
    FailureAnalyzer,
    Planner,
    Replanner,
    ResultVerifier,
    TaskDecomposer,
)
from app.decision.model_provider import ModelDecisionProvider
from app.decision.offline_provider import OfflineDecisionProvider
from app.events import EventType
from app.execution.tools import ToolExecutor
from app.knowledge import KnowledgeRetriever
from app.memory import MemoryManager
from app.memory.extractor import MemoryExtractor
from app.models import ModelAdapter
from app.observability import TraceRecorder
from app.profile import ProfilePreferenceExtractor, UserProfileService
from app.run.lifecycle import PreparedRequest, RequestLifecycleBinder
from app.runtime.action_dispatcher import RuntimeActionDispatcher
from app.runtime.action_handlers import RuntimeActionHandlers
from app.runtime.agent_runtime import AgentRuntime
from app.runtime.agent_state import AgentStateBuilder
from app.runtime.budget import BudgetExceeded, BudgetGuard
from app.runtime.checkpoint_codec import RuntimeCheckpointCodec
from app.runtime.context_manager import ContextManager
from app.runtime.controller import AgentRuntimeController
from app.runtime.lifecycle import finish_run
from app.runtime.model_input_budget import ModelInputBudget
from app.runtime.plan_execution import next_executable_step
from app.runtime.protocol_history import compact_protocol_messages
from app.runtime.tool_execution_cycle import ToolExecutionCycle
from app.state import StateStore, WorkingMemoryUpdater
from app.task.service import TaskService
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
        self.request_understanding = RequestUnderstandingPipeline(store, interpreter=RequestInterpreter())
        self.planner = Planner()
        self.replanner = Replanner(self.planner)
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
        self.state_builder = AgentStateBuilder(store)
        self.decision_engine = DecisionEngine()
        self.agent_runtime = AgentRuntime(max_runtime_transitions=self.budget.max_runtime_transitions)
        self.lifecycle_binder = RequestLifecycleBinder(store, task_service)
        self.runtime_checkpoint_codec = RuntimeCheckpointCodec()
        self.runtime_action_handlers = RuntimeActionHandlers(
            store=self.store,
            trace=self.trace,
            budget=self.budget,
            guard=self.guard,
            planner=lambda: self.planner,
            replanner=lambda: self.replanner,
            decomposer=self.decomposer,
            agent_manager=lambda: self.agent_manager,
            task_service=self.task_service,
            tool_execution_cycle=lambda: self.tool_execution_cycle,
            working_memory_updater=self.working_memory_updater,
            registry=self.registry,
            default_crs=self.settings.default_crs,
            checkpoint=lambda: self._checkpoint,
            checkpoint_codec=self.runtime_checkpoint_codec,
        )
        self.model_decision_provider = ModelDecisionProvider(
            decision_engine=self.decision_engine,
            build_messages=self._build_model_messages,
            refresh_datasets=self._refresh_model_datasets,
            resolve_request_resources=self._resolve_request_resources,
            max_tokens=self.budget.max_tokens,
        )
        self.offline_decision_provider = OfflineDecisionProvider(
            decomposer=self.decomposer,
            knowledge=self.knowledge,
            next_executable_step=next_executable_step,
            result_to_decision=lambda result, source: _decision_from_agent_result(result, source=source),
            diagnose_runs=self._diagnose_runs,
            interpret_result=self._interpret_result,
            conversation_reply=_conversation_reply,
        )
        self.runtime_action_dispatcher = RuntimeActionDispatcher(
            handlers={
                DecisionType.TOOL: self.runtime_action_handlers.handle_tool,
                DecisionType.PLAN: self.runtime_action_handlers.handle_plan,
                DecisionType.REPLAN: self.runtime_action_handlers.handle_replan,
                DecisionType.DELEGATE: self.runtime_action_handlers.handle_delegate,
            },
        )
        self.runtime_controller = AgentRuntimeController(
            store=self.store,
            budget=self.budget,
            guard=self.guard,
            runtime=self.agent_runtime,
            state_builder=self.state_builder,
            decision_engine=self.decision_engine,
            model_provider=self.model_decision_provider,
            offline_provider=self.offline_decision_provider,
            dispatcher=self.runtime_action_dispatcher,
            model_adapter_for=self._model_adapter_for,
            trace_decision=self._trace_runtime_decision,
            checkpoint=self._checkpoint,
            checkpoint_codec=self.runtime_checkpoint_codec,
            plan_fast_path=self.runtime_action_handlers.execute_plan_step,
        )

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
        runtime_resume = self.runtime_checkpoint_codec.decode(resume_state)
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
            if resume_from and (resume_state.get("current_plan") or resume_state.get("plan")):
                plan = runtime_resume.current_plan
                phase = resume_from.phase
                await self.trace.emit(run.id, EventType.RESUME_STARTED, f"从 Checkpoint 继续：{resume_from.phase}", payload={"checkpoint_id": resume_from.id, "phase": resume_from.phase}, agent_id="main")
            phase = "request_understood"
            await self.trace.emit(
                run.id,
                EventType.INTENT_RESOLVED,
                "请求理解完成",
                payload={
                    "request_frame": request_frame.model_dump(mode="json"),
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
            self.guard.check_execution_time(run)
            # Model 与 Offline 只选择不同的 Decision Provider；执行控制统一进入
            # AgentRuntimeController。
            phase = "runtime_started"
            outcome = await self.runtime_controller.run(
                request,
                run=run,
                task=task,
                datasets=datasets,
                plan=plan,
                request_frame=request_frame,
                working_memory=working_memory,
                resume=runtime_resume,
                on_model_delta=on_model_delta,
            )
            result = self._finalize_runtime_outcome(outcome, task=task, run=run)
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
            # 历史 checkpoint 可能带有旧意图字段；新 checkpoint 不再延续该字段。
            state.pop("intent", None)
            state.update(
                {
                    "request": request.model_dump(mode="json"),
                    "request_frame": request_frame.model_dump(mode="json") if request_frame else None,
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

    def _build_model_messages(
        self,
        request: AgentRequest,
        run: Run,
        task: Task | None,
        datasets,
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

    async def _checkpoint(self, run_id: str, phase: str, state: dict[str, Any]) -> None:
        if not self.checkpoint_store:
            return
        checkpoint = make_checkpoint(run_id, phase, state)
        self.checkpoint_store.save(checkpoint)
        await self.trace.emit(run_id, EventType.CHECKPOINT_SAVED, f"保存 Checkpoint：{phase}", payload={"checkpoint_id": checkpoint.id, "phase": phase}, agent_id="main")

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


def _observation_items(observation: dict[str, Any]) -> list[dict[str, Any]]:
    batch = observation.get("tool_observations")
    if isinstance(batch, list):
        return [item for item in batch if isinstance(item, dict)]
    return [observation]


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

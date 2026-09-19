"""GeoAgent 的领域模型。

这些模型描述 GIS Agent 的事实边界：请求、任务、运行、工具结果、数据集、
产物和追踪事件。执行器只接受/返回这些结构化对象，避免把异常字符串直接
当成成功结果交给上层 Agent。
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def new_id(prefix: str) -> str:
    """生成可读且不会依赖数据库自增的领域 ID。"""

    return f"{prefix}_{uuid4().hex[:12]}"


def utc_now() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, protected_namespaces=(), populate_by_name=True, serialize_by_alias=True)


class DatasetKind(StrEnum):
    VECTOR = "VECTOR"
    RASTER = "RASTER"
    TABLE = "TABLE"
    POINT_CLOUD = "POINT_CLOUD"
    TRAJECTORY = "TRAJECTORY"
    NETWORK = "NETWORK"
    SERVICE = "SERVICE"


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    SUCCEEDED = "SUCCEEDED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"


class RunStatus(StrEnum):
    CREATED = "CREATED"
    PLANNING = "PLANNING"
    RUNNING = "RUNNING"
    WAITING_TOOL = "WAITING_TOOL"
    WAITING_SUBAGENT = "WAITING_SUBAGENT"
    WAITING_USER = "WAITING_USER"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    RETRYING = "RETRYING"
    REPLANNING = "REPLANNING"
    VALIDATING = "VALIDATING"
    COMPLETED = "COMPLETED"
    PARTIAL_COMPLETED = "PARTIAL_COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"
    CANCELLED = "CANCELLED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"


class IntentType(StrEnum):
    DATA_INSPECTION = "DATA_INSPECTION"
    SPATIAL_ANALYSIS = "SPATIAL_ANALYSIS"
    DATA_TRANSFORMATION = "DATA_TRANSFORMATION"
    RESULT_INTERPRETATION = "RESULT_INTERPRETATION"
    CODE_TASK = "CODE_TASK"
    RUN_DIAGNOSIS = "RUN_DIAGNOSIS"
    KNOWLEDGE_QUERY = "KNOWLEDGE_QUERY"
    UNKNOWN = "UNKNOWN"


class InteractionMode(StrEnum):
    """用户与当前任务之间的关系，不表示具体 GIS 能力。"""

    NEW_TASK = "new_task"
    CONTINUE_TASK = "continue_task"
    MODIFY_TASK = "modify_task"
    RETRY_TASK = "retry_task"
    QUERY = "query"
    CHAT = "chat"
    CANCEL_TASK = "cancel_task"


class ResponseStyle(StrEnum):
    CONCISE = "concise"
    BALANCED = "balanced"
    DETAILED = "detailed"


class MeasurementSystem(StrEnum):
    METRIC = "metric"
    IMPERIAL = "imperial"


class DecisionType(StrEnum):
    TOOL = "TOOL"
    PLAN = "PLAN"
    DELEGATE = "DELEGATE"
    REPLAN = "REPLAN"
    ASK_USER = "ASK_USER"
    FINAL = "FINAL"
    ABORT = "ABORT"


class ToolStatus(StrEnum):
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


class DatasetOutputPolicy(StrEnum):
    """Tool 对 Dataset 输出的契约强度。"""

    NONE = "NONE"
    OPTIONAL = "OPTIONAL"
    REQUIRED = "REQUIRED"


class ErrorCategory(StrEnum):
    INPUT = "INPUT"
    CRS = "CRS"
    GEOMETRY = "GEOMETRY"
    DATA = "DATA"
    RASTER = "RASTER"
    RESOURCE = "RESOURCE"
    PERMISSION = "PERMISSION"
    EXECUTION = "EXECUTION"
    EXTERNAL = "EXTERNAL"
    UNKNOWN = "UNKNOWN"


class FailureAction(StrEnum):
    RETRY = "RETRY"
    REPAIR = "REPAIR"
    REPLAN = "REPLAN"
    ASK_USER = "ASK_USER"
    ABORT = "ABORT"


class LoopDirective(StrEnum):
    """执行结果交给上层循环时的下一步控制信号。"""

    CONTINUE = "CONTINUE"
    ASK_USER = "ASK_USER"
    REPLAN = "REPLAN"
    ABORT = "ABORT"


class RiskLevel(StrEnum):
    READ = "READ"
    WRITE = "WRITE"
    DESTRUCTIVE = "DESTRUCTIVE"
    EXTERNAL = "EXTERNAL"


class AgentResultStatus(StrEnum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"


class RequestResolutionStatus(StrEnum):
    """RequestFrame 的状态解析结果，不等同于模型置信度。"""

    RESOLVED = "resolved"
    NEEDS_CLARIFICATION = "needs_clarification"
    INVALID = "invalid"


class ArtifactKind(StrEnum):
    DATASET = "DATASET"
    MAP = "MAP"
    REPORT = "REPORT"
    TABLE = "TABLE"
    LOG = "LOG"
    OTHER = "OTHER"


class CRSInfo(StrictModel):
    authority: str | None = None
    name: str | None = None
    is_geographic: bool = False
    linear_unit: str | None = None


class BoundingBox(StrictModel):
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @field_validator("max_x")
    @classmethod
    def max_x_not_before_min_x(cls, value: float, info: Any) -> float:
        if "min_x" in info.data and value < info.data["min_x"]:
            raise ValueError("max_x must be >= min_x")
        return value

    @field_validator("max_y")
    @classmethod
    def max_y_not_before_min_y(cls, value: float, info: Any) -> float:
        if "min_y" in info.data and value < info.data["min_y"]:
            raise ValueError("max_y must be >= min_y")
        return value


class DatasetSchema(StrictModel):
    fields: dict[str, str] = Field(default_factory=dict)
    geometry_type: str | None = None
    feature_count: int | None = Field(default=None, ge=0)
    width: int | None = Field(default=None, ge=0)
    height: int | None = Field(default=None, ge=0)
    bands: int | None = Field(default=None, ge=0)
    resolution: tuple[float, float] | None = None
    nodata: float | int | None = None
    invalid_geometry_count: int | None = Field(default=None, ge=0)


class Dataset(StrictModel):
    id: str = Field(default_factory=lambda: new_id("ds"))
    name: str
    kind: DatasetKind
    path: str
    format: str
    crs: CRSInfo | None = None
    extent: BoundingBox | None = None
    schema_: DatasetSchema | None = Field(default=None, alias="schema")
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_dataset_ids: list[str] = Field(default_factory=list)
    created_by_run_id: str | None = None
    owner_user_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("name", "path", "format")
    @classmethod
    def required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("dataset text fields cannot be empty")
        return value

    @property
    def schema(self) -> DatasetSchema | None:
        return self.schema_

    def model_dump(self, *args, **kwargs):
        kwargs.setdefault("by_alias", True)
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args, **kwargs):
        kwargs.setdefault("by_alias", True)
        return super().model_dump_json(*args, **kwargs)


class Artifact(StrictModel):
    id: str = Field(default_factory=lambda: new_id("art"))
    name: str
    kind: ArtifactKind
    path: str | None = None
    media_type: str | None = None
    dataset_id: str | None = None
    run_id: str | None = None
    owner_user_id: str | None = None
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class AgentRequest(StrictModel):
    request_id: str = Field(default_factory=lambda: new_id("req"))
    conversation_id: str = Field(default_factory=lambda: new_id("conv"))
    user_id: str | None = None
    user_input: str
    dataset_ids: list[str] = Field(default_factory=list)
    attachment_ids: list[str] = Field(default_factory=list)
    referenced_run_ids: list[str] = Field(default_factory=list)
    model_profile: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)

    @field_validator("user_input")
    @classmethod
    def non_empty_input(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("user_input cannot be empty")
        return value


class Task(StrictModel):
    id: str = Field(default_factory=lambda: new_id("task"))
    goal: str
    status: TaskStatus = TaskStatus.PENDING
    subtasks: list[str] = Field(default_factory=list)
    result: str | None = None
    conversation_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SubTask(StrictModel):
    id: str = Field(default_factory=lambda: new_id("sub"))
    goal: str
    description: str
    operation: str | None = None
    dataset_ids: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    parallelizable: bool = True
    required: bool = True
    assigned_agent_id: str | None = None
    failure_policy: str = "continue_if_optional"
    status: TaskStatus = TaskStatus.PENDING


class ToolMetadata(StrictModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    deterministic: bool = True
    idempotent: bool = True
    risk_level: RiskLevel = RiskLevel.READ
    supports_retry: bool = False
    produces_dataset: bool = False
    dataset_output_policy: DatasetOutputPolicy | None = None
    produces_artifact: bool = False
    tags: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        """兼容旧注册代码，同时让输出策略成为验证时的权威字段。"""

        if self.dataset_output_policy is None:
            object.__setattr__(
                self,
                "dataset_output_policy",
                DatasetOutputPolicy.REQUIRED if self.produces_dataset else DatasetOutputPolicy.NONE,
            )


class ToolCall(StrictModel):
    id: str = Field(default_factory=lambda: new_id("call"))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None
    agent_id: str | None = None
    attempt: int = Field(default=1, ge=1)


class ToolError(StrictModel):
    code: str
    category: ErrorCategory = ErrorCategory.UNKNOWN
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class ToolResult(StrictModel):
    call_id: str
    status: ToolStatus
    output: Any = None
    error: ToolError | None = None
    warnings: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    retryable: bool = False
    duration_ms: float = Field(default=0.0, ge=0.0)


class AgentDecision(StrictModel):
    type: DecisionType
    reasoning_summary: str
    tool_call: ToolCall | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    plan_goal: str | None = None
    subtasks: list[SubTask] = Field(default_factory=list)
    final_response: str | None = None
    source: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """保留旧的单工具字段，同时让批量工具调用成为权威表示。"""

        if self.tool_call is not None and not self.tool_calls:
            self.tool_calls = [self.tool_call]
        elif self.tool_calls and self.tool_call is None:
            self.tool_call = self.tool_calls[0]

    def normalized_tool_calls(self) -> list[ToolCall]:
        """返回兼容旧调用方的工具批次视图。"""

        return list(self.tool_calls or ([self.tool_call] if self.tool_call else []))


class AgentResult(StrictModel):
    agent_id: str
    task_id: str | None = None
    status: AgentResultStatus
    summary: str
    findings: list[Any] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    evidence: list[Any] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    trace_id: str


class Run(StrictModel):
    id: str = Field(default_factory=lambda: new_id("run"))
    parent_run_id: str | None = None
    conversation_id: str | None = None
    task_id: str | None = None
    agent_id: str
    status: RunStatus = RunStatus.CREATED
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    turn_count: int = 0
    tool_call_count: int = 0
    replan_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RequestResources(StrictModel):
    """本轮请求显式带入的真实资源，不写入长期状态。"""

    datasets: list[Dataset] = Field(default_factory=list)
    runs: list[Run] = Field(default_factory=list)


class WorkingMemoryItem(StrictModel):
    """工作记忆中的轻量引用，不保存完整工具输出。"""

    kind: str
    reference_id: str | None = None
    summary: str
    source_run_id: str | None = None


class WorkingMemory(StrictModel):
    """以 Task 为作用域的结构化工作状态。"""

    task_id: str
    conversation_id: str | None = None
    active_dataset_ids: list[str] = Field(default_factory=list)
    active_artifact_ids: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    intermediate_results: list[WorkingMemoryItem] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utc_now)


class WorkingMemoryDelta(StrictModel):
    """SubAgent 本轮执行产生的局部工作状态变化，不直接持久化。"""

    added_dataset_ids: list[str] = Field(default_factory=list)
    added_artifact_ids: list[str] = Field(default_factory=list)
    intermediate_results: list[WorkingMemoryItem] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    added_assumptions: list[str] = Field(default_factory=list)
    source_run_id: str | None = None


class SubAgentExecutionResult(StrictModel):
    """SubAgent 的业务结果与局部 WorkingMemory 变化。"""

    result: AgentResult
    working_memory_delta: WorkingMemoryDelta = Field(default_factory=WorkingMemoryDelta)
    directive: LoopDirective = LoopDirective.CONTINUE
    failure_rationale: str | None = None


class Checkpoint(StrictModel):
    id: str = Field(default_factory=lambda: new_id("cp"))
    run_id: str
    phase: str
    state: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class TraceEvent(StrictModel):
    id: str = Field(default_factory=lambda: new_id("evt"))
    run_id: str
    event_type: str
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    sequence: int = 0
    timestamp: datetime = Field(default_factory=utc_now)
    agent_id: str | None = None


class Conversation(StrictModel):
    id: str = Field(default_factory=lambda: new_id("conv"))
    title: str = "新对话"
    user_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Message(StrictModel):
    id: str = Field(default_factory=lambda: new_id("msg"))
    conversation_id: str
    role: str
    content: str
    run_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class ConversationMemoryEntry(StrictModel):
    """会话级派生事实，始终保留其来源引用。"""

    id: str = Field(default_factory=lambda: new_id("cmem"))
    content: str
    source_message_id: str | None = None
    source_task_id: str | None = None
    source_run_id: str | None = None
    reference_type: str | None = None
    reference_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class ConversationMemory(StrictModel):
    """Conversation 作用域的结构化派生状态，不替代原始 Messages。"""

    conversation_id: str
    user_id: str
    summary: str = ""
    key_facts: list[ConversationMemoryEntry] = Field(default_factory=list)
    decisions: list[ConversationMemoryEntry] = Field(default_factory=list)
    important_references: list[ConversationMemoryEntry] = Field(default_factory=list)
    unresolved_topics: list[ConversationMemoryEntry] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utc_now)


class MemoryItem(StrictModel):
    id: str = Field(default_factory=lambda: new_id("mem"))
    owner_user_id: str | None = None
    scope: str = "project"
    key: str
    value: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=utc_now)


class User(StrictModel):
    """认证和资源归属使用的最小用户身份。"""

    id: str = Field(default_factory=lambda: new_id("user"))
    username: str
    email: str | None = None
    password_hash: str
    display_name: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class UserProfile(StrictModel):
    """跨会话的明确工作交互偏好，不保存用户画像或敏感信息。"""

    user_id: str
    language: str = "zh-CN"
    response_style: ResponseStyle = ResponseStyle.BALANCED
    measurement_system: MeasurementSystem = MeasurementSystem.METRIC
    preferred_output_format: str | None = None
    updated_at: datetime = Field(default_factory=utc_now)


class UserView(StrictModel):
    """可返回给前端的安全用户视图，不包含密码哈希。"""

    id: str
    username: str
    email: str | None = None
    display_name: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_user(cls, user: User) -> UserView:
        return cls.model_validate(user.model_dump(exclude={"password_hash"}))


class UserSession(StrictModel):
    """服务端 Session 记录；token 只以哈希形式持久化。"""

    id: str = Field(default_factory=lambda: new_id("session"))
    user_id: str
    token_hash: str
    expires_at: datetime
    created_at: datetime = Field(default_factory=utc_now)
    last_seen_at: datetime | None = None


class ResolvedReference(StrictModel):
    """请求中的上下文指代及其已验证的领域对象。"""

    mention: str
    type: str
    target_id: str | None = None
    label: str | None = None
    confidence: float = Field(default=1.0, ge=0, le=1)


class StateSnapshot(StrictModel):
    """供请求理解读取的轻量状态视图，不替代 Task/Run/Memory。"""

    conversation_id: str
    active_task_id: str | None = None
    active_run_id: str | None = None
    task_goal: str | None = None
    task_status: TaskStatus | None = None
    last_run_status: RunStatus | None = None
    last_action: str | None = None
    last_result: str | None = None
    last_error: str | None = None
    recent_messages: list[Message] = Field(default_factory=list)
    recent_runs: list[Run] = Field(default_factory=list)
    recent_artifacts: list[Artifact] = Field(default_factory=list)
    recent_datasets: list[Dataset] = Field(default_factory=list)
    known_task_ids: list[str] = Field(default_factory=list)
    known_run_ids: list[str] = Field(default_factory=list)
    known_artifact_ids: list[str] = Field(default_factory=list)
    known_dataset_ids: list[str] = Field(default_factory=list)


class RequestFrame(StrictModel):
    """状态感知的请求表达，也是后续决策层的唯一请求语义来源。"""

    mode: InteractionMode
    goal: str
    operations: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    dataset_roles: list[str] = Field(default_factory=list)
    render_requested: bool = False
    references: list[ResolvedReference] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    target_task_id: str | None = None
    target_run_id: str | None = None
    needs_planning: bool = False
    needs_tool: bool = False
    unresolved_references: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0, le=1)
    resolution_status: RequestResolutionStatus = RequestResolutionStatus.RESOLVED
    blocking_issues: list[str] = Field(default_factory=list)


class RunBudget(StrictModel):
    """运行预算；``max_tokens`` 仅表示模型最大输出 token 数。"""

    max_agent_turns: int = Field(default=20, ge=1)
    max_runtime_transitions: int = Field(default=100, ge=1)
    max_tool_calls: int = Field(default=40, ge=1)
    max_retry_per_action: int = Field(default=2, ge=0)
    max_replans: int = Field(default=2, ge=0)
    max_subagents: int = Field(default=5, ge=0)
    max_parallel_agents: int = Field(default=3, ge=1)
    max_tokens: int = Field(default=1200, ge=1)
    # 模型工具声明包含 GIS 工具和内部控制能力，默认输入预算需覆盖其固定成本。
    model_input_tokens: int = Field(default=13000, ge=128)
    model_context_tokens: int = Field(default=6000, ge=128)
    protocol_history_tokens: int = Field(default=3000, ge=128)
    subagent_context_tokens: int = Field(default=3000, ge=128)
    max_execution_seconds: int = Field(default=300, ge=1)


class PlanStep(StrictModel):
    id: str = Field(default_factory=lambda: new_id("step"))
    title: str
    action: str
    depends_on: list[str] = Field(default_factory=list)
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    required: bool = True
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING


class Plan(StrictModel):
    id: str = Field(default_factory=lambda: new_id("plan"))
    goal: str
    intent: IntentType
    steps: list[PlanStep] = Field(default_factory=list)
    revision: int = 1
    clarification: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def executable_steps_have_tools(self) -> Plan:
        invalid = [step.id for step in self.steps if not step.tool_name]
        if invalid:
            raise ValueError(f"计划包含不可执行步骤：{', '.join(invalid)}")
        return self


class ReplanContext(StrictModel):
    """一次确定性 Replan 所需的最小失败上下文。"""

    goal: str
    original_plan: Plan
    current_plan: Plan
    current_revision: int
    completed_steps: list[str] = Field(default_factory=list)
    step_outputs: dict[str, Any] = Field(default_factory=dict)
    failed_step: PlanStep | None = None
    failed_tool_name: str | None = None
    failed_arguments: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    verification_problems: list[str] = Field(default_factory=list)
    recovery_action: FailureAction | None = None
    directive: LoopDirective = LoopDirective.REPLAN
    attempts: int = 1
    current_dataset_ids: list[str] = Field(default_factory=list)
    current_artifact_ids: list[str] = Field(default_factory=list)
    replan_count: int = 0
    previous_replan_reasons: list[str] = Field(default_factory=list)


__all__ = [name for name in globals() if not name.startswith("_")]

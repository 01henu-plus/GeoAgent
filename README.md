# GeoAgent

> 面向 GIS 场景的智能辅助 Agent。默认由 Main Agent 完成任务；当复杂任务可以拆分为多个相互独立或弱依赖的子任务时，Main Agent 动态创建临时 SubAgent 并行执行，最后统一验证、汇总和回答。

---

## 1. 项目定位

GeoAgent 不是一个 GIS 平台，也不是一个固定工作流系统，而是一个 **GIS 垂直领域的智能辅助 Agent 项目**。

项目主要面向未来实习 / 校招 / 面试展示，希望同时体现：

- LLM Agent 设计能力
- Agent Loop 与状态机设计能力
- 动态任务拆解与并行 SubAgent 调度能力
- Tool Calling 与执行治理能力
- 失败恢复与长期任务执行能力
- GIS 专业知识与空间数据处理能力
- Context Engineering 能力
- Run / Checkpoint / Trace / Artifact 等工程能力

GeoAgent 的核心目标不是“做一个功能很多的平台”，而是实现一个 **架构完整、执行闭环清晰、GIS 特征明显、能够深入讲解的 Agent Harness**。

### 一句话定义

> GeoAgent is a GIS-oriented intelligent assistant agent with dynamic task decomposition, parallel sub-agents, tool execution, failure recovery, persistent state, and geospatial reasoning.

---

## 2. 明确不做什么

GeoAgent 当前不计划实现以下能力：

- Computer Agent
- macOS Native Helper
- Screen Capture
- Accessibility API
- Mouse / Keyboard Control
- Window Control
- Clipboard Automation
- Browser GUI Automation
- 任意第三方 App GUI 操作
- QGIS / ArcGIS 的鼠标点击式自动化
- 复杂商业级多租户平台能力
- 重型 Workflow Platform
- 无限层级的 Multi-Agent 网络

如果未来需要访问外部 GIS 服务，优先通过 API / Connector / CLI，而不是通过 GUI 自动操作。

---

## 3. 核心设计原则

### 3.1 四层主架构

整个系统按照四个层次组织：

```text
┌──────────────────────────────────────────────┐
│              1. Task Entry Layer             │
│                                              │
│ User / Conversation / Dataset / Attachment   │
│ Request Normalize / Reference Resolution     │
└─────────────────────┬────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│              2. Decision Layer               │
│                                              │
│ Intent                                       │
│ Planning                                     │
│ Task Decomposition                           │
│ Parallelism Analysis                         │
│ Tool / Agent Routing                         │
│ Failure Analysis                             │
│ Result Verification                          │
└─────────────────────┬────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│              3. Execution Layer              │
│                                              │
│ Main/Sub Agent Runtime                       │
│ GIS Tools                                    │
│ Python Executor                              │
│ Shell Executor                               │
│ Knowledge Retrieval                          │
│ External Connectors                          │
└─────────────────────┬────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│          4. Memory & State Layer             │
│                                              │
│ Conversation / Task / Run                    │
│ Agent State / Working Memory                 │
│ Checkpoint / Dataset / Artifact              │
│ Project Memory / Trace                       │
└─────────────────────┬────────────────────────┘
                      │
                      └──────────────→ Decision
```

这四层不是一次性流水线，而是由 Agent Loop 反复循环：

```text
Entry
  ↓
Decision
  ↓
Execution
  ↓
State Update
  ↓
Observe
  ↓
Decision
  ↓
Execution
  ↓
...
  ↓
Final Answer
```

### 3.2 Agent Loop 是系统核心

GeoAgent 不是简单的：

```text
LLM → Tool → LLM → Tool
```

而是：

```text
THINK
  ↓
PLAN
  ↓
ACT
  ↓
OBSERVE
  ↓
DIAGNOSE
  ↓
VERIFY
  ↓
CONTINUE / REPLAN / FINISH
```

完整逻辑：

```text
             ┌─────────────────┐
             │      START      │
             └────────┬────────┘
                      ▼
                 Load State
                      │
                      ▼
                    THINK
                      │
                      ▼
                    PLAN
                      │
        ┌─────────────┼──────────────┐
        │             │              │
       Tool        Delegate         Final
        │             │              │
        ▼             ▼              ▼
     Execute       SubAgents       Verify
        │             │              │
        └──────┬──────┘              │
               ▼                     │
            OBSERVE                  │
               │                     │
        ┌──────┴───────┐             │
        │              │             │
      success         failed         │
        │              │             │
        │          DIAGNOSE           │
        │              │             │
        │      ┌───────┼────────┐    │
        │      │       │        │    │
        │    retry   repair   replan │
        │      │       │        │    │
        └──────┴───────┴────────┘    │
                      │              │
                      ▼              │
                    VERIFY ◄─────────┘
                      │
             ┌────────┴────────┐
             │                 │
          invalid             valid
             │                 │
          REPLAN            FINISH
```

---

## 4. Main Agent 与 SubAgent

### 4.1 Main Agent

Main Agent 是整个系统唯一面向用户的 Agent。

职责：

- 理解用户需求
- 读取当前 Conversation / Dataset / Memory
- 判断任务复杂度
- 制定 Plan
- 判断是否需要拆分任务
- 判断哪些子任务可以并行
- 创建临时 SubAgent
- 调用 Tool
- 分析执行失败
- 重新规划
- 验证结果
- 汇总 SubAgent 结果
- 输出最终回答

### 4.2 SubAgent

SubAgent 是 Main Agent 临时创建的执行单元。

职责：

- 接受一个明确的 SubTask
- 获取最小必要 Context
- 调用 GIS Tool / Python / Shell
- 自己完成局部 Agent Loop
- 对局部失败进行 Retry / Repair
- 验证局部结果
- 返回结构化 AgentResult

SubAgent 不允许：

- 直接回复用户
- 修改全局 Goal
- 任意修改长期 Memory
- 无限创建更多 Agent
- 获取与自己任务无关的全部 Context

第一版：

```text
max_agent_depth = 1
```

即只有 Main Agent 可以创建 SubAgent。

### 4.3 Multi-Agent 拓扑

采用 Hub-and-Spoke：

```text
                  Main Agent
              /       |        \
             /        |         \
        SubAgent A SubAgent B SubAgent C
             \        |         /
              \       |        /
               └── AgentResult ─┘
                       │
                       ▼
                  Main Agent
                   Synthesis
```

SubAgent 之间不直接通信。

---

## 5. 动态任务拆解与并行

复杂任务可以由 Main Agent 动态生成 SubTask。

示例：

```text
用户：
综合道路、人口、土地利用和 DEM，分析适合建设医院的区域。
```

可能拆解为：

```text
Task A: 道路可达性分析
Task B: 人口覆盖分析
Task C: 土地利用约束分析
Task D: 地形约束分析
Task E: 综合评价
```

依赖关系：

```text
A ───────┐
B ───────┼──→ E
C ───────┤
D ───────┘
```

A/B/C/D 可以并行。

第一版并行实现可直接使用：

```python
results = await asyncio.gather(
    *(agent_manager.run(task) for task in parallel_tasks),
    return_exceptions=True,
)
```

但必须有：

- max_parallel_agents
- timeout
- cancel
- partial failure handling
- per-agent budget

---

## 6. 四层详细设计

### 6.1 Task Entry Layer

职责：把用户原始输入转换成规范化 AgentRequest。

主要组件：

```text
entry/
├── request.py
├── conversation_service.py
├── attachment_service.py
├── dataset_resolver.py
├── reference_resolver.py
└── normalizer.py
```

需要处理：

- User Query
- Conversation
- Attachment
- Dataset
- Project Context
- Previous Run Reference
- “刚才的结果”
- “那个 DEM”
- “上一张图”
- “第二个数据”等自然语言引用

这一层负责回答：用户到底给了系统什么？

不负责回答：应该怎么做？

### 6.2 Decision Layer

整个系统最核心的智能决策层。

```text
decision/
├── intent.py
├── planner.py
├── decomposer.py
├── parallelism.py
├── router.py
├── tool_selector.py
├── failure_analyzer.py
├── verifier.py
└── response_planner.py
```

#### Intent Resolver

识别用户意图，例如：

```text
DATA_INSPECTION
SPATIAL_ANALYSIS
CODE_TASK
DATA_TRANSFORMATION
RUN_DIAGNOSIS
KNOWLEDGE_QUERY
RESULT_INTERPRETATION
```

第一版可以使用 LLM Structured Output。

#### Planner

生成当前任务总体 Plan。Plan 是动态的，可以在 Agent Loop 中修改。

#### Task Decomposer

判断是否需要拆成 SubTask。简单任务由 Main Agent 自己执行，复杂任务按独立性和依赖关系拆分。

#### Parallelism Analyzer

判断哪些任务可以并行：

```text
没有依赖关系 → 可并行
存在输出依赖 → 必须等待前置任务
```

#### Agent Router

决定 Main Agent 自己执行还是 Spawn SubAgent。

#### Tool Selector

工具选择优先级：

```text
        Task
         │
         ▼
   有领域 Tool？
     /      \
   有        无
   │         │
   ▼         ▼
GIS Tool   Python
              │
        CLI 更合适？
              │
              ▼
            Shell
```

领域 Tool 是主路，Python / Shell 是高级能力和逃生通道。

#### Failure Analyzer

执行失败后统一恢复动作：

```text
RETRY
REPAIR
REPLAN
ASK_USER
ABORT
```

典型映射：

```text
TIMEOUT → RETRY
RATE_LIMIT → RETRY
CRS_UNIT_MISMATCH → REPAIR
INVALID_GEOMETRY → REPAIR
MISSING_FIELD → REPLAN / ASK_USER
MISSING_DATASET → ASK_USER
ALGORITHM_NOT_APPLICABLE → REPLAN
NON_RECOVERABLE → ABORT
```

#### Result Verifier

Tool 成功不代表任务成功。需要验证：

- CRS 是否正确
- Geometry 是否有效
- Result 是否为空
- Feature Count 是否合理
- Raster NoData 是否异常
- Extent 是否正确
- Resolution 是否一致
- 必要字段是否存在
- 输出 Dataset 是否可读取

### 6.3 Execution Layer

原则：执行层只执行，不承担战略决策。

目录：

```text
execution/
├── tools/
├── python/
├── shell/
└── sandbox/
```

能力包括：

- GIS Tools
- Python Execution
- Shell Execution
- SQL（后续）
- Knowledge Retrieval
- External Connector
- SubAgent Parallel Execution

### 6.4 Memory & State Layer

需要明确：

```text
State ≠ Memory
```

State：当前正在发生什么。

Memory：过去有什么值得未来继续使用。

包含：

```text
state/
├── conversation/
├── task/
├── run/
├── agent/
├── checkpoint/
├── working_memory/
├── memory/
├── artifact/
└── trace/
```

---

## 7. Memory 设计

第一版划分三层：

### Working Memory

当前 Run 生命周期，例如：

```text
target_crs = EPSG:4547
buffer_distance = 500
candidate_dataset = ds_032
```

### Project Memory

当前 GIS 项目长期有效的信息，例如：

```text
项目默认 CRS = EPSG:4547
研究区域 = 深圳市南山区
人口字段 = POP2025
面积单位 = km²
默认输出格式 = GeoPackage
```

### Long-term Memory

跨项目用户偏好。第一版接口保留，实现可以简单。

---

## 8. Context Engineering

Conversation History 不等于 Agent Context。

Main Agent Context 推荐包含：

```text
System Policy
User Request
Task Goal
Current Plan
Working Memory
Relevant Conversation
Relevant Dataset Metadata
Relevant Tool Definitions
Relevant Project Memory
SubAgent Results
Current Errors
Budget State
```

SubAgent Context 必须隔离：

```text
SubTask
Required Dataset Metadata
Allowed Tools
Relevant Parent Findings
Local Working Memory
Budget
```

原则：Context Isolation。

---

## 9. 核心数据模型

### 9.1 AgentRequest

```python
class AgentRequest:
    request_id: str
    conversation_id: str
    user_input: str
    dataset_ids: list[str]
    attachment_ids: list[str]
    referenced_run_ids: list[str]
    context: dict
```

### 9.2 Task

```python
class Task:
    id: str
    goal: str
    status: TaskStatus
    subtasks: list["SubTask"]
    result: "AgentResult | None"
```

### 9.3 SubTask

```python
class SubTask:
    id: str
    goal: str
    description: str
    dependencies: list[str]
    parallelizable: bool
    required: bool
    assigned_agent_id: str | None
    failure_policy: FailurePolicy
    status: TaskStatus
```

### 9.4 AgentDecision

```python
class AgentDecision:
    type: DecisionType
    reasoning_summary: str
    tool_call: ToolCall | None
    subtasks: list[SubTask]
    final_response: str | None
```

DecisionType：

```text
TOOL
DELEGATE
REPLAN
ASK_USER
FINAL
ABORT
```

### 9.5 ToolCall

```python
class ToolCall:
    id: str
    name: str
    arguments: dict
```

### 9.6 ToolResult

```python
class ToolResult:
    call_id: str
    status: ToolStatus
    output: object | None
    error: ToolError | None
    warnings: list[str]
    datasets: list[str]
    artifacts: list[str]
    retryable: bool
```

ToolStatus：

```text
SUCCESS
PARTIAL_SUCCESS
FAILED
BLOCKED
CANCELLED
UNKNOWN
```

### 9.7 ToolError

```python
class ToolError:
    code: str
    category: ErrorCategory
    message: str
    retryable: bool
    details: dict
```

GIS 常见错误：

```text
CRS_MISMATCH
CRS_UNIT_MISMATCH
INVALID_GEOMETRY
MISSING_FIELD
EMPTY_DATASET
NO_OVERLAP
RASTER_ALIGNMENT_ERROR
NODATA_ERROR
UNSUPPORTED_FORMAT
OUT_OF_MEMORY
EXECUTION_TIMEOUT
```

### 9.8 AgentResult

```python
class AgentResult:
    agent_id: str
    task_id: str
    status: AgentResultStatus
    summary: str
    findings: list
    datasets: list[str]
    artifacts: list[str]
    evidence: list
    warnings: list[str]
    error: str | None
    trace_id: str
```

### 9.9 Run

```python
class Run:
    id: str
    parent_run_id: str | None
    conversation_id: str
    task_id: str
    agent_id: str
    status: RunStatus
    started_at: datetime | None
    finished_at: datetime | None
    error: str | None
```

Run Tree：

```text
run_100 Main Agent
│
├── run_101 SubAgent A
├── run_102 SubAgent B
└── run_103 SubAgent C
```

### 9.10 RunStatus

```text
CREATED
PLANNING
RUNNING
WAITING_TOOL
WAITING_SUBAGENT
WAITING_APPROVAL
WAITING_USER
RETRYING
REPLANNING
VALIDATING
COMPLETED
PARTIAL_COMPLETED
FAILED
INTERRUPTED
CANCELLED
BUDGET_EXCEEDED
```

### 9.11 Dataset

```python
class Dataset:
    id: str
    name: str
    kind: DatasetKind
    path: str
    format: str
    crs: CRSInfo | None
    extent: BoundingBox | None
    schema: dict | None
    metadata: dict
    source_dataset_ids: list[str]
    created_by_run_id: str | None
```

第一版 DatasetKind：

```text
VECTOR
RASTER
TABLE
```

预留：

```text
POINT_CLOUD
TRAJECTORY
NETWORK
SERVICE
```

### 9.12 Artifact

Dataset 与 Artifact 分开。

Dataset：能继续参与空间计算的数据。

Artifact：用户可查看、下载或消费的结果产物。

例如：

```text
candidate_sites.gpkg → Dataset + Artifact
map.png              → Artifact
report.md            → Artifact
statistics.csv       → Table Dataset + Artifact
```

---

## 10. Dataset Context

LLM 不直接读取大型 GIS 数据。

正确方式：

```text
原始 Dataset
    ↓
Dataset Inspector
    ↓
metadata / statistics / schema / samples
    ↓
Agent Context
```

Vector 示例：

```text
Dataset: roads
Type: Vector
Geometry: LineString
CRS: EPSG:4326
Feature Count: 12432
Fields:
- road_id
- name
- class
BBox: ...
Invalid Geometry: 31
```

Raster 示例：

```text
Dataset: dem
Type: Raster
CRS: EPSG:32650
Width: 5020
Height: 4230
Resolution: 30m
Bands: 1
NoData: -9999
```

---

## 11. Dataset Lineage

即使第一版实现简单，也需要保留。

```text
roads.shp
   │
   │ reproject
   ▼
roads_projected.gpkg
   │
   │ buffer
   ▼
roads_buffer.gpkg
```

记录：

```text
input_dataset_ids
output_dataset_id
operation
run_id
tool_call_id / code_execution_id
parameters
```

Python / Shell 执行生成新 Dataset 时也要进入 Dataset Registry 和 Lineage。

---

## 12. Tool System

Tool 必须强类型、可追踪、可治理。

```python
class ToolMetadata:
    name: str
    description: str
    deterministic: bool
    idempotent: bool
    risk_level: RiskLevel
    supports_retry: bool
    produces_dataset: bool
    produces_artifact: bool
```

Tool 调用链：

```text
Agent
  ↓
Tool Registry
  ↓
Permission
  ↓
Tool Executor
  ↓
GIS Service / Python / Shell
  ↓
ToolResult
  ↓
Trace / State / Lineage
```

---

## 13. GIS Tool 与 Python / Shell

三种能力都保留。

```text
                Agent Capabilities

         ┌──────────────┼───────────────┐
         ▼              ▼               ▼

      GIS Tools      Code Runtime     Connectors
    高层领域能力      通用计算能力      外部系统能力

    vector.*          python          OGC
    raster.*          shell           PostGIS
    crs.*             sql             STAC
    analysis.*                        ArcGIS REST
```

原则：GIS Tool 是主路，Python / Shell 是高级能力和逃生通道，Connector 负责外部世界。

---

## 14. Python Execution

保留 Python Code Executor，适合：

- 自定义空间分析
- 批处理
- 数据清洗
- 字段计算
- 组合多个 GIS Library
- 临时算法
- 用户要求编写 / 执行脚本

Python Runtime 第一版支持：

```text
GeoPandas
Shapely
Rasterio
PyProj
GDAL
NumPy
Pandas
```

执行后：

```text
Python Executor
     ↓
Workspace Diff
     ↓
发现新文件
     ↓
Dataset Inspector
     ↓
Dataset Registry
     ↓
Lineage
```

不能让 Code Execution 绕过 Dataset / Artifact / Trace。

---

## 15. Shell Execution

保留 Shell Executor。

GIS 生态大量能力是 CLI-first：

```text
gdalinfo
gdalwarp
gdal_translate
ogr2ogr
ogrinfo
gdaldem
```

Shell 必须具备：

- Workspace 限制
- timeout 与 Run 取消
- Permission
- Trace
- Output Discovery
- Dataset Registration
- Lineage

---

## 16. Sandbox 与 Permission

Shell / Python 不是无约束执行。

### Level 1：默认允许

```text
读取 Project Workspace
读取 Dataset
生成新结果
dataset.inspect
gdalinfo
ogrinfo
```

### Level 2：受控执行

```text
Python Script
gdalwarp
ogr2ogr
复杂 GIS CLI
```

当前第一版限制：仅允许 Workspace 内路径，具备 timeout、Run 取消和 No GUI 约束。
它不是对抗恶意代码的操作系统级沙箱；CPU / RAM 硬配额暂不伪装成已实现能力，留在后续部署层处理。

### Level 3：需要 Approval

```text
覆盖原 Dataset
删除 Dataset
UPDATE / DELETE PostGIS
访问外部私有服务
覆盖已有结果
未知二进制
```

### Level 4：禁止

```text
sudo
系统目录修改
关闭安全机制
系统设置修改
```

---

## 17. Failure Recovery

失败不能统一 retry 3 次。

需要：

```text
Retry
Repair
Replan
Ask User
Abort
```

### Retry

同样操作再次执行。适用于 timeout、temporary network failure、rate limit。

### Repair

修正输入或环境后重新执行。适用于 CRS_UNIT_MISMATCH、INVALID_GEOMETRY 等。

### Replan

当前方案不适用，重新规划。

### Ask User

缺少必要信息。

### Abort

不可恢复错误。

---

## 18. GIS Self-Repair 示例

### Buffer CRS Error

```text
vector.buffer
    ↓
CRS_UNIT_MISMATCH
    ↓
FailureAnalyzer
    ↓
REPAIR
    ↓
crs.reproject
    ↓
vector.buffer
    ↓
vector.validate
```

### Invalid Geometry

```text
vector.intersection
    ↓
INVALID_GEOMETRY
    ↓
vector.validate
    ↓
vector.repair
    ↓
retry intersection
```

### Missing Field

```text
field "population" missing
    ↓
dataset.inspect
    ↓
发现 POP_2025
    ↓
repair parameters
    ↓
retry
```

---

## 19. Tool Side Effects 与幂等性

Retry 之前必须考虑：

```text
idempotent
side_effect
```

示例：

```text
dataset.inspect
idempotent = true

vector.buffer
如果输出采用 execution_id 唯一命名
→ effectively idempotent

database.update
idempotent = false
```

对于非幂等 Tool：

```text
执行过程中状态未知
→ ToolStatus.UNKNOWN
→ 检查副作用
→ 决定恢复策略
```

---

## 20. Checkpoint 与恢复

关键节点写 Checkpoint：

```text
Plan Created
Tool Completed
SubAgent Completed
Dataset Created
Agent Decision
Working Memory Updated
```

恢复时：

```text
load checkpoint
    ↓
已完成的任务不重复
    ↓
继续失败位置
```

---

## 21. Run Budget

必须从第一版存在。

```python
class RunBudget:
    max_agent_turns: int
    max_tool_calls: int
    max_retry_per_action: int
    max_subagents: int
    max_parallel_agents: int
    max_tokens: int
    max_execution_seconds: int
```

第一版可考虑：

```text
Main Agent Turns       20
SubAgent Turns         10
Retry Per Action        2
Max SubAgents           5
Max Parallel Agents     3
```

Budget 用完：

```text
BUDGET_EXCEEDED
```

作为正式状态。

---

## 22. Trace 与 Observability

必须记录完整执行过程。

事件包括：

```text
RunCreated
IntentResolved
PlanCreated
SubTaskCreated
SubAgentSpawned
ToolStarted
ToolCompleted
ToolFailed
RepairSelected
RetryStarted
ReplanStarted
DatasetCreated
ArtifactCreated
VerificationStarted
VerificationFailed
SubAgentCompleted
RunCompleted
RunFailed
```

UI 可展示：

```text
run_100
│
├─ Intent                    120ms
├─ Planning                  680ms
│
├─ SubAgent road             2.3s
│  ├─ dataset.inspect
│  └─ analysis.distance
│
├─ SubAgent population       3.1s
│  └─ python.execute
│
├─ SubAgent terrain          4.2s
│  ├─ raster.inspect
│  └─ raster.slope
│
├─ Verification              300ms
└─ Synthesis                 1.1s
```

---

## 23. 横切能力

不作为第五层，贯穿四层。

```text
Permission
Observability
Budget
Evaluation
Events
```

```text
──────────────────────────────────────────────
 Entry     Decision     Execution      State
──────────────────────────────────────────────
   │           │            │           │
   ├──────── Permission ────────────────┤
   ├──────── Observability ─────────────┤
   ├──────── Budget ────────────────────┤
   ├──────── Events ────────────────────┤
   └──────── Evaluation ────────────────┘
```

---

## 24. 第一批 GIS Tool

第一版不追求数量。

### Dataset

```text
dataset.list
dataset.inspect
dataset.register
```

### CRS

```text
crs.inspect
crs.reproject
```

### Vector

```text
vector.validate
vector.repair
vector.buffer
vector.clip
vector.intersection
vector.spatial_join
vector.dissolve
```

### Raster

```text
raster.inspect
raster.clip
raster.reproject
```

### Analysis

```text
analysis.zonal_statistics
analysis.distance
```

### Visualization

```text
map.render
```

### General Execution

```text
python.execute
shell.execute
```

---

## 25. GIS Engine

Tool 与具体 GIS Library 分层。

```text
Agent
 ↓
Tool
 ↓
Domain Service
 ↓
Engine
```

例如：

```text
VectorBufferTool
     ↓
VectorService.buffer()
     ↓
GeoPandasEngine.buffer()
```

目录：

```text
gis/engines/
├── base.py
├── geopandas.py
├── rasterio.py
└── gdal.py
```

第一版：

```text
Vector → GeoPandas + Shapely + PyProj
Raster → Rasterio + GDAL
```

未来可以增加 PostGIS / DuckDB Spatial。

---

## 26. Knowledge Retrieval

保留 Knowledge Layer，但第一版实现简单。

用于：

- GIS 方法知识
- CRS / EPSG 知识
- 空间分析规则
- 项目说明文档
- 数据字典

Knowledge ≠ Memory。

```text
Knowledge = 领域事实与资料
Memory    = 用户 / 项目历史中形成的持续状态
```

---

## 27. 项目目录

```text
geo-agent/
│
├── backend/
│   └── app/
│
│       ├── application.py
│       ├── config.py
│
│       ├── entry/
│       │   ├── request.py
│       │   ├── conversation_service.py
│       │   ├── attachment_service.py
│       │   ├── dataset_resolver.py
│       │   ├── reference_resolver.py
│       │   └── normalizer.py
│       │
│       ├── decision/
│       │   ├── intent.py
│       │   ├── planner.py
│       │   ├── decomposer.py
│       │   ├── parallelism.py
│       │   ├── router.py
│       │   ├── tool_selector.py
│       │   ├── failure_analyzer.py
│       │   ├── verifier.py
│       │   └── response_planner.py
│       │
│       ├── runtime/
│       │   ├── agent_loop.py
│       │   ├── context_manager.py
│       │   ├── budget.py
│       │   └── lifecycle.py
│       │
│       ├── agent/
│       │   ├── base.py
│       │   ├── main_agent.py
│       │   ├── sub_agent.py
│       │   ├── manager.py
│       │   ├── scheduler.py
│       │   ├── policy.py
│       │   ├── result.py
│       │   └── context.py
│       │
│       ├── task/
│       │   ├── model.py
│       │   ├── service.py
│       │   ├── graph.py
│       │   └── repository.py
│       │
│       ├── execution/
│       │   ├── tools/
│       │   │   ├── model.py
│       │   │   ├── registry.py
│       │   │   ├── executor.py
│       │   │   └── middleware.py
│       │   │
│       │   ├── python/
│       │   │   ├── executor.py
│       │   │   └── result.py
│       │   │
│       │   ├── shell/
│       │   │   ├── executor.py
│       │   │   └── result.py
│       │   │
│       │   └── sandbox/
│       │       ├── manager.py
│       │       ├── policy.py
│       │       └── limits.py
│       │
│       ├── gis/
│       │   ├── dataset/
│       │   │   ├── model.py
│       │   │   ├── registry.py
│       │   │   ├── inspector.py
│       │   │   ├── metadata.py
│       │   │   └── lineage.py
│       │   │
│       │   ├── crs/
│       │   │   ├── service.py
│       │   │   └── validator.py
│       │   │
│       │   ├── vector/
│       │   │   ├── service.py
│       │   │   └── validator.py
│       │   │
│       │   ├── raster/
│       │   │   ├── service.py
│       │   │   └── validator.py
│       │   │
│       │   ├── analysis/
│       │   │   ├── proximity.py
│       │   │   ├── overlay.py
│       │   │   ├── terrain.py
│       │   │   └── statistics.py
│       │   │
│       │   ├── visualization/
│       │   │   └── renderer.py
│       │   │
│       │   └── engines/
│       │       ├── base.py
│       │       ├── geopandas.py
│       │       ├── rasterio.py
│       │       └── gdal.py
│       │
│       ├── tools/
│       │   └── gis/
│       │       ├── dataset.py
│       │       ├── crs.py
│       │       ├── vector.py
│       │       ├── raster.py
│       │       ├── spatial.py
│       │       └── visualization.py
│       │
│       ├── knowledge/
│       │   ├── retriever.py
│       │   ├── repository.py
│       │   └── gis_knowledge.py
│       │
│       ├── state/
│       │   ├── conversation/
│       │   ├── task/
│       │   ├── run/
│       │   ├── agent/
│       │   ├── checkpoint/
│       │   ├── working_memory/
│       │   ├── memory/
│       │   ├── artifact/
│       │   └── trace/
│       │
│       ├── models/
│       │   ├── adapter.py
│       │   ├── registry.py
│       │   └── providers/
│       │
│       ├── permission/
│       │   ├── policy.py
│       │   └── approval.py
│       │
│       ├── events/
│       │   ├── model.py
│       │   └── bus.py
│       │
│       ├── observability/
│       │   ├── trace.py
│       │   ├── logging.py
│       │   └── metrics.py
│       │
│       ├── evaluation/
│       │   ├── cases/
│       │   ├── runner.py
│       │   └── metrics.py
│       │
│       └── api/
│           ├── app.py
│           ├── websocket.py
│           └── routes/
│
├── frontend/
│   ├── chat/
│   ├── datasets/
│   ├── agents/
│   ├── runs/
│   ├── trace/
│   └── results/
│
├── workspace/
│   ├── input/
│   ├── intermediate/
│   ├── output/
│   └── temp/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── agent/
│   └── evaluation/
│
├── docs/
│   ├── architecture.md
│   ├── agent-loop.md
│   ├── tool-protocol.md
│   └── state-model.md
│
└── README.md
```

---

## 28. 技术栈

推荐：

```text
Backend
- Python 3.12+
- FastAPI
- Pydantic
- SQLAlchemy
- SQLite
- asyncio

LLM
- 自定义 ModelAdapter
- OpenAI-compatible first

GIS
- GeoPandas
- Shapely
- PyProj
- Rasterio
- GDAL

Execution
- asyncio subprocess

Frontend
- React
- TypeScript
- MapLibre GL 或 Leaflet

Testing
- pytest
- pytest-asyncio
```

原则：Agent Runtime 尽量自己实现。初期不要让 CrewAI / AutoGen / LangGraph 隐藏掉最有展示价值的 Agent Loop、Task Decomposition、Failure Recovery 和 Parallel Agent Scheduling。

---

## 29. 存储

第一版 SQLite 足够。

建议至少有：

```text
conversations
messages
tasks
subtasks
runs
agent_runs
tool_calls
checkpoints
datasets
dataset_lineage
artifacts
memories
traces
```

---

## 30. UI 第一版

建议只有：

```text
Chat
Datasets
Agents
Run Trace
Results
```

示意：

```text
┌───────────────┬────────────────────────┐
│ Dataset       │ Chat                   │
│               │                        │
│ roads.gpkg    │ 帮我分析...            │
│ dem.tif       │                        │
│ population    │                        │
├───────────────┼────────────────────────┤
│ Agents        │ Result                 │
│               │                        │
│ Main          │ map                    │
│ ├─ road ✓     │ statistics             │
│ ├─ pop  ✓     │ output.gpkg            │
│ └─ dem  →     │                        │
├───────────────┴────────────────────────┤
│ Run Trace                              │
└────────────────────────────────────────┘
```

---

## 31. 第一阶段验收场景

### Case 1：单 Agent + Failure Recovery

用户：

```text
检查这个道路数据，并生成 500 米缓冲区。
```

预期：

```text
User
 ↓
AgentRequest
 ↓
Main Agent
 ↓
dataset.inspect
 ↓
发现 EPSG:4326
 ↓
vector.buffer
 ↓
CRS_UNIT_MISMATCH
 ↓
FailureAnalyzer
 ↓
REPAIR
 ↓
crs.reproject
 ↓
vector.buffer
 ↓
vector.validate
 ↓
Dataset + Artifact
 ↓
Final Answer
```

验证：Agent Loop、Tool Calling、Dataset Context、GIS Failure Recovery、Trace、Artifact、Run State。

### Case 2：动态 Multi-Agent

用户：

```text
综合道路、人口和 DEM，从三个方面评价当前区域。
```

预期：

```text
Main Agent
    ↓
Task Decomposition
    ↓

road      population      terrain
 │            │              │
 ▼            ▼              ▼
Agent A      Agent B        Agent C
 │            │              │
 └────────────┼──────────────┘
              ↓
         AgentResult[]
              ↓
          Main Agent
              ↓
           Verify
              ↓
          Synthesis
```

验证：Task Decomposition、Parallelism Analysis、SubAgent、Context Isolation、Parallel Scheduling、Partial Failure、Main Agent Synthesis、Nested Run Trace。

---

## 32. Evaluation

第一阶段做 10~20 个固定 Case 即可。

指标：

```text
Task Success
Tool Selection Accuracy
GIS Safety
CRS Handling
Failure Recovery Success
SubTask Decomposition Quality
Parallelism Correctness
Result Verification
Tool Call Count
Token Cost
Execution Time
```

典型测试：

```text
4326 数据执行 500m Buffer
→ Agent 必须发现单位问题

不同 CRS Dataset Overlay
→ 必须处理 CRS

Invalid Geometry Intersection
→ 应进行 Repair

SubAgent B 失败但 optional
→ Main Agent 应继续并返回 Partial Success

Python 生成新 GPKG
→ 自动进入 Dataset Registry 和 Lineage
```

---

## 33. 开发顺序

### Phase 1：核心协议

先写：

```text
AgentRequest
Task
SubTask
AgentDecision
ToolCall
ToolResult
ToolError
AgentResult
Run
Dataset
Artifact
Checkpoint
```

### Phase 2：Model Adapter

实现：

```text
ModelAdapter
ModelRegistry
OpenAI-Compatible Adapter
```

### Phase 3：Tool Runtime

实现：

```text
Tool
ToolRegistry
ToolExecutor
Permission
ToolResult
```

### Phase 4：Python / Shell

实现：

```text
python.execute
shell.execute
Sandbox
Workspace
```

### Phase 5：Run / State / Trace

实现：

```text
RunManager
RunState
WorkingMemory
Trace
```

### Phase 6：Main Agent Loop

先只做：

```text
reason → tool → observe → reason → final
```

### Phase 7：Failure Recovery

加入：

```text
Retry
Repair
Replan
Ask User
Abort
```

### Phase 8：Dataset Context

实现：

```text
Dataset
DatasetRegistry
DatasetInspector
DatasetLineage
```

### Phase 9：GIS Tools

加入第一批 Vector / Raster / CRS Tool。

### Phase 10：Task Decomposition

实现：

```text
Task Planner
SubTask
Task Graph
Parallelism Analyzer
```

### Phase 11：SubAgent

实现：

```text
SubAgent
SubAgentContext
AgentResult
```

### Phase 12：Parallel Agent Manager

实现：

```text
AgentManager
Scheduler
asyncio.gather
Concurrency Limit
Partial Failure
```

### Phase 13：Checkpoint / Resume

支持：

```text
interrupt
checkpoint
resume
```

### Phase 14：Memory

先实现 Working Memory + Project Memory。

### Phase 15：Frontend

实现 Chat + Dataset + Agent Tree + Run Trace + Result。

### Phase 16：Evaluation

加入固定 GIS Agent cases。

---

## 34. 与 Vesta 的关系

GeoAgent 从零重写，但参考 Vesta 的核心思想。

### 保留 / 吸收

```text
Conversation
RunManager
AgentRuntime
AgentLoop
Context Management
Tool Registry
Tool Executor
Permission
Checkpoint
Artifact
Evidence
Trace
Memory
Skill 思想
Shell / Code Execution
```

### 删除

```text
Computer Agent
macOS Native Helper
Screen Capture
Accessibility
Mouse / Keyboard
Window Management
Clipboard
Browser GUI Automation
第三方 App GUI 操作
```

### 重构

```text
Generic Agent
→ GIS Assistant Agent

Generic Context
→ GIS-aware Context

Generic Task
→ Task + SubTask + Task Graph

Generic Tool
→ GIS Tool + Python + Shell

Generic Memory
→ Working / Project / Long-term Memory

Generic Filesystem
→ Workspace + Dataset + Artifact

Generic Multi-tool Agent
→ Main Agent + Temporary SubAgents
```

---

## 35. 当前冻结的 Architecture Decisions

| 问题 | 当前决定 |
|---|---|
| 项目类型 | GIS 辅助 Agent |
| 是否平台 | 否 |
| 是否 Workflow 驱动 | 否 |
| 是否支持复杂任务拆解 | 是 |
| 是否支持 Multi-Agent | 是 |
| Multi-Agent 类型 | Main Agent + 临时 SubAgent |
| SubAgent 是否递归创建 | 第一版禁止 |
| Agent 拓扑 | Hub-and-Spoke |
| SubAgent 是否直接通信 | 否 |
| 是否支持并行 | 是 |
| 并行实现 | asyncio |
| 是否有 Agent Loop | 是，系统核心 |
| Tool 失败是否直接结束 | 否 |
| Failure Recovery | Retry / Repair / Replan / Ask User / Abort |
| GIS Tool | 是 |
| Python Execute | 是 |
| Shell Execute | 是 |
| SQL | 后续 |
| Dataset Context | 是 |
| Dataset Lineage | 是 |
| Memory | 是，第一版轻量 |
| Working Memory | 是 |
| Project Memory | 是 |
| Long-term Memory | 接口保留 |
| Checkpoint | 是 |
| Resume | 是 |
| Run | 是 |
| Trace | 是 |
| Artifact | 是 |
| Knowledge Retrieval | 是 |
| Permission | 是 |
| Budget | 是 |
| Evaluation | 是 |
| Computer Agent | 否 |
| GUI Automation | 否 |
| Browser GUI Automation | 否 |
| 第三方 App 控制 | 否 |
| GIS Engine | GeoPandas / Rasterio / GDAL |
| Agent Framework | 优先自己实现 |
| Database | SQLite first |
| Backend | Python + FastAPI |
| Frontend | React + TypeScript |
| Map | MapLibre / Leaflet |

---

## 36. 最终整体图

```text
                            USER
                              │
                              ▼
                    ┌─────────────────┐
                    │   TASK ENTRY    │
                    │                 │
                    │ Conversation    │
                    │ Dataset Resolve │
                    │ Context Resolve │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    MAIN AGENT   │
                    │                 │
                    │   Agent Loop    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    DECISION     │
                    │                 │
                    │ Intent          │
                    │ Plan            │
                    │ Decompose       │
                    │ Route           │
                    │ Failure         │
                    │ Verify          │
                    └───────┬─────────┘
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
        GIS TOOLS       CODE/SHELL     SUB AGENTS
             │              │          /    |    \
             │              │         A     B     C
             │              │          \    |    /
             └──────────────┼───────────┼────┘
                            │
                            ▼
                      OBSERVATIONS
                            │
                            ▼
                    ┌─────────────────┐
                    │ MEMORY & STATE  │
                    │                 │
                    │ Run             │
                    │ Task            │
                    │ Working Memory  │
                    │ Checkpoint      │
                    │ Dataset         │
                    │ Artifact        │
                    │ Trace           │
                    └────────┬────────┘
                             │
                             └─────────→ Main Agent Loop


        Cross-cutting:
        Permission / Budget / Events / Observability / Evaluation
```

---

## 37. 开发原则

1. 先保证 Agent Harness 闭环，再增加 GIS Tool 数量。
2. 框架可以完整，实现可以简单。
3. LLM 负责决策，不负责替代确定性 GIS 计算。
4. 标准 GIS 操作优先 Tool，自定义任务使用 Python / Shell。
5. 任何执行结果都必须能够进入 Run / Trace。
6. 任何新 GIS 数据尽量进入 Dataset Registry。
7. 任何派生 Dataset 尽量记录 Lineage。
8. Tool Success 不等于 Task Success，必须有 Verify。
9. Retry 不是默认恢复策略，要区分 Retry / Repair / Replan。
10. SubAgent 获取最小 Context，不复制 Main Agent 全上下文。
11. SubAgent 第一版禁止递归创建 Agent。
12. 所有 Agent 都受 Budget 约束。
13. GUI 自动化不属于本项目核心能力。
14. Agent Runtime 优先自己实现，不让第三方框架隐藏核心逻辑。
15. 每新增一个能力，都需要考虑 State、Trace、Failure、Permission、Evaluation。

---

## 38. 第一目标

不要先追求几十种 GIS Tool、漂亮 UI、支持所有数据类型或所有 GIS 算法。

第一目标是跑通：

```text
用户请求
 ↓
AgentRequest
 ↓
Main Agent
 ↓
Decision
 ↓
GIS Tool
 ↓
Tool Failure
 ↓
Failure Analyzer
 ↓
Repair
 ↓
重新执行
 ↓
Verify
 ↓
Dataset / Artifact
 ↓
Trace
 ↓
Final Answer
```

然后再跑通：

```text
Main Agent
 ↓
Task Decomposition
 ↓
Parallel SubAgents
 ↓
AgentResult[]
 ↓
Main Agent Verify
 ↓
Synthesis
 ↓
Final Answer
```

当这两个闭环稳定后，GeoAgent 的核心技术价值就已经成立。

---

## 39. 面试时的项目表述

> 我参考 long-running agent harness 的设计思想，从零实现了一个面向 GIS 的智能辅助 Agent。系统不是简单的 LLM + Tool Calling，而是包含任务入口、决策、执行、记忆与状态四层架构。Main Agent 会根据任务复杂度动态拆解子任务，在适合的情况下并行创建临时 SubAgent。执行层支持 GIS 领域工具、Python 和 Shell；Agent Loop 能根据结构化 Tool Result 区分 Retry、Repair 和 Replan，并通过 Run、Checkpoint、Trace 和 Dataset Lineage 保存执行状态与结果。整个项目重点解决的是 GIS 场景下智能体如何可靠地规划、执行、失败恢复和并行协作。

---

## Status

当前 README 是 GeoAgent 重写工作的 **Architecture Baseline v0**。

在没有明确架构问题前，后续开发优先在此框架内演进，避免频繁改变一级模块边界。

## Implementation Status

当前仓库已经包含一套可离线运行的独立实现：

- `backend/app/core`：AgentRequest、Task/SubTask、Run、ToolResult、Dataset、Artifact、Checkpoint 等领域协议；
- `backend/app/decision`：GIS 意图识别、计划、主题拆解、并行批次、失败恢复和结果验证；
- `backend/app/gis`：Vector/Raster/CRS 服务，以及 GeoPandas、Rasterio 引擎适配；
- `backend/app/execution`：workspace 边界、Python/Shell 执行、Tool Registry、权限、超时和结构化错误；
- `backend/app/state`：SQLite 状态、Dataset Registry、Lineage、Artifact、Memory 和 Trace 持久化；
- `backend/app/api`：FastAPI + WebSocket 接口；`frontend`：React + TypeScript 工作台；
- `backend/app/demo.py` 与 `backend/tests`：可复现的 CRS 修复、GIS 结果验证和并行 SubAgent 验收场景。
- 已补齐 P0 执行闭环：配置模型时由 OpenAI-compatible Adapter 驱动结构化 Tool Calling；无模型时仍使用离线规则路径；
- `POST /api/v1/runs` 提交后台运行，`POST /api/v1/runs/{run_id}/cancel` 取消运行，`DELETE /api/v1/runs/{run_id}` 删除已结束运行；`GET/POST/DELETE /api/v1/conversations` 管理对话，`/ws` 按 `run → event/delta → result` 推送运行事件和模型流式片段；
- `/api/v1/runs/{run_id}/resume` 从保存的请求、意图和计划 Checkpoint 继续执行，不会把恢复操作伪装成一次普通的新问答。
- 已补齐 P1 运行治理：Shell 使用 `shell=False` 参数数组执行，Python / GIS CLI 子进程可随 Run 取消；离线 GIS 步骤按检查、计算、验证、发布保存 Checkpoint，恢复时跳过已完成步骤；
- 已补齐 P2 运行质量：模型上下文接入会话历史、Working/Project Memory、引用 Run、工具定义和预算状态；工具暴露具体参数 schema；执行超时或取消会通知后台 handler/子进程停止，子 Agent Run 保留完整生命周期；`analysis.distance` 与 `vector.buffer` 分开路由；结果验证、运行指标、自然语言数据集/历史 Run 引用已接入主链路；
- 当前执行闭环分为两条明确路径：配置模型时由 `MainAgent model runtime → structured tool call → ToolResult → next model decision` 主导意图理解和执行，模型可以直接回答、追问、检查数据或动态组合工具；未配置模型时才使用 `IntentResolver → Planner → AgentRouter → AgentLoop` 作为有限离线兜底。离线 PlanStep 携带真实 Tool 参数、依赖和状态，执行结果作为 observation 写入 Checkpoint，失败后按错误类型选择重试、修复、重新规划或询问用户；常见的“重投影后缓冲/坡度”“栅格裁剪”“地图发布”等组合请求会生成多步计划；
- SubAgent 使用显式 `operation` 和 `dataset_ids`，只接收完成当前主题所需的数据集和工具白名单；Main Agent 的 `AgentDecision` 会记录下一步 ToolCall 或实际 SubTask，Task、Run、Trace、Dataset Lineage 和 Artifact 形成可回看的执行证据链；
- 工作台不会再把数据集列表的前几个条目隐式当作每次对话输入；Entry 层优先处理用户上传附件，其次按精确名称、主题角色、序号和“上一轮结果”解析数据，并在存在派生数据时优先返回本次 Lineage 输出；
- 工作台支持在对话框直接上传本地空间文件并自动登记为 Dataset，也支持 Workspace 内 Dataset 路径登记；数据集列表默认只显示名称，属性按钮可查看详细信息；同时提供对话新建/切换/删除、模型流式回复、Agents 活动视图、历史 Run 取消/恢复/删除、历史结果回看，以及结果 Artifact 打开。

快速运行：

```powershell
$geoAgentEnv = 'D:\PythonEnvs\GeoAgent'
& 'D:\Python3.12\python.exe' -m venv $geoAgentEnv
Set-Location .\backend
& "$geoAgentEnv\Scripts\python.exe" -m pip install -r requirements-dev.txt
& "$geoAgentEnv\Scripts\python.exe" -m app --demo
& "$geoAgentEnv\Scripts\python.exe" -m app --serve
```

另开终端启动前端工作台：

```bash
cd frontend
npm ci
npm run dev
```

环境约定：后端依赖只安装到 `D:\PythonEnvs\GeoAgent`，前端依赖只安装到
`frontend/node_modules`；这两个目录均为本机生成目录，不提交到仓库。首次创建
Python 虚拟环境时需要使用系统中的 Python 解释器，但运行 GeoAgent 时统一调用
`D:\PythonEnvs\GeoAgent\Scripts\python.exe`，不依赖全局 Python 包。

配置模型后，GeoAgent 使用 OpenAI 兼容 Chat Completions 接口进行真正的模型驱动意图理解：
模型收到会话历史、数据集属性、历史运行引用、工具定义和工具结果，并在每个工具回合后
重新选择下一步。模型不在工作台运行时修改，而是在 `backend/.env` 中预先配置为一个或多个
模型档案；聊天框右下角会提供下拉选择：

```text
GEOAGENT_MODEL_PROFILES=[{"id":"qwen","label":"本地千问","base_url":"http://127.0.0.1:11434/v1","model":"qwen2.5:7b","default":true},{"id":"deepseek","label":"DeepSeek","base_url":"https://api.deepseek.com/v1","api_key":"填写你的密钥","model":"deepseek-chat"}]
```

前端通过 `GET /api/v1/models` 读取已脱敏的模型列表；模型密钥不会返回浏览器。没有配置模型时，
GeoAgent 仍可离线运行有限的常见 GIS 操作，但不会把关键词规则伪装成开放域
自然语言智能。可通过 `GET /api/v1/metrics` 查看运行期计数，通过 `POST /api/v1/memories`
写入明确的 Project Memory。参考项目只用于借鉴 Run、Trace、Checkpoint、Context 和 Tool
治理等通用思想，GeoAgent 不包含其 Computer、桌面或第三方 GUI 能力。

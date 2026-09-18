import { FormEvent, useEffect, useMemo, useState } from "react";
import { api, Artifact, Conversation, Dataset, Event, fetchRunView, MeasurementSystem, ModelStatus, ResponseStyle, Result, Run, User, UserProfile } from "./api";
import { childRunsOf, eventRuns, groupRunsByTask, interactionModeLabel, isActiveRun, isMainRun, lineageSource, RESUMABLE_RUN_STATUSES, runTitle, runsForConversation } from "./domain";

type View = "chat" | "datasets" | "agents" | "runs" | "results" | "settings";
type ChatMessage = { id: string; role: "user" | "assistant"; content: string; result?: Result; events?: Event[]; artifacts?: Artifact[]; durationMs?: number };

const STATUS_LABELS: Record<string, string> = {
  CREATED: "已创建",
  PLANNING: "规划中",
  RUNNING: "运行中",
  WAITING_TOOL: "等待工具",
  WAITING_SUBAGENT: "等待子智能体",
  WAITING_USER: "等待补充信息",
  WAITING_APPROVAL: "等待确认",
  RETRYING: "重试中",
  REPLANNING: "重新规划中",
  VALIDATING: "验证中",
  COMPLETED: "已完成",
  PARTIAL_COMPLETED: "部分完成",
  SUCCESS: "成功",
  PARTIAL: "部分完成",
  FAILED: "失败",
  BLOCKED: "已阻塞",
  CANCELLED: "已取消",
  INTERRUPTED: "已中断",
  BUDGET_EXCEEDED: "超出预算",
};

const KIND_LABELS: Record<string, string> = {
  VECTOR: "矢量",
  RASTER: "栅格",
  TABLE: "表格",
  POINT_CLOUD: "点云",
  TRAJECTORY: "轨迹",
  NETWORK: "网络",
  SERVICE: "服务",
};

const FORMAT_LABELS: Record<string, string> = {
  geojson: "矢量文件",
  gpkg: "空间数据库",
  shp: "矢量文件",
  tif: "栅格文件",
  tiff: "栅格文件",
  csv: "表格文件",
  tsv: "表格文件",
  parquet: "表格文件",
};

const EVENT_LABELS: Record<string, string> = {
  RunCreated: "运行已创建",
  IntentResolved: "已识别任务意图",
  PlanCreated: "已生成执行计划",
  DecisionMade: "已确定下一步动作",
  SubTaskCreated: "已创建子任务",
  SubAgentSpawned: "已启动子智能体",
  ToolStarted: "工具开始执行",
  ToolCompleted: "工具执行完成",
  ToolFailed: "工具执行失败",
  RetryStarted: "开始重试",
  RepairSelected: "已选择修复方案",
  ReplanStarted: "开始重新规划",
  DatasetCreated: "已创建数据集",
  ArtifactCreated: "已生成结果文件",
  VerificationStarted: "开始验证结果",
  VerificationFailed: "结果验证失败",
  SubAgentCompleted: "子智能体已完成",
  CheckpointSaved: "已保存检查点",
  ResumeStarted: "开始恢复运行",
  RunCompleted: "运行完成",
  RunFailed: "运行失败",
  RunCancelled: "运行已取消",
};

const TOOL_LABELS: Record<string, string> = {
  "dataset.list": "列出数据集",
  "dataset.inspect": "检查数据集",
  "dataset.register": "登记数据集",
  "crs.inspect": "检查坐标系",
  "crs.reproject": "重投影",
  "vector.validate": "验证矢量数据",
  "vector.repair": "修复矢量几何",
  "vector.buffer": "生成矢量缓冲区",
  "vector.clip": "裁剪矢量数据",
  "vector.intersection": "计算矢量相交",
  "vector.dissolve": "融合矢量数据",
  "vector.spatial_join": "执行空间连接",
  "raster.inspect": "检查栅格数据",
  "raster.clip": "裁剪栅格数据",
  "raster.reproject": "重投影栅格",
  "raster.slope": "计算坡度",
  "analysis.distance": "距离分析",
  "analysis.zonal_statistics": "分区统计",
  "map.render": "生成地图",
  "python.execute": "执行分析代码",
  "shell.execute": "执行空间命令",
};

const FIELD_LABELS: Record<string, string> = {
  model: "模型",
  content: "内容",
  tool: "工具",
  status: "状态",
  output: "输出",
  error: "错误",
  source: "来源",
  dataset: "数据集",
  datasets: "数据集",
  distance_m: "距离（米）",
  distance: "距离",
  threshold: "距离阈值",
  within_threshold: "阈值内数量",
  min_distance: "最小距离",
  max_distance: "最大距离",
  mean_distance: "平均距离",
  repair_applied: "是否已修复坐标系",
  verification: "验证结果",
  inspection: "检查结果",
  road_quality: "道路质量",
  terrain: "地形分析",
  population_fields: "人口字段",
  feature_count: "要素数量",
  agent_id: "智能体",
  summary: "摘要",
  findings: "分析发现",
  scope: "范围",
  goal: "目标",
  context_dataset_count: "上下文数据集数量",
  run_id: "运行编号",
  result: "结果",
  path: "路径",
  name: "名称",
  kind: "类型",
  format: "格式",
  crs: "坐标系",
  extent: "范围",
  schema: "结构",
  metadata: "元数据",
};

function statusLabel(value: string): string {
  return STATUS_LABELS[value] ?? "未知状态";
}

function kindLabel(value: string): string {
  return KIND_LABELS[value] ?? "其他数据";
}

function formatLabel(value: string): string {
  return FORMAT_LABELS[value.toLowerCase()] ?? "空间数据文件";
}

function eventLabel(value: string): string {
  return EVENT_LABELS[value] ?? value;
}

function agentLabel(value: string): string {
  return value === "main" ? "主智能体" : "子智能体";
}

function replaceText(value: string, search: string, replacement: string): string {
  return value.split(search).join(replacement);
}

function displayEventMessage(message: string): string {
  let displayed = message;
  for (const [name, label] of Object.entries(TOOL_LABELS)) displayed = replaceText(displayed, name, label);
  for (const [name, label] of [["Main Agent", "主智能体"], ["SubAgent", "子智能体"], ["Dataset", "数据集"], ["Artifact", "结果文件"], ["Checkpoint", "检查点"], ["Tool", "工具"], ["CRS", "坐标系"], ["road", "道路"], ["population", "人口"], ["terrain", "地形"], ["SUCCESS", "成功"], ["PARTIAL_SUCCESS", "部分完成"], ["FAILED", "失败"], ["CANCELLED", "已取消"]]) {
    displayed = replaceText(displayed, name, label);
  }
  return displayed;
}

function translateFinding(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(translateFinding);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>).map(([key, item]) => [FIELD_LABELS[key] ?? key, translateFinding(item)]));
  }
  if (typeof value === "string") return STATUS_LABELS[value] ?? TOOL_LABELS[value] ?? value;
  return value;
}

function findingText(value: unknown): string {
  const translated = translateFinding(value);
  return typeof translated === "string" ? translated : JSON.stringify(translated, null, 2) ?? String(translated);
}

function assistantText(value: string): string {
  return value
    .replace(/\*\*(.*?)\*\*/gs, "$1")
    .replace(/__(.*?)__/gs, "$1")
    .replace(/^\s*[*-]\s+/gm, "• ")
    .replace(/\*\*/g, "");
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function formatDuration(durationMs: number): string {
  const totalSeconds = Math.max(0, Math.floor(durationMs / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes === 0) return `用时 ${seconds}秒`;
  if (seconds === 0) return `用时 ${minutes}分钟`;
  return `用时 ${minutes}分钟 ${seconds}秒`;
}

export function App() {
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [view, setView] = useState<View>("chat");
  const [accountOpen, setAccountOpen] = useState(false);
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState("");
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDatasetIds, setSelectedDatasetIds] = useState<string[]>([]);
  const [uploadedFiles, setUploadedFiles] = useState<Dataset[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [events, setEvents] = useState<Event[]>([]);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [result, setResult] = useState<Result | null>(null);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [selectedModelProfile, setSelectedModelProfile] = useState("");
  const [runStartedAt, setRunStartedAt] = useState<number | null>(null);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [streamingReply, setStreamingReply] = useState("");

  const requestDatasetIds = selectedDatasetIds;
  const requestAttachmentIds = useMemo(() => uploadedFiles.map((item) => item.id), [uploadedFiles]);
  const conversationRuns = useMemo(() => runsForConversation(conversationId, runs), [conversationId, runs]);
  const agentCount = useMemo(() => {
    const active = conversationRuns.filter(isActiveRun).length;
    if (active > 0) return active;
    const latestMain = conversationRuns.find(isMainRun);
    return latestMain ? 1 + childRunsOf(latestMain.id, conversationRuns).length : 0;
  }, [conversationRuns]);

  useEffect(() => {
    if (!busy || runStartedAt === null) return;
    const update = () => setElapsedMs(performance.now() - runStartedAt);
    update();
    const timer = window.setInterval(update, 1000);
    return () => window.clearInterval(timer);
  }, [busy, runStartedAt]);

  const refresh = async () => {
    try {
      const [nextDatasets, nextRuns, nextModel] = await Promise.all([api.datasets(), api.runs(), api.modelStatus()]);
      setDatasets(nextDatasets);
      setRuns(nextRuns);
      setModelStatus(nextModel);
      setSelectedModelProfile((current) => {
        const profileIds = nextModel.profiles.map((profile) => profile.id);
        if (current && profileIds.includes(current)) return current;
        return nextModel.default_profile ?? profileIds[0] ?? "";
      });
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const loadConversation = async (id: string) => {
    try {
      const history = await api.messages(id);
      setSelectedDatasetIds([]);
      setUploadedFiles([]);
      setSelectedRunId(null);
      setActiveRunId(null);
      setStreamingReply("");
      const restored = await Promise.all(history.map(async (item) => {
        const base: ChatMessage = { id: item.id, role: item.role === "user" ? "user" : "assistant", content: item.content };
        if (base.role !== "assistant" || !item.run_id) return base;
        try {
          const details = await fetchRunView(item.run_id, runs);
          const started = details.run.started_at ? Date.parse(details.run.started_at) : NaN;
          const finished = details.run.finished_at ? Date.parse(details.run.finished_at) : NaN;
          const durationMs = Number.isFinite(started) && Number.isFinite(finished) ? Math.max(0, finished - started) : undefined;
          return { ...base, content: details.result?.summary ?? base.content, result: details.result ?? undefined, events: details.events, artifacts: details.artifacts, durationMs };
        } catch {
          return base;
        }
      }));
      setMessages(restored);
      const latest = [...restored].reverse().find((item) => item.result);
      setResult(latest?.result ?? null);
      setEvents(latest?.events ?? []);
      setArtifacts(latest?.artifacts ?? []);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const loadConversations = async () => {
    const next = await api.conversations();
    setConversations(next);
    return next;
  };

  const initializeConversations = async () => {
    try {
      let next = await loadConversations();
      if (next.length === 0) {
        next = [await api.createConversation()];
        setConversations(next);
      }
      const savedId = window.sessionStorage.getItem("geoagent.conversation_id");
      const selected = next.find((item) => item.id === savedId) ?? next[0];
      setConversationId(selected.id);
      window.sessionStorage.setItem("geoagent.conversation_id", selected.id);
      await loadConversation(selected.id);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  useEffect(() => {
    let disposed = false;
    void api.me().then((user) => {
      if (!disposed) setCurrentUser(user);
    }).catch(() => {
      if (!disposed) setCurrentUser(null);
    }).finally(() => {
      if (!disposed) setAuthLoading(false);
    });
    return () => { disposed = true; };
  }, []);

  useEffect(() => {
    if (!currentUser) return;
    void api.profile().then(setUserProfile).catch((err) => setError(errorMessage(err)));
    void refresh();
    void initializeConversations();
  }, [currentUser?.id]);

  const clearSessionState = () => {
    setUserProfile(null);
    setConversationId("");
    setConversations([]);
    setMessages([]);
    setDatasets([]);
    setSelectedDatasetIds([]);
    setUploadedFiles([]);
    setRuns([]);
    setEvents([]);
    setArtifacts([]);
    setResult(null);
    setActiveRunId(null);
    setSelectedRunId(null);
    setStreamingReply("");
    setMessage("");
    setModelStatus(null);
    setSelectedModelProfile("");
    window.sessionStorage.removeItem("geoagent.conversation_id");
  };

  const logout = async () => {
    try {
      await api.logout();
    } catch {
      // 即使服务端 Session 已失效，也必须清理当前浏览器状态。
    }
    clearSessionState();
    setCurrentUser(null);
    setAccountOpen(false);
    setView("chat");
  };

  const selectConversation = async (id: string) => {
    if (busy) return;
    setPendingDeleteId(null);
    setView("chat");
    if (id === conversationId) return;
    setConversationId(id);
    window.sessionStorage.setItem("geoagent.conversation_id", id);
    setError("");
    await loadConversation(id);
  };

  const createNewConversation = async () => {
    if (busy) return;
    try {
      const created = await api.createConversation();
      setConversations((current) => [created, ...current]);
      setConversationId(created.id);
      window.sessionStorage.setItem("geoagent.conversation_id", created.id);
      setMessages([]);
      setEvents([]);
      setArtifacts([]);
      setResult(null);
      setSelectedDatasetIds([]);
      setUploadedFiles([]);
      setSelectedRunId(null);
      setActiveRunId(null);
      setView("chat");
      setError("");
      setPendingDeleteId(null);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const deleteConversation = async (conversation: Conversation) => {
    if (busy) return;
    try {
      await api.deleteConversation(conversation.id);
      const remaining = conversations.filter((item) => item.id !== conversation.id);
      if (conversation.id !== conversationId) {
        setConversations(remaining);
        setPendingDeleteId(null);
        return;
      }
      if (remaining.length === 0) {
        const created = await api.createConversation();
        setConversations([created]);
        setConversationId(created.id);
        window.sessionStorage.setItem("geoagent.conversation_id", created.id);
        setMessages([]);
        setResult(null);
        setEvents([]);
        setArtifacts([]);
        setSelectedDatasetIds([]);
        setUploadedFiles([]);
        setSelectedRunId(null);
        setActiveRunId(null);
        setPendingDeleteId(null);
        return;
      }
      const next = remaining[0];
      setConversations(remaining);
      setConversationId(next.id);
      window.sessionStorage.setItem("geoagent.conversation_id", next.id);
      await loadConversation(next.id);
      setPendingDeleteId(null);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const loadRun = async (runId: string) => {
    setSelectedRunId(runId);
    setResult(null);
    setArtifacts([]);
    setError("");
    try {
      const details = await fetchRunView(runId, runs);
      setEvents(details.events);
      setResult(details.result);
      setArtifacts(details.artifacts);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const send = async () => {
    const prompt = message.trim();
    if (!prompt || busy || !conversationId) return;
    setBusy(true);
    const startedAt = performance.now();
    setRunStartedAt(startedAt);
    setElapsedMs(0);
    setError("");
    setEvents([]);
    setArtifacts([]);
    setStreamingReply("");
    setMessage("");
    setMessages((current) => [...current, { id: `local-user-${Date.now()}`, role: "user", content: prompt }]);
    try {
      const next = await api.streamAsk(
        prompt,
        requestDatasetIds,
        requestAttachmentIds,
        (run) => { setActiveRunId(run.id); setSelectedRunId(run.id); },
        (event) => { setEvents((current) => [...current, event]); },
        (content) => setStreamingReply((current) => current + content),
        conversationId,
        selectedModelProfile,
      );
      const durationMs = Math.round(performance.now() - startedAt);
      const details = await fetchRunView(next.trace_id);
      const finalResult = details.result ?? next;
      setResult(finalResult);
      setEvents(details.events);
      setArtifacts(details.artifacts);
      setMessages((current) => [...current, { id: `local-assistant-${finalResult.trace_id}`, role: "assistant", content: finalResult.summary, result: finalResult, events: details.events, durationMs }]);
      setSelectedDatasetIds([]);
      setUploadedFiles([]);
      await refresh();
      await loadConversations();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
      setActiveRunId(null);
      setRunStartedAt(null);
      setStreamingReply("");
    }
  };

  const cancelRun = async (runId: string) => {
    setError("");
    try {
      await api.cancelRun(runId);
      await refresh();
      if (selectedRunId === runId) {
        const details = await fetchRunView(runId);
        setEvents(details.events);
        setResult(details.result);
        setArtifacts(details.artifacts);
      }
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const deleteRun = async (runId: string) => {
    if (busy) return;
    setError("");
    try {
      await api.deleteRun(runId);
      setRuns((current) => current.filter((item) => item.id !== runId));
      if (selectedRunId === runId) {
        setSelectedRunId(null);
        setEvents([]);
        setArtifacts([]);
        setResult(null);
      }
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const deleteRunRecords = async (runIds: string[]) => {
    if (busy || runIds.length === 0) return;
    setError("");
    try {
      const response = await api.deleteRuns(runIds);
      const deleted = new Set(response.deleted);
      setRuns((current) => current.filter((item) => !deleted.has(item.id)));
      if (selectedRunId && deleted.has(selectedRunId)) {
        setSelectedRunId(null);
        setEvents([]);
        setArtifacts([]);
        setResult(null);
      }
      if (response.skipped_active.length > 0) setError(`${response.skipped_active.length} 条正在运行的记录未删除，请先取消运行。`);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const resumeRun = async (runId: string) => {
    setBusy(true);
    setError("");
    try {
      const resumed = await api.resumeRun(runId);
      setSelectedRunId(resumed.run_id);
      setView("results");
      await refresh();
      const details = await fetchRunView(resumed.run_id);
      setResult(details.result ?? resumed.result);
      setEvents(details.events);
      setArtifacts(details.artifacts);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const registerDataset = async (path: string, name: string) => {
    setError("");
    try {
      await api.registerDataset(path, name);
      await refresh();
    } catch (err) {
      setError(errorMessage(err));
      throw err;
    }
  };

  const uploadFiles = async (files: FileList | null) => {
    if (!files?.length) return;
    setUploading(true);
    setError("");
    try {
      for (const file of Array.from(files)) {
        const dataset = await api.uploadAttachment(file);
        setUploadedFiles((current) => [...current, dataset]);
      }
      await refresh();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setUploading(false);
    }
  };

  if (authLoading) return <div className="auth-shell"><div className="auth-card"><span className="eyebrow">GeoAgent</span><h1>正在检查登录状态</h1><p>请稍候…</p></div></div>;
  if (!currentUser) return <AuthPage onAuthenticated={setCurrentUser} />;

  return <div className="shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark">G</span><div><b>GeoAgent</b><small>空间智能</small></div></div>
      <section className="sidebar-block workspace-block"><div className="sidebar-block-title">工作区</div><Nav label="数据集" icon="◇" count={datasets.length} active={view === "datasets"} onClick={() => setView("datasets")} /><Nav label="智能体" icon="◎" count={agentCount} active={view === "agents"} onClick={() => setView("agents")} /><Nav label="运行与追踪" icon="⌁" count={conversationRuns.length} active={view === "runs"} onClick={() => setView("runs")} /><Nav label="结果" icon="▣" active={view === "results"} onClick={() => setView("results")} /></section>
      <section className="sidebar-block conversation-block"><div className="sidebar-block-head"><span>对话</span><button type="button" className="new-conversation" aria-label="新建对话" title="新建对话" disabled={busy} onClick={() => void createNewConversation()}>＋</button></div><div className="conversation-list">{conversations.length === 0 ? <span className="conversation-empty">正在加载对话…</span> : conversations.map((item) => <div className="conversation-row" key={item.id}><button type="button" className={`conversation-select ${item.id === conversationId ? "active" : ""}`} disabled={busy} onClick={() => void selectConversation(item.id)}><span className="conversation-dot" /><span className="conversation-title">{item.title || "新对话"}</span></button><button type="button" className="conversation-delete" aria-label={`删除对话 ${item.title}`} title="删除对话" disabled={busy} onClick={() => setPendingDeleteId((current) => current === item.id ? null : item.id)}>×</button>{pendingDeleteId === item.id && <div className="conversation-confirm" role="dialog" aria-label={`确认删除对话 ${item.title}`}><span>删除这个对话？</span><div><button type="button" className="conversation-confirm-delete" onClick={() => void deleteConversation(item)}>删除</button><button type="button" className="conversation-confirm-cancel" onClick={() => setPendingDeleteId(null)}>取消</button></div></div>}</div>)}</div></section>
      <div className="account-area">{accountOpen && <div className="account-menu"><button type="button" onClick={() => { setView("settings"); setAccountOpen(false); }}>设置</button><button type="button" onClick={() => void logout()}>退出登录</button><div className="account-menu-note">当前账号：{currentUser.username}</div></div>}<button type="button" className="account-button" aria-expanded={accountOpen} onClick={() => setAccountOpen((current) => !current)}><span className="account-avatar">{(currentUser.display_name || currentUser.username).slice(0, 1).toUpperCase()}</span><span className="account-copy"><b>{currentUser.display_name || currentUser.username}</b><small>@{currentUser.username}</small></span><span className={`account-chevron ${accountOpen ? "open" : ""}`}>⌃</span></button></div>
    </aside>
    <main className="main">
      {view !== "chat" && <header className="topbar"><div><h1>{view === "datasets" ? "数据集登记" : view === "agents" ? "智能体活动" : view === "runs" ? "运行追踪" : view === "settings" ? "设置" : "结果中心"}</h1></div><div className="topbar-actions"><button className="ghost" onClick={() => setView("chat")}>返回对话</button><button className="close-view" type="button" aria-label="关闭当前页面" title="关闭" onClick={() => setView("chat")}>×</button><button className="ghost" onClick={() => void refresh()}>↻ 刷新</button></div></header>}
      {error && <div className="error">{error}</div>}
      {view === "chat" && <Chat message={message} setMessage={setMessage} busy={busy} conversationReady={Boolean(conversationId)} streamingReply={streamingReply} elapsedMs={elapsedMs} send={send} cancel={() => activeRunId ? cancelRun(activeRunId) : Promise.resolve()} activeRunId={activeRunId} events={events} messages={messages} onShowResult={(next) => { setResult(next); setView("results"); }} datasets={datasets} selectedDatasetIds={selectedDatasetIds} onRemoveDataset={(id) => setSelectedDatasetIds((current) => current.filter((item) => item !== id))} uploadedFiles={uploadedFiles} uploading={uploading} onUpload={uploadFiles} onRemoveFile={(id) => setUploadedFiles((current) => current.filter((item) => item.id !== id))} modelStatus={modelStatus} selectedModelProfile={selectedModelProfile} onModelChange={setSelectedModelProfile} />}
      {view === "datasets" && <DatasetPanel datasets={datasets} selectedDatasetIds={selectedDatasetIds} onToggleRequestDataset={(id) => setSelectedDatasetIds((current) => current.includes(id) ? current.filter((item) => item !== id) : [...current, id])} onRegister={registerDataset} busy={busy} />}
      {view === "agents" && <AgentPanel runs={conversationRuns} />}
      {view === "runs" && <RunPanel runs={conversationRuns} selectedRunId={selectedRunId} events={events} onSelect={loadRun} onCancel={cancelRun} onResume={resumeRun} onDelete={deleteRun} onDeleteMany={deleteRunRecords} busy={busy} />}
      {view === "results" && <ResultPanel result={result} datasets={datasets} events={events} artifacts={artifacts} />}
      {view === "settings" && <SettingsPanel currentUser={currentUser} onSaved={setCurrentUser} profile={userProfile} onProfileSaved={setUserProfile} modelStatus={modelStatus} />}
    </main>
  </div>;
}

function Nav({ label, icon, count, active, onClick }: { label: string; icon: string; count?: number; active: boolean; onClick: () => void }) { return <button className={`nav ${active ? "active" : ""}`} onClick={onClick}><span>{icon}</span>{label}{count !== undefined && <em>{count}</em>}</button>; }

function Chat({ message, setMessage, busy, conversationReady, streamingReply, elapsedMs, send, cancel, activeRunId, events, messages, onShowResult, datasets, selectedDatasetIds, onRemoveDataset, uploadedFiles, uploading, onUpload, onRemoveFile, modelStatus, selectedModelProfile, onModelChange }: { message: string; setMessage: (value: string) => void; busy: boolean; conversationReady: boolean; streamingReply: string; elapsedMs: number; send: () => Promise<void>; cancel: () => Promise<void>; activeRunId: string | null; events: Event[]; messages: ChatMessage[]; onShowResult: (result: Result) => void; datasets: Dataset[]; selectedDatasetIds: string[]; onRemoveDataset: (id: string) => void; uploadedFiles: Dataset[]; uploading: boolean; onUpload: (files: FileList | null) => Promise<void>; onRemoveFile: (id: string) => void; modelStatus: ModelStatus | null; selectedModelProfile: string; onModelChange: (value: string) => void }) {
  return <section className="chat-layout">{(messages.length > 0 || busy) && <div className="chat-history" aria-live="polite">{messages.map((item) => <ChatBubble item={item} onShowResult={onShowResult} key={item.id} />)}{busy && <><RunProgress events={events} durationMs={elapsedMs} live />{streamingReply && <div className="chat-message assistant streaming-message"><div className="chat-bubble">{assistantText(streamingReply)}<span className="typing-cursor" aria-hidden="true" /></div></div>}</>}</div>}<div className="composer"><textarea className="composer-input" value={message} onChange={(e) => setMessage(e.target.value)} rows={2} aria-label="输入空间问题" /><div className="composer-foot"><div className="composer-left"><label className="file-button" title="添加文件" aria-label="添加文件"><span aria-hidden="true">＋</span><input type="file" multiple accept=".geojson,.json,.gpkg,.shp,.zip,.kml,.gml,.tif,.tiff,.img,.vrt,.asc,.csv,.tsv,.parquet,.jsonl" disabled={busy || uploading} onChange={(event) => { void onUpload(event.currentTarget.files); event.currentTarget.value = ""; }} /></label>{selectedDatasetIds.length > 0 && <div className="file-chips request-dataset-chips"><span className="resource-chip-label">数据：</span>{selectedDatasetIds.map((id) => { const dataset = datasets.find((item) => item.id === id); return <span className="file-chip" key={id}><span className="file-chip-name">◇ {dataset?.name ?? id}</span><button type="button" className="file-remove" title={`移除数据集 ${dataset?.name ?? id}`} aria-label={`移除数据集 ${dataset?.name ?? id}`} onClick={() => onRemoveDataset(id)}>×</button></span>; })}</div>}{uploadedFiles.length > 0 && <div className="file-chips request-attachment-chips"><span className="resource-chip-label">附件：</span>{uploadedFiles.map((file) => <span className="file-chip" key={file.id}><span className="file-chip-name">📎 {file.name}</span><button type="button" className="file-remove" title={`移除 ${file.name}`} aria-label={`移除 ${file.name}`} onClick={() => onRemoveFile(file.id)}>×</button></span>)}</div>}</div><div className="composer-right">{uploading && <span className="uploading">正在上传…</span>}{modelStatus && (modelStatus.profiles.length > 0 ? <div className="model-picker"><span>模型</span><select value={selectedModelProfile} onChange={(event) => onModelChange(event.target.value)} disabled={busy} aria-label="选择模型">{modelStatus.profiles.map((profile) => <option value={profile.id} key={profile.id}>{profile.label}</option>)}</select></div> : <span className="model-picker-offline">未配置模型</span>)}{busy ? <button className="cancel" onClick={() => void cancel()}>取消运行</button> : <button className="primary send-button" aria-label="发送" title="发送" disabled={!message.trim() || uploading || !conversationReady} onClick={() => void send()}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m3 10.5 18-7.5-8.5 18-2-8.5L3 10.5Z" /><path d="m10.5 12.5 10.5-9.5" /></svg></button>}</div></div></div></section>;
}

function ChatBubble({ item, onShowResult }: { item: ChatMessage; onShowResult: (result: Result) => void }) {
  const result = item.result;
  const hasDetails = Boolean(result && (result.findings.length > 0 || result.datasets.length > 0 || result.artifacts.length > 0 || result.evidence.length > 0 || result.warnings.length > 0 || result.error || result.status !== "SUCCESS"));
  return <div className={`chat-message ${item.role}`}>{item.durationMs !== undefined && <RunProgress events={item.events ?? []} durationMs={item.durationMs} status={result?.status} /> }<div className="chat-bubble">{item.role === "assistant" ? assistantText(item.content) : item.content}</div>{result && hasDetails && <button className="chat-result-link" onClick={() => onShowResult(result)}>查看详细结果</button>}</div>;
}

function RunProgress({ events, durationMs, status, live = false }: { events: Event[]; durationMs: number; status?: string; live?: boolean }) {
  const currentEvent = events[events.length - 1];
  const displayStatus = live ? "处理中" : statusLabel(status ?? "COMPLETED");
  return <div className="run-progress"><div className="run-progress-head"><span className="run-progress-time">{formatDuration(durationMs)}</span><span className={`run-progress-status ${live ? "running" : (status ?? "COMPLETED").toLowerCase()}`}>{displayStatus}</span></div><div className="run-progress-current">{currentEvent ? <><span className="run-progress-marker">✓</span><span>{eventLabel(currentEvent.event_type)}：{displayEventMessage(currentEvent.message)}</span></> : <><span className="run-progress-spinner" />等待智能体事件…</>}</div></div>;
}

function DatasetPanel({ datasets, selectedDatasetIds, onToggleRequestDataset, onRegister, busy }: { datasets: Dataset[]; selectedDatasetIds: string[]; onToggleRequestDataset: (id: string) => void; onRegister: (path: string, name: string) => Promise<void>; busy: boolean }) {
  const [path, setPath] = useState("");
  const [name, setName] = useState("");
  const [registering, setRegistering] = useState(false);
  const [propertyDataset, setPropertyDataset] = useState<Dataset | null>(null);
  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!path.trim()) return;
    setRegistering(true);
    try {
      await onRegister(path.trim(), name.trim());
      setPath("");
      setName("");
    } catch {
      // The parent renders the request error.
    } finally {
      setRegistering(false);
    }
  };
  return <section className="panel"><div className="panel-head"><div><span className="eyebrow">已登记数据源</span><h2>数据集登记</h2></div><span className="count-badge">{datasets.length} 个数据集</span></div><form className="dataset-register" onSubmit={(event) => void submit(event)}><input value={path} onChange={(event) => setPath(event.target.value)} placeholder="请输入工作区内的数据文件路径" aria-label="数据文件路径" /><input value={name} onChange={(event) => setName(event.target.value)} placeholder="显示名称（可选）" aria-label="显示名称" /><button className="primary" disabled={busy || registering || !path.trim()}>{registering ? "正在登记…" : "登记数据集"}</button></form>{datasets.length === 0 ? <Empty text="还没有数据集。请输入工作区内的文件路径进行登记。" /> : <div className="dataset-grid">{datasets.map((item) => { const selected = selectedDatasetIds.includes(item.id); return <div className={`dataset-card ${selected ? "request-selected" : ""}`} key={item.id}><b className="dataset-name">{item.name}</b><div className="dataset-card-actions"><button type="button" className={`small-action ${selected ? "selected-action" : ""}`} onClick={() => onToggleRequestDataset(item.id)}>{selected ? "移出本轮" : "用于下一条消息"}</button><button type="button" className="small-action" onClick={() => setPropertyDataset(item)}>属性</button></div></div>; })}</div>}{propertyDataset && <div className="dataset-modal" role="dialog" aria-modal="true" aria-label="数据集属性" onClick={() => setPropertyDataset(null)}><div className="dataset-modal-card" onClick={(event) => event.stopPropagation()}><div className="dataset-modal-head"><div><span className="eyebrow">数据集属性</span><h2>{propertyDataset.name}</h2></div><button type="button" className="modal-close" onClick={() => setPropertyDataset(null)}>关闭</button></div><div className="property-grid"><Property label="数据集编号" value={propertyDataset.id} /><Property label="数据类型" value={kindLabel(propertyDataset.kind)} /><Property label="文件格式" value={formatLabel(propertyDataset.format)} /><Property label="坐标系" value={propertyDataset.crs?.authority ?? "未提供"} /><Property label="坐标系名称" value={propertyDataset.crs?.name ?? "未提供"} /><Property label="要素数量" value={propertyDataset.schema?.feature_count !== undefined ? String(propertyDataset.schema.feature_count) : "不适用"} /><Property label="几何类型" value={propertyDataset.schema?.geometry_type ?? "未提供"} /><Property label="栅格尺寸" value={propertyDataset.schema?.width !== undefined ? `${propertyDataset.schema.width} × ${propertyDataset.schema.height ?? "?"}，${propertyDataset.schema.bands ?? "?"} 个波段` : "不适用"} /><Property label="空间范围" value={propertyDataset.extent ? `${propertyDataset.extent.min_x}, ${propertyDataset.extent.min_y} 至 ${propertyDataset.extent.max_x}, ${propertyDataset.extent.max_y}` : "未提供"} /><Property label="文件路径" value={propertyDataset.path} /><Property label="创建时间" value={propertyDataset.created_at ?? "未提供"} /><Property label="创建运行" value={propertyDataset.created_by_run_id ?? "手动登记或上传"} /></div><h3>字段</h3><pre>{propertyDataset.schema?.fields && Object.keys(propertyDataset.schema.fields).length > 0 ? JSON.stringify(propertyDataset.schema.fields, null, 2) : "未提供字段信息"}</pre><h3>附加信息</h3><pre>{JSON.stringify(propertyDataset.metadata ?? {}, null, 2)}</pre><h3>完整属性</h3><pre>{JSON.stringify(propertyDataset, null, 2)}</pre></div></div>}</section>;
}

function Property({ label, value }: { label: string; value: string }) { return <div className="property-item"><span>{label}</span><b>{value}</b></div>; }

function AgentPanel({ runs }: { runs: Run[] }) {
  const mainRuns = runs.filter(isMainRun);
  return <section className="panel"><div className="panel-head"><div><span className="eyebrow">执行树</span><h2>智能体活动</h2></div><span className="count-badge">{runs.length} 个执行节点</span></div>{mainRuns.length === 0 ? <Empty text="运行任务后，这里会显示主运行和子智能体执行树。" /> : <div className="agent-grid">{mainRuns.map((main) => { const children = childRunsOf(main.id, runs); return <div className="agent-card execution-tree-card" key={main.id}><div className="agent-icon">主</div><div className="execution-tree-copy"><b>{runTitle(main)}</b><small>主运行 · {statusLabel(main.status)} · {main.id}</small><span>{children.length ? `包含 ${children.length} 个子智能体执行` : "尚未委派子智能体"}</span>{children.length > 0 && <div className="execution-children">{children.map((child) => <div className="execution-child" key={child.id}><i>↳</i><div><b>{runTitle(child)}</b><small>子智能体 · {statusLabel(child.status)}</small></div></div>)}</div>}</div></div>; })}</div>}</section>;
}

function RunPanel({ runs, selectedRunId, events, onSelect, onCancel, onResume, onDelete, onDeleteMany, busy }: { runs: Run[]; selectedRunId: string | null; events: Event[]; onSelect: (id: string) => Promise<void>; onCancel: (id: string) => Promise<void>; onResume: (id: string) => Promise<void>; onDelete: (id: string) => Promise<void>; onDeleteMany: (ids: string[]) => Promise<void>; busy: boolean }) {
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [bulkDeletePending, setBulkDeletePending] = useState(false);
  const deletableIds = runs.filter((run) => !isActiveRun(run)).map((run) => run.id);
  const allSelected = deletableIds.length > 0 && deletableIds.every((id) => selectedIds.includes(id));
  const groups = [...groupRunsByTask(runs).entries()];

  useEffect(() => {
    const currentIds = new Set(runs.map((run) => run.id));
    setSelectedIds((current) => current.filter((id) => currentIds.has(id)));
  }, [runs]);

  const toggleRun = (runId: string) => {
    setSelectedIds((current) => current.includes(runId) ? current.filter((id) => id !== runId) : [...current, runId]);
  };

  const toggleAll = () => {
    setSelectedIds(allSelected ? [] : deletableIds);
  };

  const confirmBulkDelete = async () => {
    const ids = [...selectedIds];
    setBulkDeletePending(false);
    setSelectedIds([]);
    await onDeleteMany(ids);
  };

  const renderRun = (run: Run, nested = false) => {
    const active = isActiveRun(run);
    const source = lineageSource(run);
    return <div className={`run-row ${nested ? "nested-run" : ""} ${selectedRunId === run.id ? "selected" : ""}`} key={run.id}>
      <label className="run-check" title={active ? "正在运行的记录不能删除" : "选择运行记录"}><input type="checkbox" checked={selectedIds.includes(run.id)} disabled={busy || active} onChange={() => toggleRun(run.id)} aria-label={`选择运行记录 ${run.id}`} /></label>
      <button className="run-select" onClick={() => void onSelect(run.id)}><span className={`run-state ${run.status.toLowerCase()}`} /><div><b>{runTitle(run)}</b><small>{run.id} · {run.metadata.interaction_mode ? interactionModeLabel(run.metadata.interaction_mode) : agentLabel(run.agent_id)} · {run.tool_call_count} 次工具调用</small>{source && <small className="lineage-note">来源运行：{source}</small>}</div><em>{statusLabel(run.status)}</em></button>
      <div className="run-actions">{active && <button className="small-action danger" onClick={() => void onCancel(run.id)}>取消</button>}{RESUMABLE_RUN_STATUSES.has(run.status) && <button className="small-action" disabled={busy} onClick={() => void onResume(run.id)}>恢复</button>}{!active && <button className="small-action danger" disabled={busy} onClick={() => setPendingDeleteId((current) => current === run.id ? null : run.id)}>删除</button>}</div>
      {pendingDeleteId === run.id && <div className="run-delete-confirm" role="dialog" aria-label={`确认删除运行 ${run.id}`}><span>删除这条运行记录？</span><div><button type="button" className="conversation-confirm-delete" onClick={() => { setPendingDeleteId(null); void onDelete(run.id); }}>删除</button><button type="button" className="conversation-confirm-cancel" onClick={() => setPendingDeleteId(null)}>取消</button></div></div>}
    </div>;
  };
  return <section className="panel two-col"><div><div className="panel-head run-panel-head"><div><span className="eyebrow">当前对话运行</span><h2>运行记录</h2></div><div className="run-bulk-actions"><label className="run-select-all"><input type="checkbox" checked={allSelected} disabled={busy || deletableIds.length === 0} onChange={toggleAll} />全选可删除记录</label>{selectedIds.length > 0 && <><span className="run-selected-count">已选 {selectedIds.length} 条</span>{bulkDeletePending ? <div className="run-bulk-confirm"><span>删除已选记录？</span><button type="button" className="conversation-confirm-delete" onClick={() => void confirmBulkDelete()}>确认删除</button><button type="button" className="conversation-confirm-cancel" onClick={() => setBulkDeletePending(false)}>取消</button></div> : <button type="button" className="small-action danger" disabled={busy} onClick={() => setBulkDeletePending(true)}>删除已选</button>}</>}</div></div>{runs.length === 0 ? <Empty text="当前对话还没有运行记录。" /> : <div className="run-list">{groups.map(([taskId, taskRuns]) => { const commandGroup = taskId === "__command__"; const mains = taskRuns.filter(isMainRun); const linkedIds = new Set(mains.flatMap((main) => [main.id, ...childRunsOf(main.id, taskRuns).map((child) => child.id)])); const orphanRuns = taskRuns.filter((run) => !linkedIds.has(run.id)); return <section className="run-task-group" key={taskId}><div className="run-task-heading"><b>{commandGroup ? "查询 / 命令运行" : `任务 ${taskId}`}</b>{!commandGroup && <small>{runTitle(mains[0] ?? taskRuns[0])}</small>}</div>{mains.map((main) => <div key={main.id}>{renderRun(main)}{childRunsOf(main.id, taskRuns).map((child) => renderRun(child, true))}</div>)}{orphanRuns.map((run) => renderRun(run, !isMainRun(run)))}</section>; })}</div>}</div><div className="trace-box"><div className="eyebrow">运行追踪 · {selectedRunId ?? "未选择"} · {events.length} 个事件</div>{events.length === 0 ? <Empty text="选择一个运行记录查看事件。" /> : events.map((event) => { const eventRun = eventRuns(event, runs); const eventAgent = eventRun ? (isMainRun(eventRun) ? "主智能体" : `子智能体 · ${runTitle(eventRun)}`) : (event.agent_id === "main" ? "主智能体" : "子智能体"); return <div className="event" key={event.id}><span>{String(event.sequence).padStart(2, "0")}</span><div><b>{eventAgent} · {eventLabel(event.event_type)}</b><small>{displayEventMessage(event.message)}</small></div></div>; })}</div></section>;
}

function ResultPanel({ result, datasets, events, artifacts }: { result: Result | null; datasets: Dataset[]; events: Event[]; artifacts: Artifact[] }) {
  return <section className="panel result-panel">{!result ? <Empty text="完成一次分析后，结果、证据和运行追踪会显示在这里。" /> : <><div className="result-head"><div><span className={`pill ${result.status.toLowerCase()}`}>{statusLabel(result.status)}</span><h2>{result.summary}</h2></div><code>{result.trace_id}</code></div>{result.error && <div className="result-error"><b>错误</b><span>{result.error}</span></div>}<div className="result-columns"><div><h3>分析发现</h3>{result.findings.length === 0 ? <Empty text="没有结构化发现。" /> : result.findings.map((finding, index) => <pre key={index}>{findingText(finding)}</pre>)}<h3>关联数据集</h3>{result.datasets.length === 0 ? <p className="muted-text">本次运行没有关联数据集。</p> : <div className="dataset-result-list">{result.datasets.map((datasetId) => { const dataset = datasets.find((item) => item.id === datasetId); return <div className="dataset-result-item" key={datasetId}><b>{dataset?.name ?? datasetId}</b><small>{dataset?.kind ? kindLabel(dataset.kind) : "数据集"} · {datasetId}</small></div>; })}</div>}<h3>结果文件</h3>{result.artifacts.length === 0 ? <p className="muted-text">本次运行没有产物。</p> : <div className="artifact-list">{result.artifacts.map((artifactId) => { const artifact = artifacts.find((item) => item.id === artifactId); return <a className="artifact-link" href={api.artifactUrl(artifactId)} target="_blank" rel="noreferrer" key={artifactId}>{artifact?.name ?? artifactId} <span>↗</span></a>; })}</div>}{result.evidence.length > 0 && <><h3>证据</h3>{result.evidence.map((evidence, index) => <pre key={index}>{findingText(evidence)}</pre>)}</>}</div><div><h3>执行情况</h3><div className="metric"><b>{events.length}</b><span>追踪事件</span></div><div className="metric"><b>{result.datasets.length}</b><span>涉及数据集</span></div><div className="metric"><b>{result.artifacts.length}</b><span>结果文件</span></div>{result.warnings.length > 0 && <><h3>警告</h3>{result.warnings.map((warning) => <p className="warning" key={warning}>{warning}</p>)}</>}</div></div></> }</section>;
}

function SettingsPanel({ currentUser, onSaved, profile, onProfileSaved, modelStatus }: { currentUser: User; onSaved: (user: User) => void; profile: UserProfile | null; onProfileSaved: (profile: UserProfile) => void; modelStatus: ModelStatus | null }) {
  const [displayName, setDisplayName] = useState(currentUser.display_name);
  const [email, setEmail] = useState(currentUser.email ?? "");
  const [language, setLanguage] = useState(profile?.language ?? "zh-CN");
  const [responseStyle, setResponseStyle] = useState<ResponseStyle>(profile?.response_style ?? "balanced");
  const [measurementSystem, setMeasurementSystem] = useState<MeasurementSystem>(profile?.measurement_system ?? "metric");
  const [preferredOutputFormat, setPreferredOutputFormat] = useState(profile?.preferred_output_format ?? "");
  const [saving, setSaving] = useState(false);
  const [savingProfile, setSavingProfile] = useState(false);
  const [saved, setSaved] = useState(false);
  const [profileSaved, setProfileSaved] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    setLanguage(profile?.language ?? "zh-CN");
    setResponseStyle(profile?.response_style ?? "balanced");
    setMeasurementSystem(profile?.measurement_system ?? "metric");
    setPreferredOutputFormat(profile?.preferred_output_format ?? "");
  }, [profile]);
  const save = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSaving(true);
    setSaved(false);
    setError("");
    try {
      const user = await api.updateMe(displayName, email);
      onSaved(user);
      setSaved(true);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setSaving(false);
    }
  };
  const saveProfile = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSavingProfile(true);
    setProfileSaved(false);
    setError("");
    try {
      const updated = await api.updateProfile({ language, response_style: responseStyle, measurement_system: measurementSystem, preferred_output_format: preferredOutputFormat || null });
      onProfileSaved(updated);
      setProfileSaved(true);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setSavingProfile(false);
    }
  };
  return <section className="panel settings-panel"><div className="panel-head"><div><span className="eyebrow">账号与运行配置</span><h2>设置</h2></div></div><form className="account-settings-form" onSubmit={(event) => void save(event)}><h3>账号信息</h3><label>显示名称<input value={displayName} onChange={(event) => setDisplayName(event.target.value)} /></label><label>用户名<input value={currentUser.username} readOnly /></label><label>邮箱<input value={email} onChange={(event) => setEmail(event.target.value)} type="email" /></label><button className="primary" disabled={saving || !displayName.trim()}>{saving ? "正在保存…" : "保存账号信息"}</button>{saved && <span className="settings-success">已保存</span>}</form><form className="account-settings-form" onSubmit={(event) => void saveProfile(event)}><h3>用户偏好</h3><label>语言<select value={language} onChange={(event) => setLanguage(event.target.value)}><option value="zh-CN">中文</option><option value="en-US">English</option></select></label><label>回答风格<select value={responseStyle} onChange={(event) => setResponseStyle(event.target.value as ResponseStyle)}><option value="concise">简洁</option><option value="balanced">平衡</option><option value="detailed">详细</option></select></label><label>单位制<select value={measurementSystem} onChange={(event) => setMeasurementSystem(event.target.value as MeasurementSystem)}><option value="metric">公制</option><option value="imperial">英制</option></select></label><label>默认输出格式<select value={preferredOutputFormat} onChange={(event) => setPreferredOutputFormat(event.target.value)}><option value="">自动</option><option value="GeoPackage">GeoPackage</option><option value="GeoJSON">GeoJSON</option><option value="GeoTIFF">GeoTIFF</option><option value="CSV">CSV</option></select></label><button className="primary" disabled={savingProfile}>{savingProfile ? "正在保存…" : "保存用户偏好"}</button>{profileSaved && <span className="settings-success">偏好已保存</span>}{error && <span className="settings-inline-error">{error}</span>}<p className="settings-note">这些偏好只作为默认交互方式；当前请求的明确要求优先，格式偏好仅在任务能力允许时使用。</p></form><div className="settings-grid"><div className="setting-item"><span>登录状态</span><b>已登录</b></div><div className="setting-item"><span>默认模型</span><b>{modelStatus?.profiles.find((profile) => profile.id === modelStatus.default_profile)?.label ?? "未配置"}</b></div><div className="setting-item"><span>可用模型</span><b>{modelStatus?.profiles.length ?? 0} 个</b></div></div><p className="settings-note">模型接口从后端环境配置中读取，发送消息时可在对话框右下角切换。</p></section>;
}

function AuthPage({ onAuthenticated }: { onAuthenticated: (user: User) => void }) {
  const [registering, setRegistering] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [email, setEmail] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const user = registering
        ? await api.register(username, password, displayName || username, email)
        : await api.login(username, password);
      onAuthenticated(user);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  };
  return <div className="auth-shell"><div className="auth-card"><div className="brand auth-brand"><span className="brand-mark">G</span><div><b>GeoAgent</b><small>空间智能</small></div></div><span className="eyebrow">{registering ? "创建账号" : "欢迎回来"}</span><h1>{registering ? "创建你的 GeoAgent 账号" : "登录 GeoAgent"}</h1><p>{registering ? "账号创建后，你的数据、对话和运行记录将独立保存。" : "登录后继续访问你的对话和空间数据。"}</p><form className="auth-form" onSubmit={(event) => void submit(event)}><label>用户名或邮箱<input value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="username" required /></label><label>密码<input value={password} onChange={(event) => setPassword(event.target.value)} type="password" autoComplete={registering ? "new-password" : "current-password"} required /></label>{registering && <><label>显示名称<input value={displayName} onChange={(event) => setDisplayName(event.target.value)} placeholder="可选，默认使用用户名" /></label><label>邮箱<input value={email} onChange={(event) => setEmail(event.target.value)} type="email" placeholder="可选" /></label></>}<button className="primary auth-submit" disabled={busy}>{busy ? "处理中…" : registering ? "注册并登录" : "登录"}</button>{error && <div className="error auth-error">{error}</div>}</form><button type="button" className="auth-switch" onClick={() => { setRegistering((value) => !value); setError(""); }}>{registering ? "已有账号？返回登录" : "还没有账号？注册"}</button></div></div>;
}

function Empty({ text }: { text: string }) { return <div className="empty">{text}</div>; }

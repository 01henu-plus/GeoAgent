export type RunStatus = "CREATED" | "PLANNING" | "RUNNING" | "WAITING_TOOL" | "WAITING_SUBAGENT" | "WAITING_USER" | "WAITING_APPROVAL" | "RETRYING" | "REPLANNING" | "VALIDATING" | "COMPLETED" | "PARTIAL_COMPLETED" | "FAILED" | "INTERRUPTED" | "CANCELLED" | "BUDGET_EXCEEDED";
export type AgentResultStatus = "SUCCESS" | "PARTIAL" | "FAILED" | "BLOCKED" | "CANCELLED";
export type InteractionMode = "new_task" | "continue_task" | "modify_task" | "retry_task" | "query" | "chat" | "cancel_task" | string;

export type Dataset = { id: string; name: string; kind: string; path: string; format: string; crs?: { authority?: string; name?: string; is_geographic?: boolean; linear_unit?: string } | null; extent?: { min_x: number; min_y: number; max_x: number; max_y: number } | null; schema?: { feature_count?: number; fields?: Record<string, string>; geometry_type?: string; invalid_geometry_count?: number; width?: number; height?: number; bands?: number; resolution?: [number, number]; nodata?: number | null } | null; metadata?: Record<string, unknown>; source_dataset_ids?: string[]; created_by_run_id?: string | null; created_at?: string | null };
export type Result = { agent_id: string; task_id?: string | null; status: AgentResultStatus; summary: string; findings: unknown[]; datasets: string[]; artifacts: string[]; evidence: unknown[]; warnings: string[]; error?: string | null; trace_id: string };
export type RunMetadata = { goal?: string; request_id?: string; interaction_mode?: InteractionMode; continued_from?: string; retry_of?: string; resumed_from?: string; subtask_id?: string; subtask_goal?: string; command_run?: boolean; blocked?: boolean; result?: Result; [key: string]: unknown };
export type Run = { id: string; parent_run_id?: string | null; conversation_id?: string | null; task_id?: string | null; agent_id: string; status: RunStatus; started_at?: string | null; finished_at?: string | null; error?: string | null; turn_count: number; tool_call_count: number; metadata: RunMetadata };
export type Event = { id: string; run_id: string; event_type: string; message: string; sequence: number; timestamp: string; payload: Record<string, unknown>; agent_id?: string | null };
export type Artifact = { id: string; name: string; kind: string; path?: string | null; media_type?: string | null; dataset_id?: string | null; run_id?: string | null; description: string; metadata: Record<string, unknown>; created_at?: string | null };
export type ResumeResponse = { resumed_from: string; run_id: string; checkpoint: string; result: Result };
export type Conversation = { id: string; title: string; created_at: string; updated_at: string };
export type ConversationMessage = { id: string; conversation_id: string; role: string; content: string; run_id?: string | null; created_at?: string };
export type ModelProfile = { id: string; label: string; provider: string; base_url?: string | null; model: string; timeout_seconds: number; temperature: number; has_api_key: boolean; default: boolean };
export type ModelStatus = { configured: boolean; source: string; default_profile?: string | null; profiles: ModelProfile[] };
export type RunDetails = { run: Run; result: Result | null; events: Event[]; artifacts: Artifact[] };

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, { headers: { "Content-Type": "application/json" }, ...init });
  const body = await response.text();
  if (!response.ok) {
    let message = body;
    try {
      const payload = JSON.parse(body) as { detail?: string };
      message = payload.detail ?? body;
    } catch { /* 非 JSON 错误直接使用响应文本。 */ }
    throw new Error(message || `请求失败（${response.status}）`);
  }
  return (body ? JSON.parse(body) : undefined) as T;
}

function websocketUrl(): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws`;
}

async function uploadAttachment(file: File): Promise<Dataset> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch("/api/v1/attachments", { method: "POST", body: form });
  const body = await response.text();
  if (!response.ok) {
    let message = body;
    try {
      const payload = JSON.parse(body) as { detail?: string };
      message = payload.detail ?? body;
    } catch { /* 非 JSON 错误直接使用响应文本。 */ }
    throw new Error(message || `文件上传失败（${response.status}）`);
  }
  const payload = JSON.parse(body) as { dataset: Dataset };
  return payload.dataset;
}

function streamAsk(message: string, datasetIds: string[], attachmentIds: string[], onRun: (run: Run) => void, onEvent: (event: Event) => void, onDelta: (content: string) => void, conversationId?: string, modelProfile?: string): Promise<Result> {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(websocketUrl());
    let settled = false;
    const fail = (error: Error) => {
      if (settled) return;
      settled = true;
      reject(error);
    };
    socket.onopen = () => {
      try {
        socket.send(JSON.stringify({ type: "ask", message, conversation_id: conversationId, model_profile: modelProfile || undefined, dataset_ids: datasetIds, attachment_ids: attachmentIds }));
      } catch (error) {
        fail(error instanceof Error ? error : new Error("无法发送 GeoAgent 请求"));
      }
    };
    socket.onmessage = (raw) => {
      try {
        const payload = JSON.parse(raw.data as string) as { type: string; data?: Run | Event | Result; content?: string; message?: string };
        if (payload.type === "run") onRun(payload.data as Run);
        else if (payload.type === "event") onEvent(payload.data as Event);
        else if (payload.type === "delta") onDelta(payload.content ?? "");
        else if (payload.type === "result") {
          settled = true;
          resolve(payload.data as Result);
          socket.close();
      } else if (payload.type === "error") fail(new Error(payload.message ?? "实时请求失败"));
      } catch (error) {
        fail(error instanceof Error ? error : new Error("GeoAgent 返回了无效的实时消息"));
      }
    };
    socket.onerror = () => fail(new Error("无法连接 GeoAgent 实时通道"));
    socket.onclose = () => fail(new Error("GeoAgent 实时通道已断开"));
  });
}

export const api = {
  datasets: () => request<Dataset[]>("/api/v1/datasets"),
  uploadAttachment,
  conversations: (limit = 50) => request<Conversation[]>(`/api/v1/conversations?limit=${limit}`),
  createConversation: (title = "新对话") => request<Conversation>("/api/v1/conversations", { method: "POST", body: JSON.stringify({ title }) }),
  deleteConversation: (conversationId: string) => request<{ deleted: boolean }>(`/api/v1/conversations/${encodeURIComponent(conversationId)}`, { method: "DELETE" }),
  messages: (conversationId: string) => request<ConversationMessage[]>(`/api/v1/conversations/${encodeURIComponent(conversationId)}/messages`),
  registerDataset: (path: string, name?: string) => request<Dataset>("/api/v1/datasets", { method: "POST", body: JSON.stringify({ path, name: name || undefined }) }),
  runs: () => request<Run[]>("/api/v1/runs"),
  run: (runId: string) => request<Run>(`/api/v1/runs/${runId}`),
  deleteRun: (runId: string) => request<{ deleted: boolean }>(`/api/v1/runs/${encodeURIComponent(runId)}`, { method: "DELETE" }),
  deleteRuns: (runIds: string[]) => request<{ deleted: string[]; skipped_active: string[] }>("/api/v1/runs", { method: "DELETE", body: JSON.stringify({ run_ids: runIds }) }),
  events: (runId: string) => request<Event[]>(`/api/v1/runs/${runId}/events`),
  artifacts: (runId?: string) => request<Artifact[]>(runId ? `/api/v1/artifacts?run_id=${encodeURIComponent(runId)}` : "/api/v1/artifacts"),
  artifactUrl: (artifactId: string) => `/api/v1/artifacts/${encodeURIComponent(artifactId)}/content`,
  streamAsk,
  cancelRun: (runId: string) => request<Run>(`/api/v1/runs/${runId}/cancel`, { method: "POST" }),
  resumeRun: (runId: string) => request<ResumeResponse>(`/api/v1/runs/${runId}/resume`, { method: "POST" }),
  modelStatus: () => request<ModelStatus>("/api/v1/models"),
};

function resultFromRun(run: Run): Result | null {
  const saved = run.metadata.result;
  return saved && typeof saved === "object" ? saved : null;
}

export async function fetchRunView(runId: string, knownRuns: Run[] = []): Promise<RunDetails> {
  const run = await api.run(runId);
  const allRuns = knownRuns.length > 0 ? knownRuns : await api.runs();
  const children = run.parent_run_id === null || run.parent_run_id === undefined
    ? allRuns.filter((item) => item.parent_run_id === run.id)
    : [];
  const traceRunIds = [run.id, ...children.map((item) => item.id)];
  const eventGroups = await Promise.all(traceRunIds.map((id) => api.events(id)));
  const events = eventGroups.flat().sort((left, right) => {
    const timestamp = left.timestamp.localeCompare(right.timestamp);
    if (timestamp !== 0) return timestamp;
    const runOrder = left.run_id.localeCompare(right.run_id);
    return runOrder !== 0 ? runOrder : left.sequence - right.sequence;
  });
  const relatedResults = [run, ...children].map(resultFromRun).filter((item): item is Result => item !== null);
  const result = resultFromRun(run) ?? relatedResults[0] ?? null;
  const artifactIds = new Set(relatedResults.flatMap((item) => item.artifacts));
  const artifacts = artifactIds.size > 0
    ? (await api.artifacts()).filter((item) => artifactIds.has(item.id))
    : [];
  return { run, result, events, artifacts };
}

import { Event, InteractionMode, Run, RunStatus } from "./api";

export const ACTIVE_RUN_STATUSES: ReadonlySet<RunStatus> = new Set([
  "CREATED",
  "PLANNING",
  "RUNNING",
  "WAITING_TOOL",
  "WAITING_SUBAGENT",
  "WAITING_USER",
  "WAITING_APPROVAL",
  "RETRYING",
  "REPLANNING",
  "VALIDATING",
]);

export const TERMINAL_RUN_STATUSES: ReadonlySet<RunStatus> = new Set([
  "COMPLETED",
  "PARTIAL_COMPLETED",
  "FAILED",
  "INTERRUPTED",
  "CANCELLED",
  "BUDGET_EXCEEDED",
]);

export const RESUMABLE_RUN_STATUSES: ReadonlySet<RunStatus> = new Set(["CANCELLED", "INTERRUPTED"]);

export function isActiveRun(run: Run): boolean {
  return ACTIVE_RUN_STATUSES.has(run.status);
}

export function isMainRun(run: Run): boolean {
  return run.parent_run_id === null || run.parent_run_id === undefined;
}

export function childRunsOf(runId: string, runs: Run[]): Run[] {
  return runs.filter((run) => run.parent_run_id === runId);
}

export function runsForConversation(conversationId: string, runs: Run[]): Run[] {
  return conversationId ? runs.filter((run) => run.conversation_id === conversationId) : [];
}

export function interactionModeLabel(mode: InteractionMode | undefined): string {
  const labels: Record<string, string> = {
    new_task: "新任务",
    continue_task: "继续任务",
    modify_task: "修改任务",
    retry_task: "重试任务",
    query: "查询",
    chat: "对话",
    cancel_task: "取消任务",
  };
  return mode ? labels[mode] ?? mode : "未标注";
}

export function lineageSource(run: Run): string | null {
  return run.metadata.continued_from ?? run.metadata.retry_of ?? run.metadata.resumed_from ?? null;
}

export function runTitle(run: Run): string {
  return String(run.metadata.subtask_goal ?? run.metadata.goal ?? run.agent_id);
}

export function eventRuns(event: Event, runs: Run[]): Run | undefined {
  return runs.find((run) => run.id === event.run_id);
}

export function groupRunsByTask(runs: Run[]): Map<string, Run[]> {
  const groups = new Map<string, Run[]>();
  for (const run of runs) {
    const key = run.task_id ?? "__command__";
    groups.set(key, [...(groups.get(key) ?? []), run]);
  }
  return groups;
}

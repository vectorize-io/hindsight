/**
 * The reflect loop's trace as one timeline, shared by the reflect view and the
 * mental-model diagnostics (a refresh runs the same loop).
 */

export type TraceCall = { duration_ms: number; started_at_ms?: number };
export type LLMTraceCall = TraceCall & { scope: string };
export type ToolTraceCall = TraceCall & { tool: string; iteration?: number };
export type StepTone = "turn" | "check" | "answer" | "error";
export type LLMStep = {
  title: string;
  detail: string;
  tone: StepTone;
  turn?: number;
  values?: Record<string, string | number>;
};
export type TimelineItem<T extends ToolTraceCall = ToolTraceCall> =
  | { kind: "llm"; step: LLMStep; duration_ms: number; offset_ms: number; start: number }
  | { kind: "tools"; tools: T[]; offset_ms: number; start: number };

/** What one LLM call in the reflect loop did, read from the scope the engine records. */
function describeLLMCall(scope: string, toolIterations: Set<number>): LLMStep {
  const turn = scope.match(/^agent_(\d+)(_err)?$/);
  if (turn) {
    const n = Number(turn[1]);
    if (turn[2])
      return {
        title: "traceAgentFailed",
        detail: "traceAgentFailedDetail",
        tone: "error",
        turn: n,
      };
    // A turn whose tools ran chose them; a turn without tools called done.
    return toolIterations.has(n)
      ? { title: "agentDecided", detail: "calledToolsBelow", tone: "turn", turn: n }
      : { title: "traceAgentDone", detail: "traceAgentDoneDetail", tone: "turn", turn: n };
  }
  const map = scope.match(/^final_map_(\d+)$/);
  if (map) {
    const part = Number(map[1]) + 1;
    return {
      title: "traceFinalMap",
      detail: "traceFinalMapDetail",
      tone: "turn",
      values: { part },
    };
  }
  switch (scope) {
    case "round_query":
      return { title: "traceRoundQuery", detail: "traceRoundQueryDetail", tone: "turn" };
    case "decision_sufficiency":
      return { title: "traceSufficiency", detail: "traceSufficiencyDetail", tone: "check" };
    case "decision_pages_sufficiency":
      return {
        title: "tracePagesSufficiency",
        detail: "tracePagesSufficiencyDetail",
        tone: "check",
      };
    case "decision_prune":
      return { title: "tracePrune", detail: "tracePruneDetail", tone: "check" };
    case "closing_done":
      return { title: "responseGenerated", detail: "traceClosingDoneDetail", tone: "answer" };
    case "final":
      return { title: "responseGenerated", detail: "traceFinalDetail", tone: "answer" };
    case "final_rewrite":
      return { title: "traceRewrite", detail: "traceRewriteDetail", tone: "answer" };
    case "reflect_structured":
      return { title: "traceStructured", detail: "traceStructuredDetail", tone: "answer" };
    default:
      return {
        title: "traceOtherCall",
        detail: "traceOtherCallDetail",
        tone: "turn",
        values: { scope },
      };
  }
}

/**
 * The reflect loop in the order it ran. The engine records LLM calls and tool calls in
 * separate lists, and LLM calls are not one-per-iteration (search-query rewrite, decision
 * checks, pruning and the closing answer are calls too; forced steps make none), so the
 * two lists are merged on each call's start time. Tools of one batch run in parallel and
 * share a start, so they show as one step.
 */
export function buildReflectTimeline<T extends ToolTraceCall>(
  llmCalls: LLMTraceCall[],
  toolCalls: T[]
): TimelineItem<T>[] {
  const toolIterations = new Set(toolCalls.map((tc) => tc.iteration ?? 0));
  const batches = new Map<string, T[]>();
  for (const tc of toolCalls) {
    const key = `${tc.iteration ?? 0}:${tc.started_at_ms ?? 0}`;
    batches.set(key, [...(batches.get(key) ?? []), tc]);
  }
  const items: TimelineItem<T>[] = [
    ...llmCalls.map(
      (lc): TimelineItem<T> => ({
        kind: "llm",
        step: describeLLMCall(lc.scope, toolIterations),
        duration_ms: lc.duration_ms,
        offset_ms: 0,
        start: lc.started_at_ms ?? 0,
      })
    ),
    ...[...batches.values()].map(
      (tools): TimelineItem<T> => ({
        kind: "tools",
        tools,
        offset_ms: 0,
        start: tools[0].started_at_ms ?? 0,
      })
    ),
  ];
  items.sort((x, y) => x.start - y.start);
  // With a decision model the last free turn's done is the answer itself (no closing call).
  const last = items[items.length - 1];
  if (last?.kind === "llm" && last.step.title === "traceAgentDone") {
    last.step = { ...last.step, title: "responseGenerated", detail: "finalAnswer", tone: "answer" };
  }
  const origin = items[0]?.start ?? 0;
  for (const item of items) item.offset_ms = item.start - origin;
  return items;
}

/** Loop iterations that ran tools; the response carries no iteration count of its own. */
export function traceIterations(toolCalls: ToolTraceCall[]): number {
  return Math.max(0, ...toolCalls.map((tc) => tc.iteration ?? 0));
}

/** Wall time from the first call's start to the last call's end; steps overlap, so not a sum. */
export function traceWallMs(llmCalls: TraceCall[], toolCalls: TraceCall[]): number {
  const calls = [...llmCalls, ...toolCalls];
  if (calls.length === 0) return 0;
  const start = Math.min(...calls.map((c) => c.started_at_ms ?? 0));
  const end = Math.max(...calls.map((c) => (c.started_at_ms ?? 0) + c.duration_ms));
  return end - start;
}

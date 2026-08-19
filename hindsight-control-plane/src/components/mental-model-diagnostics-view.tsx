"use client";

import { useTranslations } from "next-intl";
import {
  MentalModelDryRunRefreshResult,
  MentalModelRefreshTrace,
  ModeFallbackReason,
  RefreshOutcome,
} from "@/lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { AlertTriangle, Check, Clock, Play } from "lucide-react";
import { formatAbsoluteDateTime as formatDateTime } from "@/lib/relative-time";
import JsonView from "react18-json-view";
import "react18-json-view/src/style.css";

/** One step of the reconstructed agent timeline: an LLM turn, or the tools it then called. */
type TimelineStep =
  | { type: "llm"; iteration: number; isFinal: boolean; scope: string; durationMs: number }
  | { type: "tools"; iteration: number; tools: MentalModelRefreshTrace["tool_calls"] };

/**
 * Rebuild the interleaved LLM -> tools -> LLM timeline from the flat trace,
 * matching how the reflect view presents an agent run. Tool calls carry the
 * iteration they belong to; the final LLM turn is the one that produced the
 * answer rather than more tool calls.
 */
function buildTimeline(trace: MentalModelRefreshTrace): TimelineStep[] {
  const llmCalls = trace.llm_calls ?? [];
  const toolCalls = trace.tool_calls ?? [];
  const steps: TimelineStep[] = [];

  llmCalls.forEach((lc, idx) => {
    const iterTools = toolCalls.filter((tc) => tc.iteration === idx + 1);
    const isLast = idx === llmCalls.length - 1;
    const isFinal = lc.scope.includes("final") || (isLast && iterTools.length === 0);
    steps.push({
      type: "llm",
      iteration: isFinal ? llmCalls.length : idx + 1,
      isFinal,
      scope: lc.scope,
      durationMs: lc.duration_ms,
    });
    if (iterTools.length > 0) {
      steps.push({ type: "tools", iteration: idx + 1, tools: iterTools });
    }
  });

  // Tool calls whose iteration has no matching LLM turn would otherwise vanish.
  const covered = new Set(steps.flatMap((s) => (s.type === "tools" ? [s.iteration] : [])));
  const orphans = toolCalls.filter((tc) => !covered.has(tc.iteration));
  if (orphans.length > 0) {
    steps.push({ type: "tools", iteration: -1, tools: orphans });
  }
  return steps;
}

/** The agent run, as a vertical timeline. Shared with the History tab, which
 *  shows the trace of the refresh that produced each version. */
export function ExecutionTimeline({ trace }: { trace: MentalModelRefreshTrace }) {
  const t = useTranslations("mentalModelDiagnostics");
  const timeline = buildTimeline(trace);
  const totalMs =
    (trace.llm_calls ?? []).reduce((s, c) => s + c.duration_ms, 0) +
    (trace.tool_calls ?? []).reduce((s, c) => s + c.duration_ms, 0);

  return (
    <Card className="h-fit">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">{t("executionTraceTitle")}</CardTitle>
        <CardDescription className="text-xs">
          {t("executionTraceDescription", {
            iterations: trace.llm_calls?.length ?? 0,
            totalMs,
          })}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {timeline.length === 0 ? (
          <p className="text-sm text-muted-foreground italic">{t("noOperations")}</p>
        ) : (
          <div className="max-h-[500px] overflow-y-auto pr-2">
            {timeline.map((step, idx) => (
              <div key={idx} className="relative">
                {idx < timeline.length - 1 && (
                  <div className="absolute left-3 top-6 bottom-0 w-0.5 bg-border" />
                )}
                {step.type === "llm" ? (
                  <div className="flex items-start gap-3 pb-3">
                    <div
                      className={`w-6 h-6 rounded-full flex items-center justify-center shrink-0 ${
                        step.isFinal
                          ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400"
                          : "bg-primary/10 text-primary"
                      }`}
                    >
                      {step.isFinal ? (
                        <Check className="w-3.5 h-3.5" strokeWidth={2.5} />
                      ) : (
                        <span className="text-[10px] font-semibold">{step.iteration}</span>
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-sm">
                          {step.isFinal ? t("responseGenerated") : t("agentDecided")}
                        </span>
                        <span className="text-xs text-muted-foreground flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {step.durationMs}ms
                        </span>
                      </div>
                      <span className="text-xs text-muted-foreground font-mono">{step.scope}</span>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-start gap-3 pb-3">
                    <div className="w-6 h-6 rounded-full flex items-center justify-center bg-blue-500/15 text-blue-600 dark:text-blue-400 shrink-0">
                      <Play className="w-3 h-3" fill="currentColor" />
                    </div>
                    <div className="flex-1 min-w-0 space-y-2">
                      <div className="text-xs text-muted-foreground">
                        {t("executingTools", { count: step.tools.length })}
                      </div>
                      {step.tools.map((tc, tcIdx) => (
                        <div
                          key={tcIdx}
                          className="border border-border rounded-lg overflow-hidden"
                        >
                          <div className="flex items-center justify-between gap-2 px-3 py-1.5 bg-muted/50">
                            <span className="font-medium text-sm text-foreground truncate">
                              {tc.tool}
                            </span>
                            <span className="flex items-center gap-2 shrink-0">
                              <span
                                className={`text-[11px] px-1.5 py-0.5 rounded font-medium ${
                                  (tc.result_count ?? 0) === 0
                                    ? "bg-amber-500/15 text-amber-700 dark:text-amber-400"
                                    : "bg-muted text-muted-foreground"
                                }`}
                              >
                                {t("traceResults", { count: tc.result_count ?? 0 })}
                              </span>
                              <span className="text-xs text-muted-foreground flex items-center gap-1">
                                <Clock className="w-3 h-3" />
                                {tc.duration_ms}ms
                              </span>
                            </span>
                          </div>
                          <div className="p-2 space-y-2">
                            {tc.reason && (
                              <p className="text-xs text-muted-foreground italic">{tc.reason}</p>
                            )}
                            <p className="text-[10px] text-muted-foreground">
                              {tc.updated_at
                                ? t("toolWindowBound", { time: formatDateTime(tc.updated_at) })
                                : t("toolWindowUnbounded")}
                            </p>
                            <div>
                              <p className="text-[10px] font-semibold text-muted-foreground mb-1">
                                {t("toolInputLabel")}
                              </p>
                              <div className="bg-muted p-1.5 rounded text-xs overflow-auto max-h-32">
                                <JsonView src={tc.input} collapsed={1} theme="default" />
                              </div>
                            </div>
                            {tc.output && (
                              <div>
                                <p className="text-[10px] font-semibold text-muted-foreground mb-1">
                                  {t("toolOutputLabel")}
                                </p>
                                <div className="bg-muted p-1.5 rounded text-xs overflow-auto max-h-48">
                                  <JsonView src={tc.output} collapsed={1} theme="default" />
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function TraceSummary({ trace }: { trace: MentalModelRefreshTrace }) {
  const t = useTranslations("mentalModelDiagnostics");
  const outcomeLabels: Record<RefreshOutcome, string> = {
    content_written: t("outcomeWritten"),
    content_preserved_no_new_facts: t("outcomePreserved"),
    refresh_failed_empty_candidate: t("outcomeFailed"),
    refresh_failed_delta_not_applied: t("outcomeDeltaNotApplied"),
  };
  const fallbackLabels: Record<ModeFallbackReason, string> = {
    no_baseline_content: t("fallbackNoBaseline"),
    source_query_changed: t("fallbackQueryChanged"),
    structured_doc_unreadable: t("fallbackDocUnreadable"),
    delta_ops_failed: t("fallbackOpsFailed"),
    delta_ops_all_skipped: t("fallbackOpsAllSkipped"),
  };
  return (
    <span className="text-xs text-muted-foreground">
      <span className="font-mono">{trace.effective_mode}</span> &middot;{" "}
      {outcomeLabels[trace.outcome]}
      {trace.mode_fallback_reason && (
        <span className="text-amber-700 dark:text-amber-400">
          {" "}
          &middot; {fallbackLabels[trace.mode_fallback_reason]}
        </span>
      )}
    </span>
  );
}

function WarningList({ warnings, title }: { warnings: string[]; title: string }) {
  if (!warnings?.length) return null;
  return (
    <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 space-y-1.5">
      <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-amber-700 dark:text-amber-400">
        <AlertTriangle className="w-3.5 h-3.5" />
        {title}
      </div>
      {warnings.map((w, idx) => (
        <p key={idx} className="text-sm text-amber-900 dark:text-amber-200">
          {w}
        </p>
      ))}
    </div>
  );
}

/**
 * The result of a dry run, shown as a would-be new version: the same before/after
 * components the History tab uses, with the execution trace underneath.
 *
 * There is no tab for this — a dry run is an action, and its result is transient,
 * so it lives in a dialog. Past refreshes belong to History.
 */
export function MentalModelDryRunDialog({
  result,
  mentalModelName,
  currentBasedOn,
  onClose,
  renderPreviewDiff,
}: {
  result: MentalModelDryRunRefreshResult | null;
  mentalModelName: string;
  /** The evidence the stored version rests on — the "before" side of the diff. */
  currentBasedOn: Record<string, unknown[]> | undefined;
  onClose: () => void;
  /** Supplied by the detail modal so a preview reads like a History entry. */
  renderPreviewDiff: (args: {
    before: string;
    after: string;
    beforeBasedOn: Record<string, unknown[]> | undefined;
    afterBasedOn: Record<string, unknown[]> | undefined;
  }) => React.ReactNode;
}) {
  const t = useTranslations("mentalModelDiagnostics");

  return (
    <Dialog open={result !== null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="w-[95vw] max-w-[95vw] h-[92vh] sm:max-w-[95vw] flex flex-col overflow-hidden">
        <DialogHeader className="pr-10">
          <DialogTitle>{t("resultTitle", { name: mentalModelName })}</DialogTitle>
        </DialogHeader>
        {result && (
          <div className="flex-1 overflow-y-auto space-y-4">
            <WarningList warnings={result.warnings ?? []} title={t("warningsTitle")} />
            {renderPreviewDiff({
              before: result.current_content,
              after: result.preview_content,
              beforeBasedOn: currentBasedOn,
              afterBasedOn: result.based_on as Record<string, unknown[]> | undefined,
            })}
            <ExecutionTimeline trace={result.trace} />
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

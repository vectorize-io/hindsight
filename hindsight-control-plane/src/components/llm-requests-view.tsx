"use client";

import { useState, useEffect, useCallback } from "react";
import { useTranslations } from "next-intl";
import { resolveDateRangePreset } from "@/lib/date-range-preset";
import { useBank } from "@/lib/bank-context";
import { client, LLMCallType, LLMRequestEntry, LLMRequestStatsBucket } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { JsonViewer } from "@/components/ui/json-viewer";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ChevronLeft, ChevronRight, CheckCircle, RefreshCw, XCircle } from "lucide-react";
import { Spinner } from "@/components/ui/spinner";
import { ToneChip, type ChipTone } from "@/components/ui/facet-chip";
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";

type TranslateFn = (key: string) => string;

function getStatusOptions(t: TranslateFn) {
  return [
    { value: "all", label: t("statusAll") },
    { value: "success", label: t("statusSuccess") },
    { value: "error", label: t("statusError") },
  ];
}

function getOperationOptions(t: TranslateFn) {
  return [
    { value: "all", label: t("operationAll") },
    { value: "retain", label: t("operationRetain") },
    { value: "recall", label: t("operationRecall") },
    { value: "reflect", label: t("operationReflect") },
    { value: "consolidation", label: t("operationConsolidation") },
    { value: "refresh_mental_model", label: t("operationRefreshMentalModel") },
    { value: "mental_model_write", label: t("operationMentalModelWrite") },
  ];
}

function getCallTypeOptions(t: TranslateFn) {
  return [
    { value: "all", label: t("callTypeAll") },
    { value: "llm", label: t("callTypeLlm") },
    { value: "embedding", label: t("callTypeEmbedding") },
    { value: "rerank", label: t("callTypeRerank") },
    { value: "decision", label: t("callTypeDecision") },
  ];
}

// Mirrors the server's call_type rule (llm_trace.py): the kind of a call is read off its scope.
const CALL_KINDS: LLMCallType[] = ["llm", "embedding", "rerank", "decision"];

function callKind(scope: string | null | undefined): LLMCallType {
  if (!scope) return "llm";
  if (scope.startsWith("embedding_")) return "embedding";
  if (scope === "rerank") return "rerank";
  if (scope.startsWith("decision_")) return "decision";
  return "llm";
}

const KIND_TONES: Record<LLMCallType, ChipTone> = {
  llm: "blue",
  embedding: "violet",
  rerank: "amber",
  decision: "emerald",
};

const KIND_LABEL_KEYS: Record<LLMCallType, string> = {
  llm: "callTypeLlm",
  embedding: "callTypeEmbedding",
  rerank: "callTypeRerank",
  decision: "callTypeDecision",
};

// The count is the chip's second segment ("LLM | 5"), so the label never needs pluralising.
function KindBadge({ kind, t, count }: { kind: LLMCallType; t: TranslateFn; count?: number }) {
  return (
    <ToneChip
      tone={KIND_TONES[kind]}
      label={t(KIND_LABEL_KEYS[kind])}
      detail={count}
      className="shrink-0"
    />
  );
}

// One call's kind and scope as a single chip: the kind, then the scope with the kind's
// prefix dropped ("Embedding | document", not "Embedding embedding_document"). A scope that
// is nothing but its kind ("rerank") is just the kind. The full scope is the tooltip.
const KIND_SCOPE_PREFIXES: Partial<Record<LLMCallType, string>> = {
  embedding: "embedding_",
  rerank: "rerank",
  decision: "decision_",
};

function ScopeBadge({ scope, t }: { scope: string | null | undefined; t: TranslateFn }) {
  const kind = callKind(scope);
  const prefix = KIND_SCOPE_PREFIXES[kind];
  const rest = scope && prefix && scope.startsWith(prefix) ? scope.slice(prefix.length) : scope;
  return (
    <ToneChip
      tone={KIND_TONES[kind]}
      label={t(KIND_LABEL_KEYS[kind])}
      detail={rest}
      monoDetail
      title={scope ?? undefined}
    />
  );
}

// One badge per kind present in a run, in a fixed order, with its call count.
function KindBreakdown({ rows, t }: { rows: LLMRequestEntry[]; t: TranslateFn }) {
  const counts = new Map<LLMCallType, number>();
  for (const r of rows) counts.set(callKind(r.scope), (counts.get(callKind(r.scope)) ?? 0) + 1);
  return (
    <span className="inline-flex flex-wrap gap-1">
      {CALL_KINDS.filter((k) => counts.has(k)).map((k) => (
        <KindBadge key={k} kind={k} t={t} count={counts.get(k)} />
      ))}
    </span>
  );
}

function getPeriodOptions(t: TranslateFn) {
  return [
    { value: "1d", label: t("periodToday") },
    { value: "7d", label: t("periodLast7Days") },
    { value: "30d", label: t("periodLast30Days") },
  ];
}

function formatDurationMs(ms: number | null): string {
  if (ms === null || ms === undefined) return "—";
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

// Compact token count: 950 → "950", 1300 → "1.3k", 2_400_000 → "2.4M".
function formatTokens(n: number): string {
  if (n < 1000) return `${n}`;
  if (n < 1_000_000) return `${(n / 1000).toFixed(1).replace(/\.0$/, "")}k`;
  return `${(n / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
}

// The API keeps total_tokens as input + visible output for compatibility.
// Add reported reasoning tokens when displaying a complete token total.
export function totalTokens(entry: LLMRequestEntry): number | null {
  if (entry.total_tokens == null) {
    return entry.thoughts_tokens && entry.thoughts_tokens > 0 ? entry.thoughts_tokens : null;
  }
  return entry.total_tokens + (entry.thoughts_tokens ?? 0);
}

// Pull a list of memory_unit ids out of trace metadata (memory_ids /
// source_memory_ids). Tolerates missing / non-array values.
function metadataIdList(
  metadata: Record<string, unknown> | null | undefined,
  key: string
): string[] {
  const raw = metadata?.[key];
  return Array.isArray(raw) ? raw.filter((v): v is string => typeof v === "string") : [];
}

// Trace-level mapping of the memory_units an operation produced/consumed,
// rendered as truncated mono chips with the full id on hover.
function MemoryIdList({ label, ids }: { label: string; ids: string[] }) {
  return (
    <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
      <span className="text-muted-foreground">{label}</span>
      {ids.map((id) => (
        <code key={id} title={id} className="font-mono text-xs text-foreground">
          {id.slice(0, 8)}
        </code>
      ))}
    </div>
  );
}

function formatDateTime(ts: string | null): string {
  if (!ts) return "—";
  const date = new Date(ts);
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatChartLabel(ts: string, trunc: string): string {
  const date = new Date(ts);
  if (trunc === "hour") {
    return date.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
  }
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function StatusBadge({ status }: { status: string }) {
  const isError = status === "error";
  const Icon = isError ? XCircle : CheckCircle;
  return (
    <ToneChip
      tone={isError ? "red" : "emerald"}
      label={status}
      icon={<Icon className="h-3 w-3" aria-hidden />}
    />
  );
}

// ---- Trace grouping (parent operation run → child LLM calls) ----

interface TraceGroup {
  key: string;
  rows: LLMRequestEntry[];
  grouped: boolean; // true when this is a multi-call operation run
}

// Group page rows by trace_id, preserving the incoming (newest-first) order.
// Rows without a trace_id, or a run with a single call, render as plain rows.
function groupByTrace(rows: LLMRequestEntry[]): TraceGroup[] {
  const groups: TraceGroup[] = [];
  const indexByTrace = new Map<string, number>();
  for (const row of rows) {
    if (row.trace_id) {
      let gi = indexByTrace.get(row.trace_id);
      if (gi === undefined) {
        gi = groups.length;
        indexByTrace.set(row.trace_id, gi);
        groups.push({ key: row.trace_id, rows: [], grouped: true });
      }
      groups[gi].rows.push(row);
    } else {
      groups.push({ key: row.id, rows: [row], grouped: false });
    }
  }
  for (const g of groups) if (g.rows.length < 2) g.grouped = false;
  return groups;
}

interface GroupAgg {
  tokens: number;
  status: string;
  startIso: string | null;
  durationMs: number | null;
}

// Aggregate a run: summed tokens, error-if-any, earliest start, and wall-clock
// duration (first call start → last call end) — the operation span.
function aggregateGroup(rows: LLMRequestEntry[]): GroupAgg {
  const tokens = rows.reduce((sum, r) => sum + (totalTokens(r) ?? 0), 0);
  const starts = rows
    .map((r) => (r.started_at ? new Date(r.started_at).getTime() : NaN))
    .filter((n) => !isNaN(n));
  const ends = rows
    .map((r) => (r.ended_at ? new Date(r.ended_at).getTime() : NaN))
    .filter((n) => !isNaN(n));
  const start = starts.length ? Math.min(...starts) : null;
  const end = ends.length ? Math.max(...ends) : null;
  return {
    tokens,
    status: rows.some((r) => r.status === "error") ? "error" : "success",
    startIso: start != null ? new Date(start).toISOString() : null,
    durationMs: start != null && end != null ? end - start : null,
  };
}

// ---- Trace dialog (Langsmith-style: span waterfall + selected-span details) ----

// One span in the left-hand waterfall: name, status dot, duration/tokens, and a
// timeline bar positioned by its offset within the trace.
function SpanRow({
  span,
  traceStart,
  traceSpan,
  selected,
  onClick,
  t,
}: {
  span: LLMRequestEntry;
  traceStart: number;
  traceSpan: number;
  selected: boolean;
  onClick: () => void;
  t: TranslateFn;
}) {
  const start = span.started_at ? new Date(span.started_at).getTime() : traceStart;
  const end = span.ended_at ? new Date(span.ended_at).getTime() : start;
  const left = Math.min(98, Math.max(0, ((start - traceStart) / traceSpan) * 100));
  const width = Math.max(2, Math.min(100 - left, ((end - start) / traceSpan) * 100));
  const isError = span.status === "error";
  const spanTokens = totalTokens(span);
  return (
    <button
      type="button"
      onClick={onClick}
      className={`w-full text-left rounded-md px-2 py-1.5 transition-colors ${
        selected ? "bg-muted" : "hover:bg-muted/50"
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="inline-flex items-center gap-1.5 min-w-0">
          <span
            className={`w-1.5 h-1.5 rounded-full shrink-0 ${isError ? "bg-red-500" : "bg-green-500"}`}
          />
          <ScopeBadge scope={span.scope || span.operation || "call"} t={t} />
        </span>
        <span className="text-[10px] text-muted-foreground font-mono shrink-0">
          {formatDurationMs(span.duration_ms)}
          {spanTokens != null ? ` · ${formatTokens(spanTokens)}` : ""}
        </span>
      </div>
      <div className="relative h-1.5 mt-1 rounded bg-muted/40">
        <div
          className={`absolute top-0 h-full rounded ${isError ? "bg-red-500/70" : "bg-primary/70"}`}
          style={{ left: `${left}%`, width: `${width}%` }}
        />
      </div>
    </button>
  );
}

// Right-hand detail panel for the selected span.
function SpanDetails({ entry, t }: { entry: LLMRequestEntry; t: TranslateFn }) {
  // A non-LLM call's input count is already shown as "Texts"/"Documents" above, so it is
  // dropped here; the box only appears if anything else is left.
  const isLlm = callKind(entry.scope) === "llm";
  const llmInfo =
    entry.llm_info && !isLlm
      ? Object.fromEntries(Object.entries(entry.llm_info).filter(([k]) => k !== "input_count"))
      : entry.llm_info;
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
        <div>
          <span className="text-muted-foreground">{t("detailScope")}</span>{" "}
          <span className="font-mono">{entry.scope || "—"}</span>
        </div>
        <div>
          <span className="text-muted-foreground">{t("detailStatus")}</span>{" "}
          <StatusBadge status={entry.status} />
        </div>
        <div>
          <span className="text-muted-foreground">{t("detailProvider")}</span>{" "}
          <span className="font-mono">{entry.provider || "—"}</span>
        </div>
        <div>
          <span className="text-muted-foreground">{t("detailModel")}</span>{" "}
          <span className="font-mono">{entry.model || "—"}</span>
        </div>
        <div>
          <span className="text-muted-foreground">{t("detailStarted")}</span>{" "}
          <span className="font-mono">{formatDateTime(entry.started_at)}</span>
        </div>
        <div>
          <span className="text-muted-foreground">{t("detailDuration")}</span>{" "}
          <span className="font-mono">{formatDurationMs(entry.duration_ms)}</span>
        </div>
      </div>

      {!isLlm ? (
        // Not a chat completion: no output, reasoning or cache tokens to show. What these
        // bill by is the input tokens and, for embeddings and rerank, how many texts went in.
        <div className="flex flex-wrap gap-x-6 gap-y-1 text-sm border-t border-border pt-3">
          <div>
            <span className="text-muted-foreground">{t("tokensInput")}:</span>{" "}
            <span className="font-mono">{entry.input_tokens ?? "—"}</span>
          </div>
          {typeof entry.llm_info?.input_count === "number" && (
            <div>
              <span className="text-muted-foreground">
                {t(callKind(entry.scope) === "rerank" ? "inputCountDocuments" : "inputCountTexts")}:
              </span>{" "}
              <span className="font-mono">{entry.llm_info.input_count as number}</span>
            </div>
          )}
        </div>
      ) : (
        (entry.total_tokens !== null ||
          entry.input_tokens !== null ||
          entry.output_tokens !== null ||
          entry.thoughts_tokens != null) && (
          <div className="flex flex-wrap gap-x-6 gap-y-1 text-sm border-t border-border pt-3">
            <div>
              <span className="text-muted-foreground">{t("tokensInput")}:</span>{" "}
              <span className="font-mono">{entry.input_tokens ?? "—"}</span>
            </div>
            <div>
              <span className="text-muted-foreground">{t("tokensOutput")}:</span>{" "}
              <span className="font-mono">{entry.output_tokens ?? "—"}</span>
            </div>
            <div>
              <span className="text-muted-foreground">{t("tokensThoughts")}:</span>{" "}
              <span className="font-mono">{entry.thoughts_tokens ?? "—"}</span>
            </div>
            <div>
              <span className="text-muted-foreground">{t("tokensCached")}:</span>{" "}
              <span className="font-mono">{entry.cached_tokens ?? "—"}</span>
            </div>
            <div>
              <span className="text-muted-foreground">{t("tokensTotal")}:</span>{" "}
              <span className="font-mono font-medium">{totalTokens(entry) ?? "—"}</span>
            </div>
          </div>
        )
      )}

      <div className="border-t border-border pt-3 space-y-1 text-sm">
        <div>
          <span className="text-muted-foreground">{t("detailSpan")}</span>{" "}
          <code className="font-mono text-xs">{entry.span_id || "—"}</code>
        </div>
        <div>
          <span className="text-muted-foreground">{t("detailParentSpan")}</span>{" "}
          <code className="font-mono text-xs">{entry.parent_span_id || "—"}</code>
        </div>
      </div>

      {entry.error && (
        <div>
          <h4 className="text-sm font-semibold mb-2">{t("detailError")}</h4>
          <JsonViewer
            value={entry.error}
            className="bg-red-50 dark:bg-red-900/20 text-red-800 dark:text-red-300 max-h-[240px] overflow-y-auto"
          />
        </div>
      )}
      {entry.input !== null && entry.input !== undefined && (
        <div>
          <h4 className="text-sm font-semibold mb-2">{t("detailInput")}</h4>
          <JsonViewer value={entry.input} className="bg-muted max-h-[320px] overflow-y-auto" />
        </div>
      )}
      {entry.output !== null && entry.output !== undefined && (
        <div>
          <h4 className="text-sm font-semibold mb-2">{t("detailOutput")}</h4>
          <JsonViewer value={entry.output} className="bg-muted max-h-[320px] overflow-y-auto" />
        </div>
      )}
      {llmInfo && Object.keys(llmInfo).length > 0 && (
        <div>
          <h4 className="text-sm font-semibold mb-2">{t("detailLlmInfo")}</h4>
          <JsonViewer value={llmInfo} />
        </div>
      )}
      {entry.metadata && Object.keys(entry.metadata).length > 0 && (
        <div>
          <h4 className="text-sm font-semibold mb-2">{t("detailMetadata")}</h4>
          <JsonViewer value={entry.metadata} />
        </div>
      )}
    </div>
  );
}

// Full trace view: fetches every call sharing the entry's trace_id and shows the
// operation run as parent (header) → child LLM calls (waterfall) → details.
export function TraceDialog({
  bankId,
  entry,
  open,
  onOpenChange,
}: {
  bankId: string;
  entry: LLMRequestEntry | null;
  open: boolean;
  onOpenChange: (o: boolean) => void;
}) {
  const t = useTranslations("llmRequestsView");
  const [spans, setSpans] = useState<LLMRequestEntry[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!open || !entry) return;
    let cancelled = false;
    const load = async () => {
      if (!entry.trace_id) {
        setSpans([entry]);
        setSelectedId(entry.id);
        return;
      }
      setLoading(true);
      try {
        const data = await client.listLLMRequests(bankId, { trace_id: entry.trace_id, limit: 500 });
        const rows = (data.items || [])
          .slice()
          .sort((a, b) => (a.started_at || "").localeCompare(b.started_at || ""));
        if (!cancelled) {
          setSpans(rows.length ? rows : [entry]);
          setSelectedId(entry.id);
        }
      } catch {
        if (!cancelled) {
          setSpans([entry]);
          setSelectedId(entry.id);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [open, entry, bankId]);

  const selected = spans.find((s) => s.id === selectedId) ?? entry;
  const startTimes = spans
    .map((s) => (s.started_at ? new Date(s.started_at).getTime() : NaN))
    .filter((n) => !isNaN(n));
  const endTimes = spans
    .map((s) => (s.ended_at ? new Date(s.ended_at).getTime() : NaN))
    .filter((n) => !isNaN(n));
  const traceStart = startTimes.length ? Math.min(...startTimes) : 0;
  const traceEnd = endTimes.length ? Math.max(...endTimes) : 0;
  const traceSpan = Math.max(1, traceEnd - traceStart);
  // LLM calls only: an embedding's or decision's input tokens summed into the same "Input"
  // would read as prompt tokens. Each other call shows its own tokens on its row.
  const llmSpans = spans.filter((s) => callKind(s.scope) === "llm");
  const inputTokens = llmSpans.reduce((sum, r) => sum + (r.input_tokens ?? 0), 0);
  const outputTokens = llmSpans.reduce((sum, r) => sum + (r.output_tokens ?? 0), 0);
  const thoughtsTokens = llmSpans.reduce((sum, r) => sum + (r.thoughts_tokens ?? 0), 0);
  const anyError = spans.some((s) => s.status === "error");
  const op = entry?.operation ?? spans[0]?.operation ?? "—";
  // memory_ids / source_memory_ids are attached identically to every row of a
  // trace, so read them from whichever row carries them.
  const traceMeta =
    spans.find(
      (s) =>
        metadataIdList(s.metadata, "memory_ids").length ||
        metadataIdList(s.metadata, "source_memory_ids").length
    )?.metadata ?? entry?.metadata;
  const createdMemoryIds = metadataIdList(traceMeta, "memory_ids");
  const sourceMemoryIds = metadataIdList(traceMeta, "source_memory_ids");

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-5xl w-[92vw] h-[85vh] flex flex-col gap-3">
        <DialogHeader className="shrink-0">
          <DialogTitle className="flex items-center gap-2">
            <span className="font-mono">{op}</span>
            <StatusBadge status={anyError ? "error" : "success"} />
          </DialogTitle>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
            <KindBreakdown rows={spans} t={t} />
            {llmSpans.length > 0 && (
              <span>
                {t("tokensInput")}{" "}
                <span className="font-mono text-foreground">{formatTokens(inputTokens)}</span>
                {" · "}
                {t("tokensOutput")}{" "}
                <span className="font-mono text-foreground">{formatTokens(outputTokens)}</span>
                {spans.some((s) => s.thoughts_tokens != null) && (
                  <>
                    {" · "}
                    {t("tokensThoughts")}{" "}
                    <span className="font-mono text-foreground">
                      {formatTokens(thoughtsTokens)}
                    </span>
                  </>
                )}
              </span>
            )}
            <span>
              {t("detailDuration")}{" "}
              <span className="font-mono text-foreground">{formatDurationMs(traceSpan)}</span>
            </span>
            {entry?.trace_id && <code className="font-mono">{entry.trace_id.slice(0, 8)}</code>}
          </div>
        </DialogHeader>
        {(createdMemoryIds.length > 0 || sourceMemoryIds.length > 0) && (
          <div className="shrink-0 flex flex-col gap-1 border-t border-border pt-2 text-xs">
            {createdMemoryIds.length > 0 && (
              <MemoryIdList
                label={t("memoriesCreated", { count: createdMemoryIds.length })}
                ids={createdMemoryIds}
              />
            )}
            {sourceMemoryIds.length > 0 && (
              <MemoryIdList
                label={t("memoriesSource", { count: sourceMemoryIds.length })}
                ids={sourceMemoryIds}
              />
            )}
          </div>
        )}
        <div className="flex-1 min-h-0 grid grid-cols-1 md:grid-cols-[minmax(240px,5fr)_7fr] gap-4">
          <div className="overflow-y-auto md:border-r border-border md:pr-3 space-y-0.5">
            {loading ? (
              <div className="py-4 flex flex-col items-center gap-2 text-sm text-muted-foreground">
                <Spinner size="md" />
                <span>{t("chartLoading")}</span>
              </div>
            ) : (
              spans.map((s) => (
                <SpanRow
                  key={s.id}
                  span={s}
                  traceStart={traceStart}
                  traceSpan={traceSpan}
                  selected={selected?.id === s.id}
                  onClick={() => setSelectedId(s.id)}
                  t={t}
                />
              ))
            )}
          </div>
          <div className="overflow-y-auto pr-1">
            {selected && <SpanDetails entry={selected} t={t} />}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

// ---- Chart Section ----

function LLMRequestChart({ bankId }: { bankId: string }) {
  const t = useTranslations("llmRequestsView");
  const operationOptions = getOperationOptions(t);
  const periodOptions = getPeriodOptions(t);
  const [period, setPeriod] = useState("7d");
  const [chartOperation, setChartOperation] = useState<string | null>(null);
  const [chartCallType, setChartCallType] = useState<LLMCallType | null>(null);
  const callTypeOptions = getCallTypeOptions(t);
  const [buckets, setBuckets] = useState<LLMRequestStatsBucket[]>([]);
  const [trunc, setTrunc] = useState("day");
  const [loading, setLoading] = useState(false);
  const [metric, setMetric] = useState<"calls" | "tokens">("calls");
  const [tokenMode, setTokenMode] = useState<"total" | "breakdown">("total");
  const [cumulative, setCumulative] = useState(false);

  const loadStats = useCallback(
    async (
      p: string = period,
      o: string | null = chartOperation,
      c: LLMCallType | null = chartCallType
    ) => {
      setLoading(true);
      try {
        const data = await client.getLLMRequestStats(bankId, {
          period: p,
          operation: o || undefined,
          call_type: c || undefined,
        });
        setBuckets(data.buckets || []);
        setTrunc(data.trunc || "day");
      } catch (error) {
        console.error("Error loading LLM request stats:", error);
      } finally {
        setLoading(false);
      }
    },
    [bankId, period, chartOperation, chartCallType]
  );

  useEffect(() => {
    loadStats();
  }, [bankId]);

  // Build per-bucket points, applying a running sum when cumulative is on.
  const running = { calls: 0, input: 0, output: 0, thoughts: 0, cached: 0, total: 0 };
  const chartData = buckets.map((b) => {
    const point = {
      calls: b.total,
      input: b.tokens?.input ?? 0,
      output: b.tokens?.output ?? 0,
      thoughts: b.tokens?.thoughts ?? 0,
      cached: b.tokens?.cached ?? 0,
      total: (b.tokens?.total ?? 0) + (b.tokens?.thoughts ?? 0),
    };
    if (cumulative) {
      running.calls += point.calls;
      running.input += point.input;
      running.output += point.output;
      running.thoughts += point.thoughts;
      running.cached += point.cached;
      running.total += point.total;
    }
    const v = cumulative ? running : point;
    return { time: formatChartLabel(b.time, trunc), ...v };
  });

  const tooltipStyle = {
    backgroundColor: "var(--popover)",
    border: "1px solid var(--border)",
    borderRadius: "6px",
    fontSize: "12px",
    padding: "4px 8px",
  };

  const renderChart = () => {
    if (metric === "tokens" && tokenMode === "breakdown") {
      // Unstacked lines, not the stacked areas this used to draw: cached is a
      // subset of input, so the stack height was never a meaningful total, and
      // adding reasoning to it would have compounded the double-count. The
      // single-series "total" chart below is where a real total belongs.
      return (
        <LineChart data={chartData} margin={{ top: 5, right: 5, bottom: 0, left: 5 }}>
          <XAxis dataKey="time" tick={{ fontSize: 10 }} axisLine={false} tickLine={false} />
          <YAxis
            tick={{ fontSize: 10 }}
            axisLine={false}
            tickLine={false}
            width={40}
            allowDecimals={false}
            tickFormatter={(v) => formatTokens(Number(v))}
          />
          <Tooltip contentStyle={tooltipStyle} />
          <Legend wrapperStyle={{ fontSize: "10px" }} />
          <Line
            type="monotone"
            dataKey="input"
            name={t("tokensInput")}
            stroke="var(--chart-1, #3b82f6)"
            strokeWidth={2}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="output"
            name={t("tokensOutput")}
            stroke="var(--chart-2, #22c55e)"
            strokeWidth={2}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="thoughts"
            name={t("tokensThoughts")}
            stroke="var(--chart-4, #a855f7)"
            strokeWidth={2}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="cached"
            name={t("tokensCached")}
            stroke="var(--chart-3, #f59e0b)"
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      );
    }
    const dataKey = metric === "calls" ? "calls" : "total";
    return (
      <LineChart data={chartData} margin={{ top: 5, right: 5, bottom: 0, left: 5 }}>
        <XAxis dataKey="time" tick={{ fontSize: 10 }} axisLine={false} tickLine={false} />
        <YAxis
          tick={{ fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          width={40}
          allowDecimals={false}
          tickFormatter={(v) => formatTokens(Number(v))}
        />
        <Tooltip contentStyle={tooltipStyle} />
        <Line
          type="monotone"
          dataKey={dataKey}
          name={metric === "calls" ? t("metricCalls") : t("tokensTotal")}
          stroke="var(--primary)"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 3 }}
        />
      </LineChart>
    );
  };

  const toggleBtn = (active: boolean): "default" | "outline" => (active ? "default" : "outline");

  return (
    <Card>
      <CardHeader className="pb-2 flex flex-row items-start justify-between space-y-0 gap-3">
        <CardTitle className="text-sm font-semibold pt-1.5">
          {metric === "calls" ? t("callVolume") : t("tokenVolume")}
        </CardTitle>
        <div className="flex gap-2 flex-wrap justify-end">
          {/* Metric: calls vs tokens */}
          <div className="flex gap-1">
            <Button
              variant={toggleBtn(metric === "calls")}
              size="sm"
              className="h-8 text-xs"
              onClick={() => setMetric("calls")}
            >
              {t("metricCalls")}
            </Button>
            <Button
              variant={toggleBtn(metric === "tokens")}
              size="sm"
              className="h-8 text-xs"
              onClick={() => setMetric("tokens")}
            >
              {t("metricTokens")}
            </Button>
          </div>

          {/* Token sub-mode: total vs breakdown */}
          {metric === "tokens" && (
            <div className="flex gap-1">
              <Button
                variant={toggleBtn(tokenMode === "total")}
                size="sm"
                className="h-8 text-xs"
                onClick={() => setTokenMode("total")}
              >
                {t("tokenModeTotal")}
              </Button>
              <Button
                variant={toggleBtn(tokenMode === "breakdown")}
                size="sm"
                className="h-8 text-xs"
                onClick={() => setTokenMode("breakdown")}
              >
                {t("tokenModeBreakdown")}
              </Button>
            </div>
          )}

          {/* Cumulative toggle */}
          <Button
            variant={toggleBtn(cumulative)}
            size="sm"
            className="h-8 text-xs"
            onClick={() => setCumulative((c) => !c)}
          >
            {t("cumulative")}
          </Button>

          <Select
            value={chartOperation || "all"}
            onValueChange={(v) => {
              const o = v === "all" ? null : v;
              setChartOperation(o);
              loadStats(period, o);
            }}
          >
            <SelectTrigger className="w-[150px] h-8 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent position="popper" className="max-h-[300px] overflow-y-auto">
              {operationOptions.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select
            value={chartCallType || "all"}
            onValueChange={(v) => {
              const c = v === "all" ? null : (v as LLMCallType);
              setChartCallType(c);
              loadStats(period, chartOperation, c);
            }}
          >
            <SelectTrigger className="w-[150px] h-8 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent position="popper" className="max-h-[300px] overflow-y-auto">
              {callTypeOptions.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {periodOptions.map((opt) => (
            <Button
              key={opt.value}
              variant={toggleBtn(period === opt.value)}
              size="sm"
              className="h-8 text-xs"
              onClick={() => {
                setPeriod(opt.value);
                loadStats(opt.value, chartOperation);
              }}
            >
              {opt.label}
            </Button>
          ))}
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-[140px]">
          {loading ? (
            <div className="flex flex-col items-center justify-center gap-2 h-full text-muted-foreground text-sm">
              <Spinner size="md" />
              <span>{t("chartLoading")}</span>
            </div>
          ) : chartData.length === 0 ? (
            <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
              {t("chartNoData")}
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              {renderChart()}
            </ResponsiveContainer>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

// ---- Main Component ----

export function LLMRequestsView() {
  const t = useTranslations("llmRequestsView");
  const statusOptions = getStatusOptions(t);
  const operationOptions = getOperationOptions(t);
  const callTypeOptions = getCallTypeOptions(t);
  const { currentBank } = useBank();
  const [requests, setRequests] = useState<LLMRequestEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [statusFilter, setStatusFilter] = useState<string | null>(null);
  const [operationFilter, setOperationFilter] = useState<string | null>(null);
  const [callTypeFilter, setCallTypeFilter] = useState<LLMCallType | null>(null);
  const [dateRange, setDateRange] = useState<string>("all");
  // Group LLM calls by operation run (trace) vs. a flat per-call listing.
  const [grouped, setGrouped] = useState(true);
  const [limit] = useState(20);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<LLMRequestEntry | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);

  const loadRequests = useCallback(
    async (
      newStatusFilter: string | null = statusFilter,
      newOperationFilter: string | null = operationFilter,
      newDateRange: string = dateRange,
      newOffset: number = offset,
      newGrouped: boolean = grouped,
      newCallType: LLMCallType | null = callTypeFilter
    ) => {
      if (!currentBank) return;

      setLoading(true);
      try {
        const dates = resolveDateRangePreset(newDateRange);
        const data = await client.listLLMRequests(currentBank, {
          status: newStatusFilter || undefined,
          operation: newOperationFilter || undefined,
          call_type: newCallType || undefined,
          group: newGrouped || undefined,
          start_date: dates.start_date,
          end_date: dates.end_date,
          limit,
          offset: newOffset,
        });
        setRequests(data.items || []);
        setTotal(data.total || 0);
      } catch (error) {
        console.error("Error loading LLM requests:", error);
      } finally {
        setLoading(false);
      }
    },
    [currentBank, statusFilter, operationFilter, callTypeFilter, dateRange, offset, grouped, limit]
  );

  const handleGroupToggle = () => {
    const next = !grouped;
    setGrouped(next);
    setOffset(0);
    loadRequests(statusFilter, operationFilter, dateRange, 0, next);
  };

  const handleStatusFilterChange = (value: string) => {
    const filter = value === "all" ? null : value;
    setStatusFilter(filter);
    setOffset(0);
    loadRequests(filter, operationFilter, dateRange, 0);
  };

  const handleOperationFilterChange = (value: string) => {
    const filter = value === "all" ? null : value;
    setOperationFilter(filter);
    setOffset(0);
    loadRequests(statusFilter, filter, dateRange, 0);
  };

  const handleCallTypeFilterChange = (value: string) => {
    const filter = value === "all" ? null : (value as LLMCallType);
    setCallTypeFilter(filter);
    setOffset(0);
    loadRequests(statusFilter, operationFilter, dateRange, 0, grouped, filter);
  };

  const handleDateRangeChange = (value: string) => {
    setDateRange(value);
    setOffset(0);
    loadRequests(statusFilter, operationFilter, value, 0);
  };

  const handlePageChange = (newOffset: number) => {
    setOffset(newOffset);
    loadRequests(statusFilter, operationFilter, dateRange, newOffset);
  };

  const handleRowClick = (entry: LLMRequestEntry) => {
    setSelected(entry);
    setDialogOpen(true);
  };

  const renderPlainRow = (entry: LLMRequestEntry) => (
    <TableRow
      key={entry.id}
      className="cursor-pointer hover:bg-muted/50"
      onClick={() => handleRowClick(entry)}
    >
      <TableCell className="text-sm font-mono">{formatDateTime(entry.started_at)}</TableCell>
      <TableCell className="font-medium">{entry.operation || "—"}</TableCell>
      <TableCell className="text-sm text-muted-foreground font-mono">
        <ScopeBadge scope={entry.scope} t={t} />
      </TableCell>
      <TableCell>
        <StatusBadge status={entry.status} />
      </TableCell>
      <TableCell className="text-sm text-muted-foreground font-mono text-right">
        {totalTokens(entry)?.toLocaleString() ?? "—"}
      </TableCell>
      <TableCell className="text-sm text-muted-foreground font-mono">
        {formatDurationMs(entry.duration_ms)}
      </TableCell>
    </TableRow>
  );

  useEffect(() => {
    if (currentBank) {
      loadRequests(statusFilter, operationFilter, dateRange, offset);
    }
  }, [currentBank]);

  const totalPages = Math.ceil(total / limit);
  const currentPage = Math.floor(offset / limit) + 1;

  if (!currentBank) return null;

  return (
    <div className="space-y-6">
      {/* Chart */}
      <LLMRequestChart bankId={currentBank} />

      {/* Filters */}
      <div className="flex items-center gap-3 flex-wrap">
        <Select value={statusFilter || "all"} onValueChange={handleStatusFilterChange}>
          <SelectTrigger className="w-[160px]">
            <SelectValue placeholder={t("statusAll")} />
          </SelectTrigger>
          <SelectContent position="popper" className="max-h-[300px] overflow-y-auto">
            {statusOptions.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select value={operationFilter || "all"} onValueChange={handleOperationFilterChange}>
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder={t("operationAll")} />
          </SelectTrigger>
          <SelectContent position="popper" className="max-h-[300px] overflow-y-auto">
            {operationOptions.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select value={callTypeFilter || "all"} onValueChange={handleCallTypeFilterChange}>
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder={t("callTypeAll")} />
          </SelectTrigger>
          <SelectContent position="popper" className="max-h-[300px] overflow-y-auto">
            {callTypeOptions.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select value={dateRange} onValueChange={handleDateRangeChange}>
          <SelectTrigger className="w-[150px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent position="popper">
            <SelectItem value="all">{t("dateRangeAll")}</SelectItem>
            <SelectItem value="1h">{t("dateRangeLastHour")}</SelectItem>
            <SelectItem value="1d">{t("dateRangeLast24Hours")}</SelectItem>
            <SelectItem value="7d">{t("dateRangeLast7Days")}</SelectItem>
            <SelectItem value="30d">{t("dateRangeLast30Days")}</SelectItem>
          </SelectContent>
        </Select>

        <Button
          variant={grouped ? "default" : "outline"}
          size="sm"
          onClick={handleGroupToggle}
          title={t("groupByRunHint")}
        >
          {t("groupByRun")}
        </Button>

        <Button
          variant="outline"
          size="sm"
          onClick={() => loadRequests(statusFilter, operationFilter, dateRange, offset)}
          disabled={loading}
        >
          <RefreshCw className={`w-4 h-4 mr-1 ${loading ? "animate-spin" : ""}`} />
          {t("refresh")}
        </Button>

        <span className="text-sm text-muted-foreground ml-auto">
          {t("entryCount", { count: total })}
        </span>
      </div>

      {/* Table */}
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[190px]">{t("tableHeaderTime")}</TableHead>
            <TableHead className="w-[130px]">{t("tableHeaderOperation")}</TableHead>
            <TableHead>{t("tableHeaderScope")}</TableHead>
            <TableHead className="w-[100px]">{t("tableHeaderStatus")}</TableHead>
            <TableHead className="w-[110px] text-right">{t("tableHeaderTokens")}</TableHead>
            <TableHead className="w-[100px]">{t("tableHeaderDuration")}</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {requests.length === 0 ? (
            <TableRow>
              <TableCell colSpan={6} className="text-center text-muted-foreground py-8">
                {loading ? t("tableLoading") : t("tableNoRequests")}
              </TableCell>
            </TableRow>
          ) : !grouped ? (
            requests.map(renderPlainRow)
          ) : (
            groupByTrace(requests).map((group) => {
              // Single, untraced, or one-call run → a plain clickable row.
              if (!group.grouped) {
                return renderPlainRow(group.rows[0]);
              }

              // Multi-call operation run → one summary row; click opens the
              // trace dialog (parent run → child-call waterfall + details).
              const agg = aggregateGroup(group.rows);
              return (
                <TableRow
                  key={group.key}
                  className="cursor-pointer hover:bg-muted/50"
                  onClick={() => handleRowClick(group.rows[0])}
                >
                  <TableCell className="text-sm font-mono">
                    {formatDateTime(agg.startIso)}
                  </TableCell>
                  <TableCell className="font-medium">{group.rows[0].operation || "—"}</TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    <KindBreakdown rows={group.rows} t={t} />
                  </TableCell>
                  <TableCell>
                    <StatusBadge status={agg.status} />
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground font-mono text-right">
                    {agg.tokens.toLocaleString()}
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground font-mono">
                    {formatDurationMs(agg.durationMs)}
                  </TableCell>
                </TableRow>
              );
            })
          )}
        </TableBody>
      </Table>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">
            {t("paginationPage", { current: currentPage, total: totalPages })}
          </span>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => handlePageChange(Math.max(0, offset - limit))}
              disabled={offset === 0}
            >
              <ChevronLeft className="w-4 h-4 mr-1" />
              {t("previous")}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => handlePageChange(offset + limit)}
              disabled={offset + limit >= total}
            >
              {t("next")}
              <ChevronRight className="w-4 h-4 ml-1" />
            </Button>
          </div>
        </div>
      )}

      {/* Trace dialog: parent operation run → child-call waterfall + details */}
      <TraceDialog
        bankId={currentBank}
        entry={selected}
        open={dialogOpen}
        onOpenChange={setDialogOpen}
      />
    </div>
  );
}

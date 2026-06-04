"use client";

import { useState, useEffect, useCallback } from "react";
import { useTranslations } from "next-intl";
import { useBank } from "@/lib/bank-context";
import { client, LLMRequestEntry, LLMRequestStatsBucket } from "@/lib/api";
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
import { RefreshCw, ChevronLeft, ChevronRight } from "lucide-react";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

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
    { value: "reflect", label: t("operationReflect") },
    { value: "consolidation", label: t("operationConsolidation") },
    { value: "refresh_mental_model", label: t("operationRefreshMentalModel") },
  ];
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
  const styles: Record<string, string> = {
    success: "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300",
    error: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300",
  };
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${styles[status] || styles.success}`}
    >
      {status}
    </span>
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
  const tokens = rows.reduce((sum, r) => sum + (r.total_tokens ?? 0), 0);
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
}: {
  span: LLMRequestEntry;
  traceStart: number;
  traceSpan: number;
  selected: boolean;
  onClick: () => void;
}) {
  const start = span.started_at ? new Date(span.started_at).getTime() : traceStart;
  const end = span.ended_at ? new Date(span.ended_at).getTime() : start;
  const left = Math.min(98, Math.max(0, ((start - traceStart) / traceSpan) * 100));
  const width = Math.max(2, Math.min(100 - left, ((end - start) / traceSpan) * 100));
  const isError = span.status === "error";
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
          <span className="text-xs font-mono truncate">
            {span.scope || span.operation || "call"}
          </span>
        </span>
        <span className="text-[10px] text-muted-foreground font-mono shrink-0">
          {formatDurationMs(span.duration_ms)}
          {span.total_tokens != null ? ` · ${formatTokens(span.total_tokens)}` : ""}
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

      {(entry.total_tokens !== null ||
        entry.input_tokens !== null ||
        entry.output_tokens !== null) && (
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
            <span className="text-muted-foreground">{t("tokensCached")}:</span>{" "}
            <span className="font-mono">{entry.cached_tokens ?? "—"}</span>
          </div>
          <div>
            <span className="text-muted-foreground">{t("tokensTotal")}:</span>{" "}
            <span className="font-mono font-medium">{entry.total_tokens ?? "—"}</span>
          </div>
        </div>
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
      {entry.llm_info && Object.keys(entry.llm_info).length > 0 && (
        <div>
          <h4 className="text-sm font-semibold mb-2">{t("detailLlmInfo")}</h4>
          <JsonViewer value={entry.llm_info} />
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
  const inputTokens = spans.reduce((sum, r) => sum + (r.input_tokens ?? 0), 0);
  const outputTokens = spans.reduce((sum, r) => sum + (r.output_tokens ?? 0), 0);
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
            <span>{t("entryCount", { count: spans.length })}</span>
            <span>
              {t("tokensInput")}{" "}
              <span className="font-mono text-foreground">{formatTokens(inputTokens)}</span>
              {" · "}
              {t("tokensOutput")}{" "}
              <span className="font-mono text-foreground">{formatTokens(outputTokens)}</span>
            </span>
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
              <div className="text-sm text-muted-foreground py-4">{t("chartLoading")}</div>
            ) : (
              spans.map((s) => (
                <SpanRow
                  key={s.id}
                  span={s}
                  traceStart={traceStart}
                  traceSpan={traceSpan}
                  selected={selected?.id === s.id}
                  onClick={() => setSelectedId(s.id)}
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
  const [buckets, setBuckets] = useState<LLMRequestStatsBucket[]>([]);
  const [trunc, setTrunc] = useState("day");
  const [loading, setLoading] = useState(false);
  const [metric, setMetric] = useState<"calls" | "tokens">("calls");
  const [tokenMode, setTokenMode] = useState<"total" | "breakdown">("total");
  const [cumulative, setCumulative] = useState(false);

  const loadStats = useCallback(
    async (p: string = period, o: string | null = chartOperation) => {
      setLoading(true);
      try {
        const data = await client.getLLMRequestStats(bankId, {
          period: p,
          operation: o || undefined,
        });
        setBuckets(data.buckets || []);
        setTrunc(data.trunc || "day");
      } catch (error) {
        console.error("Error loading LLM request stats:", error);
      } finally {
        setLoading(false);
      }
    },
    [bankId, period, chartOperation]
  );

  useEffect(() => {
    loadStats();
  }, [bankId]);

  // Build per-bucket points, applying a running sum when cumulative is on.
  const running = { calls: 0, input: 0, output: 0, cached: 0, total: 0 };
  const chartData = buckets.map((b) => {
    const point = {
      calls: b.total,
      input: b.tokens?.input ?? 0,
      output: b.tokens?.output ?? 0,
      cached: b.tokens?.cached ?? 0,
      total: b.tokens?.total ?? 0,
    };
    if (cumulative) {
      running.calls += point.calls;
      running.input += point.input;
      running.output += point.output;
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
      return (
        <AreaChart data={chartData} margin={{ top: 5, right: 5, bottom: 0, left: 5 }}>
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
          <Area
            type="monotone"
            dataKey="input"
            name={t("tokensInput")}
            stackId="t"
            stroke="var(--chart-1, #3b82f6)"
            fill="var(--chart-1, #3b82f6)"
            fillOpacity={0.5}
          />
          <Area
            type="monotone"
            dataKey="output"
            name={t("tokensOutput")}
            stackId="t"
            stroke="var(--chart-2, #22c55e)"
            fill="var(--chart-2, #22c55e)"
            fillOpacity={0.5}
          />
          <Area
            type="monotone"
            dataKey="cached"
            name={t("tokensCached")}
            stackId="t"
            stroke="var(--chart-3, #f59e0b)"
            fill="var(--chart-3, #f59e0b)"
            fillOpacity={0.5}
          />
        </AreaChart>
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
            <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
              {t("chartLoading")}
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
  const { currentBank } = useBank();
  const [requests, setRequests] = useState<LLMRequestEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [statusFilter, setStatusFilter] = useState<string | null>(null);
  const [operationFilter, setOperationFilter] = useState<string | null>(null);
  const [dateRange, setDateRange] = useState<string>("all");
  // Group LLM calls by operation run (trace) vs. a flat per-call listing.
  const [grouped, setGrouped] = useState(true);
  const [limit] = useState(20);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<LLMRequestEntry | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);

  const getDateRange = useCallback((range: string): { start_date?: string; end_date?: string } => {
    if (range === "all") return {};
    const now = new Date();
    const start = new Date();
    if (range === "1h") start.setHours(now.getHours() - 1);
    else if (range === "1d") start.setDate(now.getDate() - 1);
    else if (range === "7d") start.setDate(now.getDate() - 7);
    else if (range === "30d") start.setDate(now.getDate() - 30);
    return { start_date: start.toISOString() };
  }, []);

  const loadRequests = useCallback(
    async (
      newStatusFilter: string | null = statusFilter,
      newOperationFilter: string | null = operationFilter,
      newDateRange: string = dateRange,
      newOffset: number = offset,
      newGrouped: boolean = grouped
    ) => {
      if (!currentBank) return;

      setLoading(true);
      try {
        const dates = getDateRange(newDateRange);
        const data = await client.listLLMRequests(currentBank, {
          status: newStatusFilter || undefined,
          operation: newOperationFilter || undefined,
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
    [currentBank, statusFilter, operationFilter, dateRange, offset, grouped, limit, getDateRange]
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
        {entry.scope || "—"}
      </TableCell>
      <TableCell>
        <StatusBadge status={entry.status} />
      </TableCell>
      <TableCell className="text-sm text-muted-foreground font-mono text-right">
        {entry.total_tokens != null ? entry.total_tokens.toLocaleString() : "—"}
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
                    {t("entryCount", { count: group.rows.length })}
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

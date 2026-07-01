"use client";

import { useState, useEffect, useMemo } from "react";
import { useTranslations } from "next-intl";
import { OperatorShell } from "@/components/operator-shell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Activity,
  Loader2,
  Search,
  RefreshCw,
  BarChart3,
  Clock,
  DollarSign,
  Zap,
  AlertCircle,
  CheckCircle2,
  ExternalLink,
  Filter,
  ChevronDown,
  ChevronRight,
  Layers,
} from "lucide-react";
import { Input } from "@/components/ui/input";

// ── Types ─────────────────────────────────────────────────────────

interface LangfuseTrace {
  id: string;
  name: string;
  userId: string | null;
  sessionId: string | null;
  timestamp: string;
  tags: string[];
  input?: unknown;
  output?: unknown;
  metadata?: Record<string, unknown>;
  public: boolean;
}

interface TraceObservation {
  id: string;
  traceId: string;
  name: string;
  type: string;
  startTime: string;
  endTime: string | null;
  input: unknown;
  output: unknown;
  level: string;
  statusMessage: string | null;
  parentObservationId: string | null;
  metadata?: Record<string, unknown>;
  model: string | null;
  modelParameters: Record<string, unknown> | null;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  unit: string | null;
  inputPrice: number | null;
  outputPrice: number | null;
  totalPrice: number | null;
  latency: number | null;
  cost: number | null;
}

interface TracesResponse {
  data: LangfuseTrace[];
  meta: {
    page: number;
    limit: number;
    totalItems: number;
    totalPages: number;
  };
}

// ── Helpers ────────────────────────────────────────────────────────

function formatDuration(ms: number | null | undefined): string {
  if (ms == null) return "—";
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

function formatCost(cost: number | null | undefined): string {
  if (cost == null || cost === 0) return "—";
  if (cost < 0.001) return `${(cost * 1000000).toFixed(0)}µ$`;
  if (cost < 1) return `${(cost * 1000).toFixed(2)}m$`;
  return `$${cost.toFixed(4)}`;
}

function formatTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

const statusColor = (level: string) => {
  switch (level) {
    case "DEFAULT":
    case "INFO":
      return "text-green-500";
    case "WARNING":
      return "text-amber-500";
    case "ERROR":
      return "text-red-500";
    default:
      return "text-muted-foreground";
  }
};

// ── Trace Detail Modal ──────────────────────────────────────────────

function TraceDetail({ trace, onClose }: { trace: LangfuseTrace; onClose: () => void }) {
  const [observations, setObservations] = useState<TraceObservation[]>([]);
  const [obsLoading, setObsLoading] = useState(false);
  const [expandedObs, setExpandedObs] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      setObsLoading(true);
      try {
        const res = await fetch(`/api/langfuse/traces/observations?traceId=${trace.id}`);
        if (res.ok) {
          const data = await res.json();
          setObservations(data.data || []);
        }
      } catch {
        // ignore
      }
      setObsLoading(false);
    };
    load();
  }, [trace.id]);

  const totalTokens = observations.reduce((s, o) => s + (o.totalTokens || 0), 0);
  const totalCost = observations.reduce((s, o) => s + (o.totalPrice || 0), 0);
  const totalLatency = observations.reduce((s, o) => s + (o.latency || 0), 0);

  return (
    <div className="fixed inset-0 z-50 bg-black/50 flex items-start justify-center pt-12">
      <div className="bg-background rounded-lg shadow-xl w-full max-w-3xl max-h-[80vh] overflow-y-auto mx-4">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b sticky top-0 bg-background z-10">
          <div>
            <h2 className="font-semibold text-sm">{trace.name || "Unnamed Trace"}</h2>
            <p className="text-[10px] font-mono text-muted-foreground">{trace.id}</p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-[10px]">
              {formatDate(trace.timestamp)} {formatTime(trace.timestamp)}
            </Badge>
            <Button variant="ghost" size="sm" onClick={onClose} className="h-7 w-7 p-0">
              ✕
            </Button>
          </div>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-3 gap-3 p-4 border-b">
          <div className="text-center">
            <p className="text-[10px] text-muted-foreground">Total Tokens</p>
            <p className="text-lg font-semibold font-mono">{totalTokens.toLocaleString()}</p>
          </div>
          <div className="text-center">
            <p className="text-[10px] text-muted-foreground">Cost</p>
            <p className="text-lg font-semibold font-mono">{formatCost(totalCost)}</p>
          </div>
          <div className="text-center">
            <p className="text-[10px] text-muted-foreground">Latency</p>
            <p className="text-lg font-semibold font-mono">{formatDuration(totalLatency)}</p>
          </div>
        </div>

        {/* Observations / Spans */}
        <div className="p-4">
          <h3 className="text-xs font-semibold mb-3 flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5" />
            Spans ({observations.length})
          </h3>
          {obsLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-4 w-4 animate-spin" />
            </div>
          ) : observations.length === 0 ? (
            <div className="text-xs text-muted-foreground text-center py-4">No spans recorded</div>
          ) : (
            <div className="space-y-1.5">
              {observations.map((obs) => (
                <div key={obs.id} className="border rounded-md">
                  <button
                    onClick={() => setExpandedObs(expandedObs === obs.id ? null : obs.id)}
                    className="w-full flex items-center justify-between p-2.5 hover:bg-accent/30 text-left"
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      {expandedObs === obs.id ? (
                        <ChevronDown className="w-3 h-3 shrink-0 text-muted-foreground" />
                      ) : (
                        <ChevronRight className="w-3 h-3 shrink-0 text-muted-foreground" />
                      )}
                      <span className={statusColor(obs.level)}>
                        {obs.level === "ERROR" ? "●" : "○"}
                      </span>
                      <span className="text-xs font-medium truncate">{obs.name || obs.type}</span>
                      <Badge variant="outline" className="text-[9px] h-4 px-1">
                        {obs.type}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-3 text-[10px] text-muted-foreground shrink-0">
                      {obs.totalTokens > 0 && <span>{obs.totalTokens}t</span>}
                      {obs.latency != null && <span>{formatDuration(obs.latency)}</span>}
                      {obs.model && (
                        <span className="font-mono max-w-[100px] truncate">{obs.model}</span>
                      )}
                    </div>
                  </button>
                  {expandedObs === obs.id && (
                    <div className="px-2.5 pb-2.5 space-y-2 border-t pt-2">
                      {obs.model && (
                        <div className="text-[10px]">
                          <span className="text-muted-foreground">Model: </span>
                          <span className="font-mono">{obs.model}</span>
                        </div>
                      )}
                      {obs.modelParameters && Object.keys(obs.modelParameters).length > 0 && (
                        <div className="text-[10px]">
                          <span className="text-muted-foreground">Params: </span>
                          <span className="font-mono">{JSON.stringify(obs.modelParameters)}</span>
                        </div>
                      )}
                      <div className="grid grid-cols-3 gap-2">
                        <div className="bg-muted/30 rounded p-1.5 text-center">
                          <p className="text-[9px] text-muted-foreground">Prompt</p>
                          <p className="text-xs font-mono">{obs.promptTokens || 0}</p>
                        </div>
                        <div className="bg-muted/30 rounded p-1.5 text-center">
                          <p className="text-[9px] text-muted-foreground">Completion</p>
                          <p className="text-xs font-mono">{obs.completionTokens || 0}</p>
                        </div>
                        <div className="bg-muted/30 rounded p-1.5 text-center">
                          <p className="text-[9px] text-muted-foreground">Cost</p>
                          <p className="text-xs font-mono">{formatCost(obs.totalPrice)}</p>
                        </div>
                      </div>
                      {obs.statusMessage && (
                        <div className="text-[10px] text-red-500 bg-red-50/50 dark:bg-red-950/20 rounded p-1.5">
                          {obs.statusMessage}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Trace Item ──────────────────────────────────────────────────────

function TraceItem({ trace, onClick }: { trace: LangfuseTrace; onClick: () => void }) {
  return (
    <div
      className="flex items-center justify-between p-3 rounded-lg border hover:bg-accent/30 cursor-pointer transition-colors"
      onClick={onClick}
    >
      <div className="flex items-center gap-3 min-w-0">
        <Activity className="w-4 h-4 shrink-0 text-primary" />
        <div className="min-w-0">
          <p className="text-sm font-medium truncate">{trace.name || "Unnamed Trace"}</p>
          <p className="text-[10px] font-mono text-muted-foreground truncate">
            {trace.id.substring(0, 12)}...
            {trace.userId && <span> · user: {trace.userId}</span>}
            {trace.sessionId && <span> · session: {trace.sessionId}</span>}
          </p>
        </div>
      </div>
      <div className="flex items-center gap-2 shrink-0">
        {trace.tags.length > 0 && (
          <div className="hidden sm:flex gap-1">
            {trace.tags.slice(0, 2).map((tag) => (
              <Badge key={tag} variant="secondary" className="text-[9px] h-4 px-1">
                {tag}
              </Badge>
            ))}
          </div>
        )}
        <span className="text-[10px] text-muted-foreground">
          {formatDate(trace.timestamp)} {formatTime(trace.timestamp)}
        </span>
        <ChevronRight className="w-3.5 h-3.5 text-muted-foreground" />
      </div>
    </div>
  );
}

// ── Main Page ───────────────────────────────────────────────────────

export default function TracesPage() {
  const t = useTranslations("operator");
  const [traces, setTraces] = useState<LangfuseTrace[]>([]);
  const [loading, setLoading] = useState(true);
  const [meta, setMeta] = useState({ page: 1, limit: 20, totalItems: 0, totalPages: 0 });
  const [search, setSearch] = useState("");
  const [selectedTrace, setSelectedTrace] = useState<LangfuseTrace | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchTraces = async (page = 1) => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({ limit: "20", page: String(page) });
      if (search) params.set("userId", search);

      const res = await fetch(`/api/langfuse/traces?${params}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: TracesResponse = await res.json();
      setTraces(data.data);
      setMeta(data.meta);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setTraces([]);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchTraces();
  }, []);

  const stats = useMemo(() => {
    if (traces.length === 0) return null;
    return {
      total: meta.totalItems,
      page: meta.page,
      pages: meta.totalPages,
    };
  }, [traces, meta]);

  return (
    <OperatorShell>
      <div className="p-6 space-y-6">
        {/* Page Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">LLM Traces</h1>
            <p className="text-sm text-muted-foreground">
              {t("descriptions.traces") || "LLM trace observability from Langfuse"}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {stats && (
              <Badge variant="outline" className="text-[10px] font-mono">
                {stats.total} traces
              </Badge>
            )}
            <Button
              variant="outline"
              size="sm"
              onClick={() => fetchTraces(meta.page)}
              disabled={loading}
            >
              <RefreshCw className={`h-4 w-4 mr-1 ${loading ? "animate-spin" : ""}`} />
              Refresh
            </Button>
          </div>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Card>
            <CardHeader className="pb-1 pt-3 px-4">
              <CardTitle className="flex items-center gap-1.5 text-[10px] font-medium text-muted-foreground">
                <BarChart3 className="w-3 h-3" />
                Total Traces
              </CardTitle>
            </CardHeader>
            <CardContent className="px-4 pb-3">
              <p className="text-xl font-bold font-mono">
                {loading ? "..." : meta.totalItems.toLocaleString()}
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-1 pt-3 px-4">
              <CardTitle className="flex items-center gap-1.5 text-[10px] font-medium text-muted-foreground">
                <Zap className="w-3 h-3" />
                This Page
              </CardTitle>
            </CardHeader>
            <CardContent className="px-4 pb-3">
              <p className="text-xl font-bold font-mono">{traces.length}</p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-1 pt-3 px-4">
              <CardTitle className="flex items-center gap-1.5 text-[10px] font-medium text-muted-foreground">
                <Clock className="w-3 h-3" />
                Page
              </CardTitle>
            </CardHeader>
            <CardContent className="px-4 pb-3">
              <p className="text-xl font-bold font-mono">
                {meta.page} / {meta.totalPages || 1}
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-1 pt-3 px-4">
              <CardTitle className="flex items-center gap-1.5 text-[10px] font-medium text-muted-foreground">
                <DollarSign className="w-3 h-3" />
                Status
              </CardTitle>
            </CardHeader>
            <CardContent className="px-4 pb-3">
              <p className="text-base font-semibold flex items-center gap-1.5">
                {error ? (
                  <>
                    <AlertCircle className="w-4 h-4 text-red-500" />
                    Error
                  </>
                ) : (
                  <>
                    <CheckCircle2 className="w-4 h-4 text-green-500" />
                    Connected
                  </>
                )}
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Filter Bar */}
        <div className="flex items-center gap-2">
          <div className="relative flex-1 max-w-xs">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
            <Input
              placeholder="Filter by user ID..."
              className="pl-8 h-8 text-xs"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && fetchTraces(1)}
            />
          </div>
          <Button
            variant="outline"
            size="sm"
            className="h-8 text-xs"
            onClick={() => fetchTraces(1)}
          >
            <Filter className="w-3 h-3 mr-1" />
            Filter
          </Button>
        </div>

        {/* Trace List */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Activity className="h-4 w-4" />
              Trace Events
            </CardTitle>
            <CardDescription>
              {loading
                ? "Loading traces..."
                : error
                  ? `Error: ${error}`
                  : traces.length === 0
                    ? "No traces recorded yet — traces appear when LLM calls are made"
                    : `${meta.totalItems} trace(s) found`}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex items-center justify-center py-12 text-muted-foreground">
                <Loader2 className="h-5 w-5 animate-spin mr-2" />
                Loading traces...
              </div>
            ) : traces.length === 0 ? (
              <div className="text-center py-12 space-y-3">
                <Activity className="w-12 h-12 mx-auto text-muted-foreground/40" />
                <div>
                  <p className="text-sm font-medium">No traces yet</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    {error
                      ? `Could not connect to Langfuse: ${error}`
                      : "Traces will appear here once LLM calls are recorded through Langfuse."}
                  </p>
                </div>
                <Button variant="outline" size="sm" onClick={() => fetchTraces(1)}>
                  <RefreshCw className="h-3 w-3 mr-1" />
                  Retry
                </Button>
              </div>
            ) : (
              <div className="space-y-2">
                {traces.map((trace) => (
                  <TraceItem key={trace.id} trace={trace} onClick={() => setSelectedTrace(trace)} />
                ))}
              </div>
            )}

            {/* Pagination */}
            {meta.totalPages > 1 && (
              <div className="flex items-center justify-center gap-2 mt-4 pt-3 border-t">
                <Button
                  variant="outline"
                  size="sm"
                  className="text-xs h-7"
                  disabled={meta.page <= 1 || loading}
                  onClick={() => fetchTraces(meta.page - 1)}
                >
                  Previous
                </Button>
                <span className="text-[10px] text-muted-foreground font-mono">
                  {meta.page} / {meta.totalPages}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  className="text-xs h-7"
                  disabled={meta.page >= meta.totalPages || loading}
                  onClick={() => fetchTraces(meta.page + 1)}
                >
                  Next
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Trace Detail Modal */}
        {selectedTrace && (
          <TraceDetail trace={selectedTrace} onClose={() => setSelectedTrace(null)} />
        )}
      </div>
    </OperatorShell>
  );
}

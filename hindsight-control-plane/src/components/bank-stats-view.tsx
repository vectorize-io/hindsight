"use client";

import { useState, useEffect } from "react";
import { useBank } from "@/lib/bank-context";
import { useFeatures } from "@/lib/features-context";
import { client, MentalModel } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Database,
  Link2,
  FolderOpen,
  Activity,
  Clock,
  Brain,
  CheckCircle2,
  AlertCircle,
  XCircle,
  RefreshCw,
  ExternalLink,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import type { TooltipContentProps } from "recharts";

interface BankStats {
  bank_id: string;
  total_nodes: number;
  total_links: number;
  total_documents: number;
  nodes_by_fact_type: {
    world?: number;
    experience?: number;
    opinion?: number;
  };
  links_by_link_type: {
    temporal?: number;
    semantic?: number;
    entity?: number;
  };
  pending_operations: number;
  failed_operations: number;
  operations_by_status?: Record<string, number>;
  last_consolidated_at: string | null;
  pending_consolidation: number;
  failed_consolidation: number;
  total_observations: number;
}

type Period = "1h" | "12h" | "1d" | "7d" | "30d" | "90d";
const PERIODS: Period[] = ["1h", "12h", "1d", "7d", "30d", "90d"];

interface TimeseriesBucket {
  time: string;
  world: number;
  experience: number;
  observation: number;
  // formatted client-side
  label?: string;
}

type FactKey = "world" | "experience" | "observation";

// Palette. Memory-type hues deliberately avoid the link-type hues (teal/blue/amber)
// so the two distributions read as different dimensions.
const CHART_COLORS = {
  primary: "var(--primary)",
  // Memory composition — modern violet / pink / indigo palette.
  // Picked to stay distinct from the link-type hues (teal/blue/amber).
  world: "#8b5cf6", // violet-500
  experience: "#ec4899", // pink-500
  observation: "#6366f1", // indigo-500
  // Link types — match the graph view exactly (data-view.tsx:215)
  temporal: "#009296", // brand teal
  semantic: "#0074d9", // brand blue
  entity: "#f59e0b", // brand amber
  // Status
  success: "var(--chart-5)",
  warning: "var(--chart-4)",
  danger: "var(--destructive)",
  muted: "var(--muted)",
  mutedFg: "var(--muted-foreground)",
  border: "var(--border)",
};

const FACT_META: Record<FactKey, { label: string; color: string }> = {
  world: { label: "World", color: CHART_COLORS.world },
  experience: { label: "Experience", color: CHART_COLORS.experience },
  observation: { label: "Observations", color: CHART_COLORS.observation },
};

function formatCompact(n: number): string {
  if (n < 1000) return n.toString();
  if (n < 1_000_000) {
    const k = n / 1000;
    return `${k >= 10 ? k.toFixed(0) : k.toFixed(1).replace(/\.0$/, "")}k`;
  }
  if (n < 1_000_000_000) {
    const m = n / 1_000_000;
    return `${m >= 10 ? m.toFixed(0) : m.toFixed(1).replace(/\.0$/, "")}M`;
  }
  const b = n / 1_000_000_000;
  return `${b >= 10 ? b.toFixed(0) : b.toFixed(1).replace(/\.0$/, "")}B`;
}

function CompactNumber({ value, className }: { value: number; className?: string }) {
  return (
    <span className={className} title={value.toLocaleString()}>
      {formatCompact(value)}
    </span>
  );
}

// Custom tooltip — clean shadow card, no harsh borders, tabular numbers.
// Recharts' content prop can be rendered without all the normally-required
// TooltipContentProps fields populated, so we make them Partial here.
type ChartTooltipProps = Partial<TooltipContentProps<number, string>> & {
  valueLabel?: string;
};

function ChartTooltip({ active, payload, label, valueLabel }: ChartTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="rounded-lg border border-border/60 bg-popover/95 backdrop-blur-sm px-3 py-2 shadow-md">
      {label != null && (
        <div className="text-[11px] font-medium text-foreground mb-1.5">{label}</div>
      )}
      <div className="space-y-1">
        {payload.map((p, i) => (
          <div key={i} className="flex items-center gap-2 text-[11px]">
            <span
              className="w-2 h-2 rounded-[2px]"
              style={{ backgroundColor: p.color || (p.payload as { fill?: string })?.fill }}
            />
            <span className="text-muted-foreground">{p.name || valueLabel || "Value"}</span>
            <span className="ml-auto pl-3 font-semibold tabular-nums text-foreground">
              {(p.value ?? 0).toLocaleString()}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function HeroCard({
  icon: Icon,
  label,
  value,
  pulse,
}: {
  icon: typeof Database;
  label: string;
  value: number;
  pulse?: boolean;
}) {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-md bg-muted">
            <Icon
              className={`w-4 h-4 ${pulse ? "animate-pulse text-amber-500" : "text-muted-foreground"}`}
            />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-xs text-muted-foreground font-medium">{label}</p>
            <CompactNumber
              value={value}
              className="text-2xl font-semibold text-foreground leading-tight tabular-nums block"
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-[0.08em] mb-3">
      {children}
    </h3>
  );
}

function formatRelativeTime(ts: string | null): string {
  if (!ts) return "Never";
  const diff = Date.now() - new Date(ts).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function formatBucketLabel(iso: string, trunc: string): string {
  const d = new Date(iso);
  if (trunc === "day") {
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  }
  return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

function formatBucketTooltip(iso: string, trunc: string): string {
  const d = new Date(iso);
  if (trunc === "day") {
    return d.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
  }
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

// Slim horizontal progress bar — replaces ugly recharts radial gauge.
function ProgressRow({
  done,
  total,
  doneColor,
}: {
  done: number;
  total: number;
  doneColor: string;
}) {
  const ratio = total > 0 ? done / total : 0;
  const percent = Math.round(ratio * 100);
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between">
        <span className="text-xs text-muted-foreground">
          <span className="tabular-nums font-semibold text-foreground text-sm">
            {done.toLocaleString()}
          </span>
          <span className="text-muted-foreground"> / {total.toLocaleString()}</span>
        </span>
        <span className="text-xs font-semibold tabular-nums text-foreground">{percent}%</span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${percent}%`, backgroundColor: doneColor }}
        />
      </div>
    </div>
  );
}

interface DistributionItem {
  name: string;
  value: number;
  color: string;
}

function Distribution({
  title,
  items,
  emptyLabel,
}: {
  title: string;
  items: DistributionItem[];
  emptyLabel: string;
}) {
  const total = items.reduce((s, i) => s + i.value, 0);
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-[0.08em]">
          {title}
        </h4>
        <CompactNumber value={total} className="text-xs text-muted-foreground tabular-nums" />
      </div>
      {total === 0 ? (
        <div className="text-xs text-muted-foreground py-2">{emptyLabel}</div>
      ) : (
        <>
          <div className="h-1.5 flex w-full rounded-full overflow-hidden bg-muted">
            {items
              .filter((d) => d.value > 0)
              .map((d) => (
                <div
                  key={d.name}
                  className="h-full"
                  style={{
                    width: `${(d.value / total) * 100}%`,
                    backgroundColor: d.color,
                  }}
                  title={`${d.name}: ${d.value.toLocaleString()}`}
                />
              ))}
          </div>
          <div className="grid grid-cols-3 gap-3 text-sm">
            {items.map((d) => (
              <div key={d.name} className="space-y-0.5">
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-[2px]" style={{ backgroundColor: d.color }} />
                  <span className="text-[11px] uppercase tracking-wider text-muted-foreground font-medium">
                    {d.name}
                  </span>
                </div>
                <div className="flex items-baseline gap-1.5">
                  <CompactNumber value={d.value} className="text-base font-semibold tabular-nums" />
                  <span className="text-[10px] text-muted-foreground tabular-nums">
                    {total > 0 ? `${((d.value / total) * 100).toFixed(0)}%` : "0%"}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function InlineStat({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Database;
  label: string;
  value: number;
}) {
  return (
    <div className="flex items-center gap-3 p-4">
      <div className="p-2 rounded-md bg-muted">
        <Icon className="w-4 h-4 text-muted-foreground" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-xs text-muted-foreground font-medium">{label}</p>
        <CompactNumber
          value={value}
          className="text-2xl font-semibold text-foreground leading-tight tabular-nums block"
        />
      </div>
    </div>
  );
}

const OPS_STATUS_ORDER = ["completed", "in_progress", "pending", "failed", "cancelled"] as const;
const OPS_STATUS_COLORS: Record<string, string> = {
  completed: CHART_COLORS.success,
  in_progress: CHART_COLORS.semantic,
  pending: CHART_COLORS.warning,
  failed: CHART_COLORS.danger,
  cancelled: CHART_COLORS.mutedFg,
};
const OPS_STATUS_LABELS: Record<string, string> = {
  completed: "Completed",
  in_progress: "In progress",
  pending: "Pending",
  failed: "Failed",
  cancelled: "Cancelled",
};

interface OpsStatusEntry {
  status: string;
  label: string;
  value: number;
  color: string;
}

function OperationsCard({ byStatus }: { byStatus: Record<string, number> }) {
  const entries: OpsStatusEntry[] = OPS_STATUS_ORDER.map((s) => ({
    status: s,
    label: OPS_STATUS_LABELS[s],
    value: byStatus[s] || 0,
    color: OPS_STATUS_COLORS[s],
  }));
  for (const [key, val] of Object.entries(byStatus)) {
    if (!OPS_STATUS_ORDER.includes(key as (typeof OPS_STATUS_ORDER)[number])) {
      entries.push({ status: key, label: key, value: val, color: CHART_COLORS.mutedFg });
    }
  }
  const total = entries.reduce((sum, e) => sum + e.value, 0);
  const visible = entries.filter((e) => e.value > 0);

  return (
    <Card>
      <CardHeader className="pb-2 flex flex-row items-center justify-between space-y-0">
        <CardTitle className="text-sm font-semibold flex items-center gap-2">
          <Activity className="w-3.5 h-3.5 text-muted-foreground" />
          Operations
        </CardTitle>
        <span className="text-xs text-muted-foreground tabular-nums">
          <CompactNumber value={total} /> total
        </span>
      </CardHeader>
      <CardContent>
        {total === 0 ? (
          <div className="h-[100px] flex items-center justify-center text-sm text-muted-foreground">
            No operations yet
          </div>
        ) : (
          <div className="space-y-3">
            <div className="h-1.5 flex w-full rounded-full overflow-hidden bg-muted">
              {visible.map((e) => (
                <div
                  key={e.status}
                  className="h-full"
                  style={{
                    width: `${(e.value / total) * 100}%`,
                    backgroundColor: e.color,
                  }}
                  title={`${e.label}: ${e.value.toLocaleString()}`}
                />
              ))}
            </div>
            <div className="space-y-1.5">
              {visible.map((e) => (
                <div key={e.status} className="flex items-center gap-2 text-xs">
                  <span
                    className="w-2 h-2 rounded-[2px] flex-shrink-0"
                    style={{ backgroundColor: e.color }}
                  />
                  <span className="text-muted-foreground flex-1">{e.label}</span>
                  <CompactNumber
                    value={e.value}
                    className="font-semibold tabular-nums text-foreground"
                  />
                  <span className="text-[10px] text-muted-foreground tabular-nums w-9 text-right">
                    {((e.value / total) * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ConsolidationCard({
  done,
  pending,
  failed,
  total,
  lastConsolidatedAt,
}: {
  done: number;
  pending: number;
  failed: number;
  total: number;
  lastConsolidatedAt: string | null;
}) {
  const [failedOpen, setFailedOpen] = useState(false);
  const hasFailed = failed > 0;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold">Consolidation</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <ProgressRow done={done} total={total} doneColor={CHART_COLORS.success} />
        <div className="grid grid-cols-4 gap-3 pt-1">
          <div className="space-y-0.5">
            <div className="flex items-center gap-1.5">
              <CheckCircle2 className="w-3 h-3 text-emerald-500" />
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium">
                Done
              </span>
            </div>
            <CompactNumber
              value={done}
              className="text-base font-semibold tabular-nums text-foreground block"
            />
          </div>
          <div className="space-y-0.5">
            <div className="flex items-center gap-1.5">
              <AlertCircle className="w-3 h-3 text-amber-500" />
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium">
                Pending
              </span>
            </div>
            <CompactNumber
              value={pending}
              className="text-base font-semibold tabular-nums text-foreground block"
            />
          </div>
          <button
            type="button"
            onClick={() => hasFailed && setFailedOpen(true)}
            disabled={!hasFailed}
            className={`text-left space-y-0.5 rounded-md -mx-1 px-1 py-0.5 transition-colors ${
              hasFailed
                ? "cursor-pointer hover:bg-red-500/10 focus:outline-none focus:ring-2 focus:ring-red-500/40"
                : "cursor-default"
            }`}
            title={hasFailed ? "View failed memories" : undefined}
          >
            <div className="flex items-center gap-1.5">
              <XCircle
                className={`w-3 h-3 ${
                  hasFailed ? "text-red-600 dark:text-red-400" : "text-muted-foreground/50"
                }`}
              />
              <span
                className={`text-[10px] uppercase tracking-wider font-medium ${
                  hasFailed ? "text-red-600 dark:text-red-400" : "text-muted-foreground"
                }`}
              >
                Failed
              </span>
            </div>
            <span
              className={`text-base font-semibold tabular-nums block ${
                hasFailed ? "text-red-600 dark:text-red-400" : "text-foreground"
              }`}
            >
              <CompactNumber value={failed} />
              {hasFailed && <ExternalLink className="inline w-3 h-3 ml-1 opacity-70" />}
            </span>
          </button>
          <div className="space-y-0.5">
            <div className="flex items-center gap-1.5">
              <Clock className="w-3 h-3 text-muted-foreground" />
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium">
                Last
              </span>
            </div>
            <span className="text-base font-semibold text-foreground block leading-tight">
              {formatRelativeTime(lastConsolidatedAt)}
            </span>
          </div>
        </div>
      </CardContent>
      <FailedConsolidationsDialog open={failedOpen} onOpenChange={setFailedOpen} />
    </Card>
  );
}

interface FailedMemoryItem {
  id: string;
  text: string;
  context: string;
  fact_type: string;
  consolidation_failed_at: string | null;
}

function FailedConsolidationsDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (value: boolean) => void;
}) {
  const { currentBank } = useBank();
  const [items, setItems] = useState<FailedMemoryItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [recovering, setRecovering] = useState(false);
  const [refreshTick, setRefreshTick] = useState(0);

  useEffect(() => {
    if (!open || !currentBank) return;
    let cancelled = false;
    setLoading(true);
    client
      .listMemories(currentBank, { consolidationState: "failed", limit: 200 })
      .then((res) => {
        if (cancelled) return;
        setItems(res.items as FailedMemoryItem[]);
        setTotal(res.total);
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, currentBank, refreshTick]);

  const handleRecover = async () => {
    if (!currentBank) return;
    setRecovering(true);
    try {
      const res = await client.recoverConsolidation(currentBank);
      // The backend recovery endpoint only clears consolidation_failed_at; it does
      // not queue a consolidation task. Kick one off here so the reset memories
      // are actually picked up by the worker instead of sitting in pending state.
      if (res.retried_count > 0) {
        await client.triggerConsolidation(currentBank);
      }
      toast.success(
        `Queued ${res.retried_count} memor${res.retried_count === 1 ? "y" : "ies"} for re-consolidation`
      );
      setRefreshTick((t) => t + 1);
    } catch {
      // toast shown by interceptor
    } finally {
      setRecovering(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[80vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <XCircle className="w-4 h-4 text-red-600 dark:text-red-400" />
            Failed consolidations
          </DialogTitle>
          <DialogDescription>
            Source memories whose consolidation permanently failed. Recovery resets them so they are
            retried on the next consolidation run.
          </DialogDescription>
        </DialogHeader>
        <div className="flex items-center justify-between gap-2 py-2">
          <span className="text-sm text-muted-foreground">
            {loading ? "Loading…" : `${total} failed`}
          </span>
          <Button size="sm" onClick={handleRecover} disabled={recovering || loading || total === 0}>
            <RefreshCw className={`w-3.5 h-3.5 mr-1.5 ${recovering ? "animate-spin" : ""}`} />
            Recover all
          </Button>
        </div>
        <div className="flex-1 min-h-0 overflow-auto border border-border rounded-md">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[140px]">Failed at</TableHead>
                <TableHead className="w-[100px]">Type</TableHead>
                <TableHead>Text</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {items.length === 0 && !loading && (
                <TableRow>
                  <TableCell colSpan={3} className="text-center text-muted-foreground py-8">
                    No failed consolidations.
                  </TableCell>
                </TableRow>
              )}
              {items.map((row) => (
                <TableRow key={row.id}>
                  <TableCell className="text-xs text-muted-foreground whitespace-nowrap">
                    {formatRelativeTime(row.consolidation_failed_at)}
                  </TableCell>
                  <TableCell className="text-xs capitalize">{row.fact_type}</TableCell>
                  <TableCell className="text-sm">
                    <div className="line-clamp-2">{row.text}</div>
                    {row.context && (
                      <div className="text-xs text-muted-foreground mt-0.5 line-clamp-1">
                        {row.context}
                      </div>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function MentalModelsCard({
  models,
  lastConsolidatedAt,
}: {
  models: MentalModel[];
  lastConsolidatedAt: string | null;
}) {
  const total = models.length;
  const consolidatedTime = lastConsolidatedAt ? new Date(lastConsolidatedAt).getTime() : 0;
  const upToDate = models.filter((m) => {
    if (!consolidatedTime) return true;
    if (!m.last_refreshed_at) return false;
    return new Date(m.last_refreshed_at).getTime() >= consolidatedTime;
  }).length;
  const stale = total - upToDate;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold flex items-center gap-2">
          <Brain className="w-3.5 h-3.5 text-muted-foreground" />
          Mental Models
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {total === 0 ? (
          <div className="text-sm text-muted-foreground py-4">No mental models</div>
        ) : (
          <>
            <ProgressRow done={upToDate} total={total} doneColor={CHART_COLORS.success} />
            <div className="grid grid-cols-3 gap-3 pt-1">
              <div className="space-y-0.5">
                <div className="flex items-center gap-1.5">
                  <CheckCircle2 className="w-3 h-3 text-emerald-500" />
                  <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium">
                    Up to date
                  </span>
                </div>
                <span className="text-base font-semibold tabular-nums text-foreground block">
                  {upToDate}
                </span>
              </div>
              <div className="space-y-0.5">
                <div className="flex items-center gap-1.5">
                  <AlertCircle className="w-3 h-3 text-amber-500" />
                  <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium">
                    Stale
                  </span>
                </div>
                <span className="text-base font-semibold tabular-nums text-foreground block">
                  {stale}
                </span>
              </div>
              <div className="space-y-0.5">
                <div className="flex items-center gap-1.5">
                  <Brain className="w-3 h-3 text-muted-foreground" />
                  <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium">
                    Total
                  </span>
                </div>
                <span className="text-base font-semibold tabular-nums text-foreground block">
                  {total}
                </span>
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}

const AXIS_TICK_STYLE = {
  fontSize: 11,
  fill: CHART_COLORS.mutedFg,
  fontWeight: 500,
};

export function BankStatsView() {
  const { currentBank } = useBank();
  const { features } = useFeatures();
  const observationsEnabled = features?.observations ?? false;
  const [stats, setStats] = useState<BankStats | null>(null);
  const [mentalModels, setMentalModels] = useState<MentalModel[]>([]);
  const [loading, setLoading] = useState(false);
  const [period, setPeriod] = useState<Period>("7d");
  const [timeseries, setTimeseries] = useState<{ trunc: string; buckets: TimeseriesBucket[] }>({
    trunc: "day",
    buckets: [],
  });
  const [enabledSeries, setEnabledSeries] = useState<Record<FactKey, boolean>>({
    world: true,
    experience: true,
    observation: true,
  });

  const loadData = async () => {
    if (!currentBank) return;

    setLoading(true);
    try {
      const [statsData, mentalModelsData] = await Promise.all([
        client.getBankStats(currentBank),
        client.listMentalModels(currentBank),
      ]);
      setStats(statsData as BankStats);
      setMentalModels(mentalModelsData.items || []);
    } catch (error) {
      console.error("Error loading bank stats:", error);
    } finally {
      setLoading(false);
    }
  };

  const loadTimeseries = async () => {
    if (!currentBank) return;
    try {
      const data = await client.getMemoriesTimeseries(currentBank, period);
      setTimeseries({ trunc: data.trunc, buckets: data.buckets || [] });
    } catch (error) {
      console.error("Error loading memories timeseries:", error);
    }
  };

  useEffect(() => {
    if (currentBank) {
      loadData();
      const interval = setInterval(loadData, 5000);
      return () => clearInterval(interval);
    }
  }, [currentBank]);

  useEffect(() => {
    if (currentBank) {
      loadTimeseries();
      const interval = setInterval(loadTimeseries, 5000);
      return () => clearInterval(interval);
    }
  }, [currentBank, period]);

  if (loading && !stats) {
    return (
      <div className="flex items-center justify-center py-12">
        <Clock className="w-12 h-12 mx-auto mb-3 text-muted-foreground animate-pulse" />
      </div>
    );
  }

  if (!stats) return null;

  const factSeries: FactKey[] = observationsEnabled
    ? ["world", "experience", "observation"]
    : ["world", "experience"];

  const chartData = timeseries.buckets.map((b) => ({
    ...b,
    label: formatBucketLabel(b.time, timeseries.trunc),
    tooltipLabel: formatBucketTooltip(b.time, timeseries.trunc),
  }));
  const ingestedTotal = chartData.reduce(
    (sum, b) => sum + factSeries.reduce((s, k) => s + (enabledSeries[k] ? b[k] || 0 : 0), 0),
    0
  );

  const toggleSeries = (k: FactKey) => setEnabledSeries((prev) => ({ ...prev, [k]: !prev[k] }));

  const consolidatedDone = Math.max(0, stats.total_nodes - stats.pending_consolidation);

  return (
    <div className="space-y-8">
      {/* MEMORY STORE — unified card: top stat strip + composition + link types */}
      <section>
        <SectionHeading>Memory store</SectionHeading>
        <Card>
          <CardContent className="p-0">
            {/* Stat strip */}
            <div className="grid grid-cols-1 md:grid-cols-3 md:divide-x divide-y md:divide-y-0 divide-border/60">
              <InlineStat icon={Database} label="Memories" value={stats.total_nodes} />
              <InlineStat icon={FolderOpen} label="Documents" value={stats.total_documents} />
              <InlineStat icon={Link2} label="Links" value={stats.total_links} />
            </div>

            {/* Composition + Link types side by side */}
            <div className="grid grid-cols-1 md:grid-cols-2 md:divide-x divide-y md:divide-y-0 divide-border/60 border-t border-border/60">
              <div className="p-5">
                <Distribution
                  title="Memory composition"
                  items={[
                    {
                      name: "World",
                      value: stats.nodes_by_fact_type?.world || 0,
                      color: CHART_COLORS.world,
                    },
                    {
                      name: "Experience",
                      value: stats.nodes_by_fact_type?.experience || 0,
                      color: CHART_COLORS.experience,
                    },
                    ...(observationsEnabled
                      ? [
                          {
                            name: "Observations",
                            value: stats.total_observations || 0,
                            color: CHART_COLORS.observation,
                          },
                        ]
                      : []),
                  ]}
                  emptyLabel="No memories yet"
                />
              </div>
              <div className="p-5">
                <Distribution
                  title="Link types"
                  items={[
                    {
                      name: "Temporal",
                      value: stats.links_by_link_type?.temporal || 0,
                      color: CHART_COLORS.temporal,
                    },
                    {
                      name: "Semantic",
                      value: stats.links_by_link_type?.semantic || 0,
                      color: CHART_COLORS.semantic,
                    },
                    {
                      name: "Entity",
                      value: stats.links_by_link_type?.entity || 0,
                      color: CHART_COLORS.entity,
                    },
                  ]}
                  emptyLabel="No links yet"
                />
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* CONSOLIDATION */}
      <section>
        <SectionHeading>Consolidation</SectionHeading>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <ConsolidationCard
            done={consolidatedDone}
            pending={stats.pending_consolidation}
            failed={stats.failed_consolidation ?? 0}
            total={stats.total_nodes}
            lastConsolidatedAt={stats.last_consolidated_at}
          />
          <MentalModelsCard models={mentalModels} lastConsolidatedAt={stats.last_consolidated_at} />
        </div>
      </section>

      {/* ACTIVITY */}
      <section>
        <SectionHeading>Activity</SectionHeading>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <Card className="lg:col-span-2">
            <CardHeader className="pb-2 space-y-2">
              <div className="flex flex-row items-center justify-between">
                <CardTitle className="text-sm font-semibold">Memories ingested</CardTitle>
                <div className="flex items-center gap-0.5 rounded-md bg-muted/60 p-0.5">
                  {PERIODS.map((p) => (
                    <button
                      key={p}
                      onClick={() => setPeriod(p)}
                      className={`px-2 py-0.5 text-[11px] font-medium rounded transition-colors tabular-nums ${
                        period === p
                          ? "bg-background text-foreground shadow-sm"
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  {factSeries.map((k) => {
                    const meta = FACT_META[k];
                    const on = enabledSeries[k];
                    return (
                      <button
                        key={k}
                        onClick={() => toggleSeries(k)}
                        className={`flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[11px] font-medium transition-all ${
                          on
                            ? "bg-muted text-foreground"
                            : "bg-transparent text-muted-foreground/60 hover:text-muted-foreground"
                        }`}
                      >
                        <span
                          className="w-2 h-2 rounded-[2px] transition-opacity"
                          style={{
                            backgroundColor: meta.color,
                            opacity: on ? 1 : 0.3,
                          }}
                        />
                        {meta.label}
                      </button>
                    );
                  })}
                </div>
                <span className="text-xs text-muted-foreground tabular-nums">
                  <CompactNumber value={ingestedTotal} /> total
                </span>
              </div>
            </CardHeader>
            <CardContent>
              {chartData.length === 0 || ingestedTotal === 0 ? (
                <div className="h-[180px] flex items-center justify-center text-sm text-muted-foreground">
                  No memories ingested in this period
                </div>
              ) : (
                <div className="h-[180px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: -10 }}>
                      <defs>
                        {factSeries.map((k) => (
                          <linearGradient key={k} id={`grad-${k}`} x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor={FACT_META[k].color} stopOpacity={0.35} />
                            <stop offset="100%" stopColor={FACT_META[k].color} stopOpacity={0} />
                          </linearGradient>
                        ))}
                      </defs>
                      <CartesianGrid
                        vertical={false}
                        stroke={CHART_COLORS.border}
                        strokeOpacity={0.5}
                        strokeDasharray="2 4"
                      />
                      <XAxis
                        dataKey="label"
                        tick={AXIS_TICK_STYLE}
                        axisLine={false}
                        tickLine={false}
                        dy={4}
                        minTickGap={20}
                      />
                      <YAxis
                        tick={AXIS_TICK_STYLE}
                        axisLine={false}
                        tickLine={false}
                        width={40}
                        allowDecimals={false}
                        tickFormatter={(v: number) => formatCompact(v)}
                      />
                      <Tooltip
                        content={<ChartTooltip />}
                        labelFormatter={(_v, payload) => payload?.[0]?.payload?.tooltipLabel || ""}
                        cursor={{
                          stroke: CHART_COLORS.mutedFg,
                          strokeWidth: 1,
                          strokeDasharray: "3 3",
                        }}
                      />
                      {factSeries.map((k) =>
                        enabledSeries[k] ? (
                          <Area
                            key={k}
                            type="monotone"
                            dataKey={k}
                            name={FACT_META[k].label}
                            stackId="a"
                            stroke={FACT_META[k].color}
                            strokeWidth={2}
                            fill={`url(#grad-${k})`}
                            isAnimationActive={false}
                          />
                        ) : null
                      )}
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              )}
            </CardContent>
          </Card>

          <OperationsCard byStatus={stats.operations_by_status || {}} />
        </div>
      </section>
    </div>
  );
}

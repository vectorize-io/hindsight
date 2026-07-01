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
  RefreshCw,
  Server,
  Cpu,
  HardDrive,
  Zap,
  AlertCircle,
  CheckCircle2,
  Network,
  Brain,
  Bot,
  Workflow,
  BarChart3,
  TrendingUp,
  TrendingDown,
} from "lucide-react";
import { EngineLoadBar } from "@/components/engine-load-bar";
import { MetricTile, useMetricHistory } from "@/components/metric-sparkline";
import { ServiceHealthTable, categorizeServices } from "@/components/service-health-table";

// ── Types ─────────────────────────────────────────────────────────

interface ServiceStatus {
  name: string;
  port: number;
  status: string;
  health?: string;
  uptime?: string;
  cpu?: number;
  memory?: number;
  pid?: number;
}

interface Operation {
  id: string;
  task_type: string;
  items_count: number;
  status: string;
  created_at: string;
  error_message: string | null;
  retry_count: number;
}

interface LlmStats {
  total_requests?: number;
  total_tokens?: number;
  avg_latency_ms?: number;
  error_rate?: number;
  requests_by_model?: Record<string, number>;
  requests_by_status?: Record<string, number>;
  time_series?: Array<{ timestamp: string; requests: number; tokens: number }>;
  [key: string]: unknown;
}

interface BankStats {
  memories?: {
    total?: number;
    by_type?: Record<string, number>;
    [key: string]: unknown;
  };
  operations?: {
    total?: number;
    completed?: number;
    failed?: number;
    pending?: number;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

// ── Helper Functions ──────────────────────────────────────────────

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

// ── Service Status Card ───────────────────────────────────────────

function ServiceCard({ service }: { service: ServiceStatus }) {
  const isHealthy = service.status === "running" && service.health === "healthy";
  const isDegraded = service.status === "running" && service.health !== "healthy";
  const isStopped = service.status === "stopped";

  return (
    <div
      className={`rounded-lg border p-3 transition-colors ${
        isHealthy
          ? "bg-green-50/30 dark:bg-green-950/10 border-green-200 dark:border-green-800"
          : isDegraded
            ? "bg-amber-50/30 dark:bg-amber-950/10 border-amber-200 dark:border-amber-800"
            : "bg-red-50/30 dark:bg-red-950/10 border-red-200 dark:border-red-800"
      }`}
    >
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-xs font-medium truncate">{service.name}</span>
        <div className="flex items-center gap-1">
          {isHealthy ? (
            <CheckCircle2 className="w-3 h-3 text-green-500" />
          ) : isDegraded ? (
            <AlertCircle className="w-3 h-3 text-amber-500" />
          ) : (
            <AlertCircle className="w-3 h-3 text-red-500" />
          )}
          <span
            className={`text-[10px] font-mono ${
              isHealthy
                ? "text-green-600 dark:text-green-400"
                : isDegraded
                  ? "text-amber-600 dark:text-amber-400"
                  : "text-red-600 dark:text-red-400"
            }`}
          >
            {isHealthy ? "OK" : isDegraded ? "warn" : "down"}
          </span>
        </div>
      </div>
      <div className="text-[10px] text-muted-foreground space-y-0.5">
        <div className="flex items-center justify-between">
          <span>Port {service.port}</span>
          {service.cpu !== undefined && <span>CPU {service.cpu.toFixed(0)}%</span>}
        </div>
        {service.memory !== undefined && (
          <div className="flex items-center justify-between">
            <span>Memory</span>
            <span>{service.memory.toFixed(0)}%</span>
          </div>
        )}
        {service.uptime && (
          <div className="flex items-center justify-between">
            <span>Uptime</span>
            <span className="font-mono">{service.uptime}</span>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Alert Banner ──────────────────────────────────────────────────

function AlertBanner({
  failedOps,
  pendingOps,
  opsLoading,
}: {
  failedOps: number;
  pendingOps: number;
  opsLoading: boolean;
}) {
  const hasIssues = failedOps > 0 || pendingOps > 0;

  if (opsLoading || !hasIssues) return null;

  return (
    <Card className="border-amber-200 dark:border-amber-800 bg-amber-50/30 dark:bg-amber-950/10">
      <CardContent className="p-3 flex items-center gap-3">
        <AlertCircle className="w-5 h-5 text-amber-500 shrink-0" />
        <div className="text-xs">
          <p className="font-medium">Attention Required</p>
          <p className="text-muted-foreground mt-0.5">
            {failedOps > 0 && `${failedOps} failed operation(s) `}
            {pendingOps > 0 && `${pendingOps} pending operation(s) `}
            need review
          </p>
        </div>
        <Button variant="outline" size="sm" className="ml-auto text-xs h-7" asChild>
          <a href="/runs">View Runs</a>
        </Button>
      </CardContent>
    </Card>
  );
}

// ── Simple Bar Chart ──────────────────────────────────────────────

function SimpleBarChart({
  data,
  height = 100,
  color = "var(--primary)",
}: {
  data: number[];
  height?: number;
  color?: string;
}) {
  if (data.length === 0) return null;

  const max = Math.max(...data, 1);

  return (
    <div className="flex items-end gap-[2px]" style={{ height }}>
      {data.map((value, i) => {
        const pct = (value / max) * 100;
        return (
          <div
            key={i}
            className="flex-1 rounded-t transition-all duration-300"
            style={{
              height: `${Math.max(pct, 2)}%`,
              backgroundColor: color,
              opacity: 0.5 + (value / max) * 0.5,
            }}
            title={`${value}`}
          />
        );
      })}
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────

export default function MonitoringPage() {
  const t = useTranslations("operator");

  // Services
  const [services, setServices] = useState<ServiceStatus[]>([]);
  const [servicesLoading, setServicesLoading] = useState(true);

  // Operations
  const [operations, setOperations] = useState<Operation[]>([]);
  const [opsLoading, setOpsLoading] = useState(true);

  // LLM Stats
  const [llmStats, setLlmStats] = useState<LlmStats | null>(null);
  const [llmStatsLoading, setLlmStatsLoading] = useState(true);

  // Bank Stats
  const [bankStats, setBankStats] = useState<BankStats | null>(null);
  const [bankStatsLoading, setBankStatsLoading] = useState(true);

  // History tracking
  const servicesHistory = useMetricHistory("servicesTotal", services.length);
  const [latencyHistory, setLatencyHistory] = useState<number[]>([]);

  // Derived stats
  const failedOps = useMemo(
    () => operations.filter((o) => o.status === "failed").length,
    [operations]
  );
  const pendingOps = useMemo(
    () => operations.filter((o) => o.status === "pending" || o.status === "processing").length,
    [operations]
  );
  const completedOps = useMemo(
    () => operations.filter((o) => o.status === "completed").length,
    [operations]
  );

  const runningServices = useMemo(
    () => services.filter((s) => s.status === "running").length,
    [services]
  );
  const healthyServices = useMemo(
    () => services.filter((s) => s.status === "running" && s.health === "healthy").length,
    [services]
  );

  // Fetch functions
  const fetchServices = async () => {
    setServicesLoading(true);
    try {
      const res = await fetch("/api/system/services", { signal: AbortSignal.timeout(5000) });
      if (res.ok) {
        const data = await res.json();
        setServices(data.services || []);
      }
    } catch {
      // ignore
    }
    setServicesLoading(false);
  };

  const fetchOperations = async () => {
    setOpsLoading(true);
    try {
      const res = await fetch("/api/system/operations", { signal: AbortSignal.timeout(5000) });
      if (res.ok) {
        const data = await res.json();
        setOperations(data.operations || []);
      }
    } catch {
      // ignore
    }
    setOpsLoading(false);
  };

  const fetchLlmStats = async () => {
    setLlmStatsLoading(true);
    try {
      const res = await fetch("/api/monitoring/llm-stats", { signal: AbortSignal.timeout(5000) });
      if (res.ok) {
        const data = await res.json();
        setLlmStats(data);
        // Track latency history
        if (data.avg_latency_ms != null) {
          setLatencyHistory((prev) => {
            const next = [...prev, data.avg_latency_ms];
            return next.slice(-60);
          });
        }
      }
    } catch {
      // ignore
    }
    setLlmStatsLoading(false);
  };

  const fetchAll = () => {
    fetchServices();
    fetchOperations();
    fetchLlmStats();
  };

  useEffect(() => {
    fetchAll();
    const sInterval = setInterval(fetchServices, 30000);
    const oInterval = setInterval(fetchOperations, 15000);
    const lInterval = setInterval(fetchLlmStats, 30000);
    return () => {
      clearInterval(sInterval);
      clearInterval(oInterval);
      clearInterval(lInterval);
    };
  }, []);

  // Group services for grid display
  const coreServices = services.filter((s) =>
    ["Hindsight API", "PostgreSQL", "Central API", "Cockpit"].includes(s.name)
  );
  const aiServices = services.filter((s) =>
    ["Ollama Embeddings", "Ollama LLM", "LM Studio", "Workers"].includes(s.name)
  );
  const infraServices = services.filter((s) =>
    ["Memlord", "Grafana", "Control Plane"].includes(s.name)
  );

  return (
    <OperatorShell>
      <div className="p-6 space-y-6">
        {/* Page Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Monitoring</h1>
            <p className="text-sm text-muted-foreground">
              {t("descriptions.monitoring") || "System health, metrics, and graphs"}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-[10px] font-mono">
              {healthyServices}/{services.length} healthy
            </Badge>
            <Button variant="outline" size="sm" onClick={fetchAll} disabled={servicesLoading}>
              <RefreshCw className={`h-4 w-4 mr-1 ${servicesLoading ? "animate-spin" : ""}`} />
              Refresh
            </Button>
          </div>
        </div>

        {/* Alert banner */}
        <AlertBanner failedOps={failedOps} pendingOps={pendingOps} opsLoading={opsLoading} />

        {/* Top Metric Tiles */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricTile
            title="Services"
            value={`${healthyServices}/${services.length}`}
            icon={<Server className="w-3.5 h-3.5" />}
            history={servicesHistory}
            loading={servicesLoading && services.length === 0}
            color="#10b981"
          />
          <MetricTile
            title="Completed Ops"
            value={completedOps}
            icon={<CheckCircle2 className="w-3.5 h-3.5" />}
            history={[]}
            loading={opsLoading}
            color="#3b82f6"
          />
          <MetricTile
            title="Failed Ops"
            value={failedOps}
            icon={<AlertCircle className="w-3.5 h-3.5" />}
            history={[]}
            loading={opsLoading}
            color={failedOps > 0 ? "#ef4444" : "#6b7280"}
          />
          <MetricTile
            title="Pending Ops"
            value={pendingOps}
            icon={<Activity className="w-3.5 h-3.5" />}
            history={[]}
            loading={opsLoading}
            color={pendingOps > 0 ? "#f59e0b" : "#6b7280"}
          />
        </div>

        {/* Main grid: metrics + graphs */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* LLM Request Latency Trend */}
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <Zap className="h-4 w-4" /> LLM Latency
                </CardTitle>
                <Badge variant="outline" className="text-[10px]">
                  {llmStatsLoading ? "..." : `${latencyHistory.length} samples`}
                </Badge>
              </div>
              <CardDescription>Avg LLM request latency (ms)</CardDescription>
            </CardHeader>
            <CardContent>
              {llmStatsLoading && latencyHistory.length === 0 ? (
                <div className="flex items-center justify-center h-[100px] text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin mr-2" /> Loading...
                </div>
              ) : latencyHistory.length === 0 ? (
                <div className="flex items-center justify-center h-[100px] text-xs text-muted-foreground">
                  No latency data yet
                </div>
              ) : (
                <>
                  <SimpleBarChart
                    data={latencyHistory}
                    height={100}
                    color="var(--color-primary, #8b5cf6)"
                  />
                  <div className="flex items-center justify-between mt-2 text-[10px] text-muted-foreground">
                    <span>Now</span>
                    <span className="font-mono font-medium">
                      {llmStats?.avg_latency_ms?.toFixed(0) || "—"} ms avg
                    </span>
                    <span>{latencyHistory.length}m ago</span>
                  </div>
                </>
              )}
            </CardContent>
          </Card>

          {/* Operation Status Breakdown */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-sm">
                <Workflow className="h-4 w-4" /> Operation Status
              </CardTitle>
              <CardDescription>Breakdown of all operations</CardDescription>
            </CardHeader>
            <CardContent>
              {opsLoading && operations.length === 0 ? (
                <div className="flex items-center justify-center h-[100px] text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin mr-2" /> Loading...
                </div>
              ) : operations.length === 0 ? (
                <div className="flex items-center justify-center h-[100px] text-xs text-muted-foreground">
                  No operations recorded
                </div>
              ) : (
                <div className="space-y-3">
                  {/* Visual bar */}
                  <div className="h-4 rounded-full overflow-hidden flex">
                    {completedOps > 0 && (
                      <div
                        className="bg-green-500 transition-all"
                        style={{ flex: completedOps }}
                        title={`${completedOps} completed`}
                      />
                    )}
                    {pendingOps > 0 && (
                      <div
                        className="bg-amber-500 transition-all"
                        style={{ flex: pendingOps }}
                        title={`${pendingOps} pending`}
                      />
                    )}
                    {failedOps > 0 && (
                      <div
                        className="bg-red-500 transition-all"
                        style={{ flex: failedOps }}
                        title={`${failedOps} failed`}
                      />
                    )}
                  </div>
                  {/* Legend */}
                  <div className="grid grid-cols-3 gap-2 text-center text-[10px]">
                    <div>
                      <div className="flex items-center justify-center gap-1">
                        <div className="w-2 h-2 rounded-full bg-green-500" />
                        <span className="font-medium">{completedOps}</span>
                      </div>
                      <span className="text-muted-foreground">Completed</span>
                    </div>
                    <div>
                      <div className="flex items-center justify-center gap-1">
                        <div className="w-2 h-2 rounded-full bg-amber-500" />
                        <span className="font-medium">{pendingOps}</span>
                      </div>
                      <span className="text-muted-foreground">Pending</span>
                    </div>
                    <div>
                      <div className="flex items-center justify-center gap-1">
                        <div className="w-2 h-2 rounded-full bg-red-500" />
                        <span className="font-medium">{failedOps}</span>
                      </div>
                      <span className="text-muted-foreground">Failed</span>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* LLM Request Stats */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-sm">
                <BarChart3 className="h-4 w-4" /> LLM Requests
              </CardTitle>
              <CardDescription>
                {llmStatsLoading
                  ? "Loading..."
                  : llmStats?.total_requests != null
                    ? `${llmStats.total_requests} total requests`
                    : "No data"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {llmStatsLoading && !llmStats ? (
                <div className="flex items-center justify-center h-[100px] text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin mr-2" /> Loading...
                </div>
              ) : !llmStats || llmStats.total_requests == null ? (
                <div className="flex items-center justify-center h-[100px] text-xs text-muted-foreground">
                  No LLM request stats yet
                </div>
              ) : (
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">Total Tokens</span>
                    <span className="font-mono font-medium">
                      {llmStats.total_tokens?.toLocaleString() || "—"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">Avg Latency</span>
                    <span className="font-mono font-medium">
                      {llmStats.avg_latency_ms != null
                        ? `${llmStats.avg_latency_ms.toFixed(0)}ms`
                        : "—"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">Error Rate</span>
                    <span className="font-mono font-medium">
                      {llmStats.error_rate != null
                        ? `${(llmStats.error_rate * 100).toFixed(1)}%`
                        : "—"}
                    </span>
                  </div>
                  {llmStats.requests_by_model && (
                    <div className="mt-2 pt-2 border-t">
                      <p className="text-[10px] text-muted-foreground mb-1">By Model</p>
                      {Object.entries(llmStats.requests_by_model).map(([model, count]) => (
                        <div
                          key={model}
                          className="flex items-center justify-between text-[10px] py-0.5"
                        >
                          <span className="truncate font-mono">{model}</span>
                          <span>{count}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Service Health Grid */}
        <Card>
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2 text-sm">
                <Server className="h-4 w-4" /> Service Health
              </CardTitle>
              <Badge variant="outline" className="text-[10px] font-mono">
                {servicesLoading ? "..." : `${runningServices}/${services.length} running`}
              </Badge>
            </div>
            <CardDescription>Real-time status of all services</CardDescription>
          </CardHeader>
          <CardContent>
            {servicesLoading && services.length === 0 ? (
              <div className="flex items-center justify-center py-8 text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin mr-2" /> Loading services...
              </div>
            ) : (
              <div className="space-y-4">
                {/* Core Services */}
                <div>
                  <h4 className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-2">
                    Core
                  </h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                    {coreServices.map((s) => (
                      <ServiceCard key={s.name} service={s} />
                    ))}
                  </div>
                </div>

                {/* AI Services */}
                <div>
                  <h4 className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-2">
                    AI & Workers
                  </h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                    {aiServices.map((s) => (
                      <ServiceCard key={s.name} service={s} />
                    ))}
                  </div>
                </div>

                {/* Infrastructure */}
                <div>
                  <h4 className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-2">
                    Infrastructure
                  </h4>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                    {infraServices.map((s) => (
                      <ServiceCard key={s.name} service={s} />
                    ))}
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Full Service Health Table */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Network className="h-4 w-4" /> All Services — Detailed
            </CardTitle>
            <CardDescription>Categorized with resource metrics</CardDescription>
          </CardHeader>
          <CardContent>
            <ServiceHealthTable services={categorizeServices(services)} compact />
          </CardContent>
        </Card>

        {/* Engine Load */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Cpu className="h-4 w-4" /> Engine Resource Load
            </CardTitle>
            <CardDescription>CPU and memory utilization across services</CardDescription>
          </CardHeader>
          <CardContent>
            <EngineLoadBar
              cpu={services.find((s) => s.name === "Hindsight API")?.cpu}
              memory={services.find((s) => s.name === "Hindsight API")?.memory}
            />
          </CardContent>
        </Card>
      </div>
    </OperatorShell>
  );
}

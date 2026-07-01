"use client";

import { useState, useEffect, useMemo } from "react";
import { useTranslations } from "next-intl";
import { OperatorShell } from "@/components/operator-shell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Activity,
  Brain,
  Database,
  Cpu,
  Network,
  Shield,
  Bot,
  RefreshCw,
  ArrowRight,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Zap,
  BarChart3,
  Server,
  Workflow,
} from "lucide-react";
import Link from "next/link";
import { EngineLoadBar } from "@/components/engine-load-bar";
import { TraceStream } from "@/components/trace-stream";
import { MetricTile, useMetricHistory } from "@/components/metric-sparkline";
import { ServiceHealthTable, categorizeServices } from "@/components/service-health-table";
import { FusionStrategyMix } from "@/components/fusion-strategy-mix";

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
  document_id: string | null;
  created_at: string;
  status: string;
  error_message: string | null;
  retry_count: number;
  next_retry_at: string | null;
}

interface BankInfo {
  bank_id: string;
  name: string | null;
  created_at: string;
}

export default function CockpitPage() {
  const t = useTranslations("operator");

  // Services
  const [services, setServices] = useState<ServiceStatus[]>([]);
  const [servicesLoading, setServicesLoading] = useState(true);
  const [apiVersion, setApiVersion] = useState("");

  // Banks
  const [banks, setBanks] = useState<BankInfo[]>([]);
  const [banksLoading, setBanksLoading] = useState(true);

  // Operations
  const [operations, setOperations] = useState<Operation[]>([]);
  const [opsLoading, setOpsLoading] = useState(true);

  // Derived data
  const activeOps = useMemo(
    () => operations.filter((o) => o.status === "processing" || o.status === "pending"),
    [operations]
  );
  const apiService = services.find((s) => s.name === "Hindsight API");
  const dbService = services.find((s) => s.name === "PostgreSQL");
  const workersService = services.find((s) => s.name === "Workers");
  const engineCpu = apiService?.cpu;
  const engineMem = apiService?.memory;

  // Metric history (accumulated for sparklines)
  const bankCount = banks.length;
  const bankHistory = useMetricHistory("banks", bankCount);
  const activeOpsCount = activeOps.length;
  const activeOpsHistory = useMetricHistory("activeOps", activeOpsCount);
  const totalOpsCount = operations.length;
  const totalOpsHistory = useMetricHistory("totalOps", totalOpsCount);
  const runningCount = services.filter((s) => s.status === "running").length;
  const runningHistory = useMetricHistory("runningServices", runningCount);

  // Helper to parse worker count from health string
  const workerInfo = useMemo(() => {
    if (!workersService?.health) return undefined;
    const match = workersService.health.match(/(\d+)/);
    if (!match) return undefined;
    return { active: parseInt(match[1]), total: 10 };
  }, [workersService?.health]);

  const checkHealth = async () => {
    setServicesLoading(true);
    try {
      const res = await fetch("/api/system/services", { signal: AbortSignal.timeout(3000) });
      if (res.ok) {
        const data = await res.json();
        setServices(data.services || []);
      }
    } catch {
      setServices([]);
    }
    try {
      const vRes = await fetch("/api/system/config", { signal: AbortSignal.timeout(3000) });
      if (vRes.ok) {
        const vData = await vRes.json();
        setApiVersion(vData.api_version || vData.version || "");
      }
    } catch { /* ignore */ }
    setServicesLoading(false);
  };

  const loadBanks = async () => {
    setBanksLoading(true);
    try {
      const res = await fetch("/api/banks");
      if (res.ok) {
        const data = await res.json();
        setBanks(data.banks || []);
      }
    } catch { /* ignore */ }
    setBanksLoading(false);
  };

  const loadOperations = async () => {
    setOpsLoading(true);
    try {
      const res = await fetch("/api/system/operations?limit=10");
      if (res.ok) {
        const data = await res.json();
        setOperations(data.operations || []);
      }
    } catch { /* ignore */ }
    setOpsLoading(false);
  };

  useEffect(() => {
    checkHealth();
    loadBanks();
    loadOperations();
    const hInterval = setInterval(checkHealth, 30000);
    const oInterval = setInterval(loadOperations, 10000);
    return () => {
      clearInterval(hInterval);
      clearInterval(oInterval);
    };
  }, []);

  return (
    <OperatorShell>
      <div className="p-6 space-y-6">
        {/* Page Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Cockpit</h1>
            <p className="text-sm text-muted-foreground">
              Operator overview — all systems at a glance
            </p>
          </div>
          <div className="flex items-center gap-2">
            {apiVersion ? (
              <span className="text-[11px] font-mono text-muted-foreground">v{apiVersion}</span>
            ) : null}
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                checkHealth();
                loadBanks();
                loadOperations();
              }}
              disabled={servicesLoading}
            >
              <RefreshCw className={`h-4 w-4 mr-1 ${servicesLoading ? "animate-spin" : ""}`} />
              Refresh
            </Button>
          </div>
        </div>

        {/* P1-5: Metric Tiles with Sparklines */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <MetricTile
            title="Memory Banks"
            value={bankCount}
            icon={<Brain className="w-3.5 h-3.5" />}
            history={bankHistory}
            loading={banksLoading}
            color="#8b5cf6"
          />
          <MetricTile
            title="Active Ops"
            value={activeOpsCount}
            icon={<Activity className="w-3.5 h-3.5" />}
            history={activeOpsHistory}
            loading={opsLoading}
            color={activeOpsCount > 0 ? "#3b82f6" : "#6b7280"}
          />
          <MetricTile
            title="Running Services"
            value={runningCount}
            unit={`/ ${services.length}`}
            icon={<Server className="w-3.5 h-3.5" />}
            history={runningHistory}
            loading={servicesLoading}
            color="#10b981"
          />
          <MetricTile
            title="Total Operations"
            value={totalOpsCount}
            icon={<Zap className="w-3.5 h-3.5" />}
            history={totalOpsHistory}
            loading={opsLoading}
            color="#f59e0b"
          />
          <MetricTile
            title="API Status"
            value={apiService?.health === "healthy" ? "Healthy" : apiService?.health || "—"}
            icon={
              apiService?.health === "healthy" ? (
                <CheckCircle2 className="w-3.5 h-3.5 text-green-500" />
              ) : (
                <AlertCircle className="w-3.5 h-3.5 text-amber-500" />
              )
            }
            history={[]}
            loading={servicesLoading && !apiService}
            color="#10b981"
          />
        </div>

        {/* P1-1: Engine Load Visualization */}
        <Card>
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2 text-sm">
                <Cpu className="h-4 w-4" /> Engine Load
              </CardTitle>
              <Badge variant="outline" className="text-[10px] font-mono">
                {servicesLoading ? "..." : `${runningCount}/${services.length} services`}
              </Badge>
            </div>
            <CardDescription>Real-time resource utilization</CardDescription>
          </CardHeader>
          <CardContent className="pb-4">
            <EngineLoadBar
              cpu={engineCpu}
              memory={engineMem}
              workers={workerInfo}
              llmLoad={{ active: activeOpsCount > 0 ? activeOpsCount : 0, max: 4 }}
            />
          </CardContent>
        </Card>

        {/* Main Grid: System Health + Trace Stream */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* P1-3: Service Health Table */}
          <Card className="lg:col-span-2">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <Shield className="h-5 w-5" /> Service Health
                </CardTitle>
              </div>
              <CardDescription>
                All {services.length} services — categorized by function
              </CardDescription>
            </CardHeader>
            <CardContent>
              {servicesLoading && services.length === 0 ? (
                <div className="flex items-center justify-center py-6 text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin mr-2" /> Loading services...
                </div>
              ) : (
                <ServiceHealthTable services={categorizeServices(services)} compact />
              )}
            </CardContent>
          </Card>

          {/* P1-2: Trace Stream (enhanced ops feed) */}
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <Activity className="h-5 w-5" /> Trace Stream
                </CardTitle>
                {activeOpsCount > 0 && (
                  <Badge className="bg-blue-500 text-[10px] h-5">{activeOpsCount} active</Badge>
                )}
              </div>
              <CardDescription>Real-time operation events</CardDescription>
            </CardHeader>
            <CardContent>
              <TraceStream events={operations} loading={opsLoading} maxItems={7} showViewAll />
            </CardContent>
          </Card>
        </div>

        {/* P1-4: Fusion Strategy + Quick Access Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Fusion Strategy Mix */}
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <Workflow className="h-4 w-4" /> Fusion Strategy Mix
                </CardTitle>
                <Badge variant="outline" className="text-[10px]">
                  RRF merge
                </Badge>
              </div>
              <CardDescription>
                Multi-strategy retrieval fusion — semantic, BM25, graph, temporal
              </CardDescription>
            </CardHeader>
            <CardContent>
              <FusionStrategyMix compact />
            </CardContent>
          </Card>

          {/* Quick Access Panel Group */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-sm">
                <Zap className="h-4 w-4" /> Quick Access
              </CardTitle>
              <CardDescription>Jump to operator panels</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                <Link href="/backplane">
                  <div className="flex items-center gap-2 p-2 rounded border hover:bg-accent/50 transition-colors cursor-pointer">
                    <Network className="h-3.5 w-3.5 text-primary shrink-0" />
                    <span className="text-xs font-medium">Backplane</span>
                  </div>
                </Link>
                <Link href="/memories">
                  <div className="flex items-center gap-2 p-2 rounded border hover:bg-accent/50 transition-colors cursor-pointer">
                    <Brain className="h-3.5 w-3.5 text-primary shrink-0" />
                    <span className="text-xs font-medium">Memory Console</span>
                  </div>
                </Link>
                <Link href="/agents">
                  <div className="flex items-center gap-2 p-2 rounded border hover:bg-accent/50 transition-colors cursor-pointer">
                    <Bot className="h-3.5 w-3.5 text-primary shrink-0" />
                    <span className="text-xs font-medium">Agents</span>
                  </div>
                </Link>
                <Link href="/config">
                  <div className="flex items-center gap-2 p-2 rounded border hover:bg-accent/50 transition-colors cursor-pointer">
                    <Server className="h-3.5 w-3.5 text-primary shrink-0" />
                    <span className="text-xs font-medium">Model Config</span>
                  </div>
                </Link>
                <Link href="/evaluation">
                  <div className="flex items-center gap-2 p-2 rounded border hover:bg-accent/50 transition-colors cursor-pointer">
                    <BarChart3 className="h-3.5 w-3.5 text-primary shrink-0" />
                    <span className="text-xs font-medium">Eval Lab</span>
                  </div>
                </Link>
                <Link href="/runs">
                  <div className="flex items-center gap-2 p-2 rounded border hover:bg-accent/50 transition-colors cursor-pointer">
                    <Activity className="h-3.5 w-3.5 text-primary shrink-0" />
                    <span className="text-xs font-medium">Runs</span>
                  </div>
                </Link>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Bottom: Bank overview strip */}
        {banks.length > 0 && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-sm">
                <Database className="h-4 w-4" /> Configured Banks ({banks.length})
              </CardTitle>
              <CardDescription>Click to open bank management</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {banks.map((bank) => (
                  <Link key={bank.bank_id} href={`/banks/${bank.bank_id}`}>
                    <Badge
                      variant="outline"
                      className="text-[11px] py-1 px-2.5 hover:bg-accent/50 transition-colors cursor-pointer"
                    >
                      <Brain className="w-3 h-3 mr-1 text-primary" />
                      {bank.name || bank.bank_id}
                    </Badge>
                  </Link>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </OperatorShell>
  );
}

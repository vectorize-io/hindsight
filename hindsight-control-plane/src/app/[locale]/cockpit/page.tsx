"use client";

import { useState, useEffect } from "react";
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
  FileText,
  Users,
  Server,
} from "lucide-react";
import Link from "next/link";

interface ApiHealth {
  status: string;
  database: string;
}

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
    } catch {}
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
    } catch {}
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
    } catch {}
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

  const activeOps = operations.filter((o) => o.status === "processing" || o.status === "pending");
  const apiService = services.find((s) => s.name === "Hindsight API");
  const dbService = services.find((s) => s.name === "PostgreSQL");
  const runningServices = services.filter((s) => s.status === "running");

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
            <Button variant="outline" size="sm" onClick={() => { checkHealth(); loadBanks(); loadOperations(); }} disabled={servicesLoading}>
              <RefreshCw className={`h-4 w-4 mr-1 ${servicesLoading ? "animate-spin" : ""}`} />
              Refresh
            </Button>
          </div>
        </div>

        {/* Stat Cards Row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Brain className="h-4 w-4" /> Memory Banks
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {banksLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : banks.length}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Activity className="h-4 w-4" /> Active Ops
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-600">
                {opsLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : activeOps.length}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Server className="h-4 w-4" /> API Health
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2">
                {apiService?.health === "healthy" ? (
                  <CheckCircle2 className="h-5 w-5 text-green-500" />
                ) : (
                  <AlertCircle className="h-5 w-5 text-amber-500" />
                )}
                <span className="text-sm font-medium">
                  {apiService?.health || "unknown"}
                </span>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Bot className="h-4 w-4" /> Total Ops
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{operations.length}</div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Engine Health Panel */}
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Shield className="h-5 w-5" /> System Health
              </CardTitle>
              <CardDescription>
                API status and bank overview
                {apiVersion ? <span className="ml-2 text-muted-foreground">· v{apiVersion}</span> : null}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* API Health */}
                <div className="flex items-center justify-between p-3 rounded-lg border bg-card">
                  <div className="flex items-center gap-3">
                    <Server className="h-5 w-5 text-primary" />
                    <div>
                      <div className="text-sm font-medium">Hindsight API</div>
                      <div className="text-xs text-muted-foreground">
                        {apiVersion ? `v${apiVersion}` : "—"} · port {apiService?.port || 8888}
                      </div>
                    </div>
                  </div>
                  <Badge
                    variant="outline"
                    className={
                      apiService?.health === "healthy"
                        ? "border-green-300 text-green-700 bg-green-50"
                        : "border-red-300 text-red-700 bg-red-50"
                    }
                  >
                    {apiService?.health === "healthy" ? (
                      <><CheckCircle2 className="w-3 h-3 mr-1" /> Healthy</>
                    ) : (
                      <><AlertCircle className="w-3 h-3 mr-1" /> Down</>
                    )}
                  </Badge>
                </div>
                {/* Database */}
                <div className="flex items-center justify-between p-3 rounded-lg border bg-card">
                  <div className="flex items-center gap-3">
                    <Database className="h-5 w-5 text-primary" />
                    <div>
                      <div className="text-sm font-medium">Database</div>
                      <div className="text-xs text-muted-foreground">
                        PostgreSQL · :{dbService?.port || 5433}
                        {dbService?.uptime ? ` · up ${dbService.uptime}` : ""}
                      </div>
                    </div>
                  </div>
                  <Badge
                    variant="outline"
                    className={
                      dbService?.health === "connected"
                        ? "border-green-300 text-green-700 bg-green-50"
                        : "border-red-300 text-red-700 bg-red-50"
                    }
                  >
                    {dbService?.health === "connected" ? (
                      <><CheckCircle2 className="w-3 h-3 mr-1" /> Connected</>
                    ) : (
                      <><AlertCircle className="w-3 h-3 mr-1" /> Disconnected</>
                    )}
                  </Badge>
                </div>
                {/* Banks grid */}
                <div>
                  <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
                    Configured Banks ({banks.length})
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    {banks.length === 0 && !banksLoading && (
                      <p className="text-xs text-muted-foreground col-span-full">No banks configured</p>
                    )}
                    {banks.map((bank) => (
                      <Link key={bank.bank_id} href={`/banks/${bank.bank_id}`}>
                        <div className="flex items-center gap-2 p-2 rounded border hover:bg-accent/50 transition-colors cursor-pointer">
                          <Brain className="h-4 w-4 text-primary shrink-0" />
                          <div className="min-w-0">
                            <div className="text-xs font-medium truncate">
                              {bank.name || bank.bank_id}
                            </div>
                            <div className="text-[10px] text-muted-foreground truncate">{bank.bank_id}</div>
                          </div>
                        </div>
                      </Link>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Active Operations Feed */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Zap className="h-5 w-5" /> Operations Feed
              </CardTitle>
              <CardDescription>
                {activeOps.length > 0
                  ? `${activeOps.length} active · ${operations.length} total`
                  : "Recent system operations"}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {opsLoading ? (
                <div className="flex items-center justify-center py-6 text-muted-foreground">
                  <Loader2 className="h-5 w-5 animate-spin mr-2" /> Loading...
                </div>
              ) : operations.length === 0 ? (
                <div className="text-sm text-muted-foreground text-center py-6">No operations yet</div>
              ) : (
                operations.slice(0, 8).map((op) => (
                  <div key={op.id} className="p-3 rounded-lg border bg-card space-y-1.5">
                    <div className="flex items-center justify-between">
                      <Badge variant="outline" className="text-xs font-mono">
                        {op.task_type}
                      </Badge>
                      <span className="text-[10px] text-muted-foreground">
                        {new Date(op.created_at).toLocaleTimeString()}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-muted-foreground">
                        {op.items_count} items
                      </span>
                      {op.retry_count > 0 && (
                        <span className="text-[10px] text-amber-500">
                          (retry {op.retry_count})
                        </span>
                      )}
                    </div>
                    <div>
                      {op.status === "completed" ? (
                        <Badge className="bg-green-500 text-[10px] h-5">Completed</Badge>
                      ) : op.status === "failed" ? (
                        <Badge variant="destructive" className="text-[10px] h-5">
                          {op.error_message ? "Error" : "Failed"}
                        </Badge>
                      ) : op.status === "processing" ? (
                        <Badge className="bg-blue-500 text-[10px] h-5">
                          <Loader2 className="w-2.5 h-2.5 mr-1 animate-spin" /> Processing
                        </Badge>
                      ) : (
                        <Badge variant="outline" className="text-[10px] h-5">{op.status}</Badge>
                      )}
                    </div>
                    {op.error_message && (
                      <p className="text-[10px] text-red-500 truncate">{op.error_message}</p>
                    )}
                  </div>
                ))
              )}
              <div className="pt-2">
                <Link href="/runs">
                  <Button variant="ghost" size="sm" className="w-full text-xs">
                    View all operations <ArrowRight className="w-3 h-3 ml-1" />
                  </Button>
                </Link>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Quick Access Panel Group */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          <Link href="/backplane">
            <Card className="hover:bg-accent/50 transition-colors cursor-pointer h-full">
              <CardHeader className="pb-1 pt-3 px-3">
                <CardTitle className="text-xs font-medium flex items-center gap-1.5">
                  <Network className="h-3.5 w-3.5 text-primary" /> Backplane
                </CardTitle>
              </CardHeader>
              <CardContent className="text-[11px] text-muted-foreground px-3 pb-3">
                Agent orchestra
              </CardContent>
            </Card>
          </Link>
          <Link href="/memories">
            <Card className="hover:bg-accent/50 transition-colors cursor-pointer h-full">
              <CardHeader className="pb-1 pt-3 px-3">
                <CardTitle className="text-xs font-medium flex items-center gap-1.5">
                  <Brain className="h-3.5 w-3.5 text-primary" /> Memory Console
                </CardTitle>
              </CardHeader>
              <CardContent className="text-[11px] text-muted-foreground px-3 pb-3">
                Banks & facts
              </CardContent>
            </Card>
          </Link>
          <Link href="/agents">
            <Card className="hover:bg-accent/50 transition-colors cursor-pointer h-full">
              <CardHeader className="pb-1 pt-3 px-3">
                <CardTitle className="text-xs font-medium flex items-center gap-1.5">
                  <Bot className="h-3.5 w-3.5 text-primary" /> Agents
                </CardTitle>
              </CardHeader>
              <CardContent className="text-[11px] text-muted-foreground px-3 pb-3">
                Registry & management
              </CardContent>
            </Card>
          </Link>
          <Link href="/config">
            <Card className="hover:bg-accent/50 transition-colors cursor-pointer h-full">
              <CardHeader className="pb-1 pt-3 px-3">
                <CardTitle className="text-xs font-medium flex items-center gap-1.5">
                  <Server className="h-3.5 w-3.5 text-primary" /> Model Manager
                </CardTitle>
              </CardHeader>
              <CardContent className="text-[11px] text-muted-foreground px-3 pb-3">
                Providers & config
              </CardContent>
            </Card>
          </Link>
          <Link href="/evaluation">
            <Card className="hover:bg-accent/50 transition-colors cursor-pointer h-full">
              <CardHeader className="pb-1 pt-3 px-3">
                <CardTitle className="text-xs font-medium flex items-center gap-1.5">
                  <BarChart3 className="h-3.5 w-3.5 text-primary" /> Evaluation Lab
                </CardTitle>
              </CardHeader>
              <CardContent className="text-[11px] text-muted-foreground px-3 pb-3">
                Tests & benchmarks
              </CardContent>
            </Card>
          </Link>
          <Link href="/runs">
            <Card className="hover:bg-accent/50 transition-colors cursor-pointer h-full">
              <CardHeader className="pb-1 pt-3 px-3">
                <CardTitle className="text-xs font-medium flex items-center gap-1.5">
                  <Activity className="h-3.5 w-3.5 text-primary" /> Runs
                </CardTitle>
              </CardHeader>
              <CardContent className="text-[11px] text-muted-foreground px-3 pb-3">
                Execution timeline
              </CardContent>
            </Card>
          </Link>
        </div>
      </div>
    </OperatorShell>
  );
}

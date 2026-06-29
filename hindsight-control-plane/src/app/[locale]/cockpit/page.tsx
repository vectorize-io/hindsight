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
  HardDrive,
  Network,
  Shield,
  Bot,
  Clock,
  RefreshCw,
  ArrowRight,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Zap,
  BarChart3,
  MessageSquare,
  Search,
  Sparkles,
  Key,
} from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import Link from "next/link";

interface EngineHealth {
  name: string;
  status: "healthy" | "degraded" | "down";
  port: number;
  icon: React.ElementType;
  category: string;
}

interface ActiveRun {
  id: string;
  agent: string;
  intent: string;
  status: "processing" | "awaiting_approval" | "completed" | "failed";
  startedAt: string;
  progress?: string;
}

interface MemoryStats {
  totalFacts: number;
  totalBanks: number;
  activeOperations: number;
  lastIngest: string | null;
}

export default function CockpitPage() {
  const t = useTranslations("operator");

  // Engine health
  const [engineHealth, setEngineHealth] = useState<EngineHealth[]>([
    { name: "Hindsight API", status: "healthy", port: 8888, icon: Brain, category: "Memory" },
    { name: "Ollama Embeddings", status: "healthy", port: 11434, icon: Cpu, category: "AI" },
    { name: "PostgreSQL", status: "healthy", port: 5432, icon: Database, category: "Storage" },
    { name: "Qdrant", status: "healthy", port: 6233, icon: Database, category: "Storage" },
    { name: "Jaeger", status: "healthy", port: 16686, icon: Activity, category: "Observability" },
    { name: "Langfuse", status: "healthy", port: 3002, icon: MessageSquare, category: "LLM" },
    { name: "Memlord", status: "healthy", port: 8005, icon: Shield, category: "Memory" },
    { name: "Control Plane", status: "healthy", port: 9998, icon: Activity, category: "System" },
  ]);
  const [healthLoading, setHealthLoading] = useState(true);

  // Active runs
  const [activeRuns] = useState<ActiveRun[]>([
    {
      id: "run_001",
      agent: "Coding Agent",
      intent: "Refactor POF parser error handling",
      status: "awaiting_approval",
      startedAt: "2m ago",
    },
    {
      id: "run_002",
      agent: "Researcher",
      intent: "Analyze telemetry data patterns",
      status: "processing",
      startedAt: "5m ago",
      progress: "Vector search phase",
    },
    {
      id: "run_003",
      agent: "Memory Agent",
      intent: "Consolidate daily observations",
      status: "completed",
      startedAt: "12m ago",
    },
    {
      id: "run_004",
      agent: "Security Agent",
      intent: "Audit access control policies",
      status: "failed",
      startedAt: "18m ago",
      progress: "Authentication check failed",
    },
    {
      id: "run_005",
      agent: "Planner",
      intent: "Generate sprint roadmap",
      status: "processing",
      startedAt: "1m ago",
      progress: "Dependency analysis",
    },
    {
      id: "run_006",
      agent: "Tool Agent",
      intent: "Index documentation corpus",
      status: "completed",
      startedAt: "25m ago",
    },
  ]);

  // Memory stats
  const [memoryStats, setMemoryStats] = useState<MemoryStats>({
    totalFacts: 0,
    totalBanks: 0,
    activeOperations: 0,
    lastIngest: null,
  });
  const [statsLoading, setStatsLoading] = useState(true);

  // Check engine health
  const checkEngines = async () => {
    setHealthLoading(true);
    const updated = await Promise.all(
      engineHealth.map(async (engine) => {
        try {
          const res = await fetch(`http://localhost:${engine.port}/health`, {
            signal: AbortSignal.timeout(3000),
          });
          return { ...engine, status: res.ok ? ("healthy" as const) : ("degraded" as const) };
        } catch {
          return { ...engine, status: "down" as const };
        }
      })
    );
    setEngineHealth(updated);
    setHealthLoading(false);
  };

  // Load memory stats
  const loadStats = async () => {
    setStatsLoading(true);
    try {
      const res = await fetch("/api/system/operations?limit=5");
      const data = await res.json();
      const ops = data.operations || [];
      setMemoryStats((prev) => ({
        ...prev,
        activeOperations: ops.filter(
          (o: any) => o.status === "processing" || o.status === "pending"
        ).length,
      }));

      // Try to get bank stats
      const banksRes = await fetch("/api/banks");
      const banksData = await banksRes.json();
      if (banksData.banks) {
        setMemoryStats((prev) => ({
          ...prev,
          totalBanks: banksData.banks.length,
        }));
      }
    } catch (e) {
      console.error("Failed to load stats:", e);
    } finally {
      setStatsLoading(false);
    }
  };

  useEffect(() => {
    checkEngines();
    loadStats();

    // Poll health every 30s
    const interval = setInterval(checkEngines, 30000);
    return () => clearInterval(interval);
  }, []);

  const statusColor = (status: string) => {
    switch (status) {
      case "healthy":
        return "text-green-500 bg-green-50 dark:bg-green-950/30 border-green-200 dark:border-green-800";
      case "degraded":
        return "text-amber-500 bg-amber-50 dark:bg-amber-950/30 border-amber-200 dark:border-amber-800";
      case "down":
        return "text-red-500 bg-red-50 dark:bg-red-950/30 border-red-200 dark:border-red-800";
      default:
        return "text-muted-foreground";
    }
  };

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
            <Button variant="outline" size="sm" onClick={checkEngines} disabled={healthLoading}>
              <RefreshCw className={`h-4 w-4 mr-1 ${healthLoading ? "animate-spin" : ""}`} />
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
                {statsLoading ? (
                  <Loader2 className="h-5 w-5 animate-spin" />
                ) : (
                  memoryStats.totalBanks
                )}
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
                {statsLoading ? (
                  <Loader2 className="h-5 w-5 animate-spin" />
                ) : (
                  memoryStats.activeOperations
                )}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Cpu className="h-4 w-4" /> Engines
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">
                {engineHealth.filter((e) => e.status === "healthy").length}/{engineHealth.length}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Bot className="h-4 w-4" /> Agents
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{activeRuns.length}</div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Engine Health Panel */}
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Shield className="h-5 w-5" /> Engine Health
              </CardTitle>
              <CardDescription>Real-time status of all CollabMind engines</CardDescription>
            </CardHeader>
            <CardContent>
              {healthLoading && engineHealth.every((e) => e.status === "healthy") ? (
                <div className="flex items-center justify-center py-8 text-muted-foreground">
                  <Loader2 className="h-5 w-5 animate-spin mr-2" /> Checking engines...
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {engineHealth.map((engine) => {
                    const Icon = engine.icon;
                    return (
                      <div
                        key={engine.name}
                        className={`flex items-center justify-between p-3 rounded-lg border ${statusColor(engine.status)}`}
                      >
                        <div className="flex items-center gap-3">
                          <Icon className="h-5 w-5" />
                          <div>
                            <div className="text-sm font-medium">{engine.name}</div>
                            <div className="text-xs text-muted-foreground">
                              {engine.category} · :{engine.port}
                            </div>
                          </div>
                        </div>
                        <Badge
                          variant="outline"
                          className={
                            engine.status === "healthy"
                              ? "border-green-300 text-green-700 bg-green-50"
                              : engine.status === "degraded"
                                ? "border-amber-300 text-amber-700 bg-amber-50"
                                : "border-red-300 text-red-700 bg-red-50"
                          }
                        >
                          {engine.status === "healthy" ? (
                            <>
                              <CheckCircle2 className="w-3 h-3 mr-1" /> Healthy
                            </>
                          ) : engine.status === "degraded" ? (
                            <>
                              <AlertCircle className="w-3 h-3 mr-1" /> Degraded
                            </>
                          ) : (
                            <>
                              <AlertCircle className="w-3 h-3 mr-1" /> Down
                            </>
                          )}
                        </Badge>
                      </div>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Active Runs / Quick Actions */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Zap className="h-5 w-5" /> Active Runs
              </CardTitle>
              <CardDescription>Recent agent execution activity</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {activeRuns.length === 0 ? (
                <div className="text-sm text-muted-foreground text-center py-6">No active runs</div>
              ) : (
                activeRuns.map((run) => (
                  <div key={run.id} className="p-3 rounded-lg border bg-card space-y-2">
                    <div className="flex items-center justify-between">
                      <Badge variant="outline" className="text-xs">
                        {run.agent}
                      </Badge>
                      <span className="text-xs text-muted-foreground">{run.startedAt}</span>
                    </div>
                    <p className="text-sm font-medium">{run.intent}</p>
                    <div className="flex items-center gap-2">
                      {run.status === "processing" && (
                        <Badge className="bg-blue-500 text-xs">
                          <Loader2 className="w-3 h-3 mr-1 animate-spin" /> Processing
                        </Badge>
                      )}
                      {run.status === "awaiting_approval" && (
                        <Badge className="bg-amber-500 text-xs">
                          <AlertCircle className="w-3 h-3 mr-1" /> Awaiting Approval
                        </Badge>
                      )}
                      {run.status === "completed" && (
                        <Badge className="bg-green-500 text-xs">Completed</Badge>
                      )}
                      {run.status === "failed" && (
                        <Badge variant="destructive" className="text-xs">
                          Failed
                        </Badge>
                      )}
                    </div>
                    {run.progress && (
                      <p className="text-xs text-muted-foreground">{run.progress}</p>
                    )}
                  </div>
                ))
              )}

              <div className="pt-2">
                <Link href="/runs">
                  <Button variant="ghost" size="sm" className="w-full text-xs">
                    View all runs <ArrowRight className="w-3 h-3 ml-1" />
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
                  <Key className="h-3.5 w-3.5 text-primary" /> Model Manager
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

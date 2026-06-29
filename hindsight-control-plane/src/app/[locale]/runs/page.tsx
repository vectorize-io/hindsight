"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { OperatorShell } from "@/components/operator-shell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Activity,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Clock,
  Search,
  Filter,
  Bot,
  Zap,
  RefreshCw,
} from "lucide-react";

interface Run {
  id: string;
  agent: string;
  intent: string;
  status: "processing" | "awaiting_approval" | "completed" | "failed";
  startedAt: string;
  duration: string;
  steps: number;
}

export default function RunsPage() {
  const t = useTranslations("operator");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState("");

  const runs: Run[] = [
    { id: "run_001", agent: "Coding Agent", intent: "Refactor POF parser error handling", status: "awaiting_approval", startedAt: "2m ago", duration: "45s", steps: 4 },
    { id: "run_002", agent: "Researcher", intent: "Analyze telemetry data patterns", status: "processing", startedAt: "5m ago", duration: "2m 30s", steps: 7 },
    { id: "run_003", agent: "Memory Agent", intent: "Consolidate daily observations", status: "completed", startedAt: "12m ago", duration: "1m 15s", steps: 3 },
    { id: "run_004", agent: "Security Agent", intent: "Audit access control policies", status: "failed", startedAt: "18m ago", duration: "3m", steps: 2 },
    { id: "run_005", agent: "Planner", intent: "Generate sprint roadmap", status: "processing", startedAt: "1m ago", duration: "30s", steps: 5 },
    { id: "run_006", agent: "Tool Agent", intent: "Index documentation corpus", status: "completed", startedAt: "25m ago", duration: "4m 10s", steps: 6 },
    { id: "run_007", agent: "Evaluator", intent: "Run evaluation suite v2.3", status: "completed", startedAt: "35m ago", duration: "2m 45s", steps: 8 },
    { id: "run_008", agent: "Coding Agent", intent: "Fix Tool Agent connection timeout", status: "processing", startedAt: "30s ago", duration: "15s", steps: 3 },
    { id: "run_009", agent: "Orchestrator", intent: "Route pending security audit task", status: "completed", startedAt: "42m ago", duration: "8s", steps: 2 },
    { id: "run_010", agent: "Memory Agent", intent: "Purge expired cache entries", status: "completed", startedAt: "50m ago", duration: "22s", steps: 2 },
    { id: "run_011", agent: "Researcher", intent: "Cross-reference findings with knowledge graph", status: "failed", startedAt: "55m ago", duration: "1m 5s", steps: 3 },
    { id: "run_012", agent: "Planner", intent: "Optimize worker allocation strategy", status: "completed", startedAt: "1h ago", duration: "55s", steps: 4 },
  ];

  const totalToday = runs.length;
  const successCount = runs.filter(r => r.status === "completed").length;
  const failedCount = runs.filter(r => r.status === "failed").length;
  const successRate = totalToday > 0 ? (successCount / totalToday) * 100 : 0;

  const filtered = runs.filter(r => {
    if (statusFilter !== "all" && r.status !== statusFilter) return false;
    if (searchQuery && !r.intent.toLowerCase().includes(searchQuery.toLowerCase()) && !r.agent.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  });

  const statusBadge = (status: string) => {
    switch (status) {
      case "processing": return <Badge className="bg-blue-500 text-[10px] h-5"><Loader2 className="w-2.5 h-2.5 mr-1 animate-spin" />Processing</Badge>;
      case "awaiting_approval": return <Badge className="bg-amber-500 text-[10px] h-5"><AlertCircle className="w-2.5 h-2.5 mr-1" />Awaiting Approval</Badge>;
      case "completed": return <Badge variant="outline" className="border-green-300 text-green-700 bg-green-50 dark:border-green-800 dark:text-green-400 dark:bg-green-950/30 text-[10px] h-5"><CheckCircle2 className="w-2.5 h-2.5 mr-1" />Completed</Badge>;
      case "failed": return <Badge variant="destructive" className="text-[10px] h-5">Failed</Badge>;
      default: return null;
    }
  };

  return (
    <OperatorShell>
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight flex items-center gap-3">
              <Activity className="h-6 w-6 text-primary" />
              {t("runs.title")}
            </h1>
            <p className="text-sm text-muted-foreground mt-1">{t("descriptions.runs")}</p>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2"><Activity className="h-4 w-4" /> {t("runs.statTotalToday")}</CardTitle>
            </CardHeader>
            <CardContent><div className="text-2xl font-bold">{totalToday}</div></CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2"><CheckCircle2 className="h-4 w-4" /> {t("runs.statSuccessRate")}</CardTitle>
            </CardHeader>
            <CardContent><div className="text-2xl font-bold text-green-600">{successRate.toFixed(0)}%</div></CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2"><Clock className="h-4 w-4" /> {t("runs.statAvgDuration")}</CardTitle>
            </CardHeader>
            <CardContent><div className="text-2xl font-bold">1m 12s</div></CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2"><AlertCircle className="h-4 w-4" /> {t("runs.statFailed")}</CardTitle>
            </CardHeader>
            <CardContent><div className="text-2xl font-bold text-red-600">{failedCount}</div></CardContent>
          </Card>
        </div>

        {/* Filters */}
        <div className="flex items-center gap-3 flex-wrap">
          <div className="relative flex-1 max-w-sm">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <input type="text" placeholder={t("runs.searchRuns")} className="w-full h-9 pl-9 pr-3 rounded-lg border bg-background text-sm" value={searchQuery} onChange={e => setSearchQuery(e.target.value)} />
          </div>
          <div className="flex gap-1">
            {["all", "processing", "completed", "failed", "awaiting_approval"].map(s => (
              <Button key={s} variant={statusFilter === s ? "default" : "outline"} size="sm" className="text-xs h-8" onClick={() => setStatusFilter(s)}>
                {s === "all" ? "All" : s.charAt(0).toUpperCase() + s.slice(1).replace("_", " ")}
              </Button>
            ))}
          </div>
        </div>

        {/* Timeline */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2"><Zap className="h-5 w-5 text-primary" /> Execution Timeline</CardTitle>
            <CardDescription>Run history sorted by recency</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-1">
              {filtered.map((run, idx) => (
                <div key={run.id} className={`flex items-center gap-4 p-3 rounded-lg hover:bg-accent/30 transition-colors ${idx < filtered.length - 1 ? "border-b border-border/50" : ""}`}>
                  {/* Timeline dot */}
                  <div className="flex flex-col items-center">
                    <div className={`w-2.5 h-2.5 rounded-full ${
                      run.status === "completed" ? "bg-green-500" :
                      run.status === "failed" ? "bg-red-500" :
                      run.status === "processing" ? "bg-blue-500 animate-pulse" :
                      "bg-amber-500"
                    }`} />
                    {idx < filtered.length - 1 && <div className="w-px h-6 bg-border mt-1" />}
                  </div>
                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium truncate">{run.intent}</span>
                      {statusBadge(run.status)}
                    </div>
                    <div className="flex items-center gap-3 text-xs text-muted-foreground mt-0.5">
                      <span className="flex items-center gap-1"><Bot className="w-3 h-3" />{run.agent}</span>
                      <span>{run.startedAt}</span>
                      <span className="font-mono">{run.duration}</span>
                      <span>{run.steps} steps</span>
                    </div>
                  </div>
                  {run.status === "awaiting_approval" && (
                    <div className="flex gap-1">
                      <Button size="sm" variant="outline" className="text-xs h-7">Review</Button>
                      <Button size="sm" variant="default" className="text-xs h-7">Approve</Button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </OperatorShell>
  );
}

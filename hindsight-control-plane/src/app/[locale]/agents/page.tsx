"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { OperatorShell } from "@/components/operator-shell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Bot,
  Activity,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Clock,
  Zap,
  Plus,
  Search,
  Cpu,
  Users,
  FileText,
  RefreshCw,
} from "lucide-react";

interface Agent {
  id: string;
  name: string;
  role: string;
  status: "active" | "idle" | "error" | "offline";
  model: string;
  tasksCompleted: number;
  tasksFailed: number;
  uptime: string;
  lastActive: string;
}

export default function AgentsPage() {
  const t = useTranslations("operator");
  const [searchQuery, setSearchQuery] = useState("");

  const [agents] = useState<Agent[]>([
    {
      id: "agent_001",
      name: "Orchestrator",
      role: "Task Routing & Coordination",
      status: "active",
      model: "gpt-5-mini",
      tasksCompleted: 1427,
      tasksFailed: 3,
      uptime: "12d 4h",
      lastActive: "Just now",
    },
    {
      id: "agent_002",
      name: "Coding Agent",
      role: "Implementation & Refactoring",
      status: "active",
      model: "claude-sonnet-4",
      tasksCompleted: 843,
      tasksFailed: 12,
      uptime: "8d 2h",
      lastActive: "30s ago",
    },
    {
      id: "agent_003",
      name: "Researcher",
      role: "Information Retrieval & Analysis",
      status: "active",
      model: "gpt-5-mini",
      tasksCompleted: 621,
      tasksFailed: 5,
      uptime: "6d 18h",
      lastActive: "1m ago",
    },
    {
      id: "agent_004",
      name: "Memory Agent",
      role: "Memory Consolidation & Retrieval",
      status: "active",
      model: "gemini-2.0-flash",
      tasksCompleted: 2354,
      tasksFailed: 1,
      uptime: "14d 0h",
      lastActive: "Just now",
    },
    {
      id: "agent_005",
      name: "Security Agent",
      role: "Access Control & Audit",
      status: "idle",
      model: "gpt-5-mini",
      tasksCompleted: 189,
      tasksFailed: 8,
      uptime: "4d 12h",
      lastActive: "25m ago",
    },
    {
      id: "agent_006",
      name: "Planner",
      role: "Task Decomposition & Scheduling",
      status: "active",
      model: "claude-sonnet-4",
      tasksCompleted: 456,
      tasksFailed: 2,
      uptime: "7d 6h",
      lastActive: "2m ago",
    },
    {
      id: "agent_007",
      name: "Tool Agent",
      role: "External Tool Execution",
      status: "error",
      model: "gpt-5-mini",
      tasksCompleted: 312,
      tasksFailed: 28,
      uptime: "3d 9h",
      lastActive: "5m ago",
    },
    {
      id: "agent_008",
      name: "Evaluator",
      role: "Output Validation & Scoring",
      status: "idle",
      model: "gpt-5-mini",
      tasksCompleted: 567,
      tasksFailed: 15,
      uptime: "5d 22h",
      lastActive: "15m ago",
    },
  ]);

  const activeCount = agents.filter((a) => a.status === "active").length;
  const totalTasks = agents.reduce((s, a) => s + a.tasksCompleted, 0);
  const errorRate = agents.reduce((s, a) => s + a.tasksFailed, 0) / Math.max(totalTasks, 1);

  const filtered = agents.filter(
    (a) =>
      a.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      a.role.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const statusBadge = (status: string) => {
    switch (status) {
      case "active":
        return (
          <Badge
            variant="outline"
            className="border-green-300 text-green-700 bg-green-50 dark:border-green-800 dark:text-green-400 dark:bg-green-950/30 text-[10px] h-5"
          >
            <CheckCircle2 className="w-2.5 h-2.5 mr-1" />
            Active
          </Badge>
        );
      case "idle":
        return (
          <Badge variant="outline" className="text-[10px] h-5">
            Idle
          </Badge>
        );
      case "error":
        return (
          <Badge
            variant="outline"
            className="border-red-300 text-red-700 bg-red-50 dark:border-red-800 dark:text-red-400 dark:bg-red-950/30 text-[10px] h-5"
          >
            <AlertCircle className="w-2.5 h-2.5 mr-1" />
            Error
          </Badge>
        );
      case "offline":
        return (
          <Badge variant="outline" className="text-[10px] h-5 text-muted-foreground">
            Offline
          </Badge>
        );
      default:
        return null;
    }
  };

  return (
    <OperatorShell>
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight flex items-center gap-3">
              <Bot className="h-6 w-6 text-primary" />
              {t("agents.title")}
            </h1>
            <p className="text-sm text-muted-foreground mt-1">{t("descriptions.agents")}</p>
          </div>
          <Button size="sm" className="gap-1">
            <Plus className="h-4 w-4" /> {t("agents.createAgent")}
          </Button>
        </div>

        {/* Stat cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Users className="h-4 w-4" /> {t("agents.statTotal")}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{agents.length}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Activity className="h-4 w-4" /> {t("agents.statActive")}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">{activeCount}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Zap className="h-4 w-4" /> {t("agents.statTasks")}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{totalTasks.toLocaleString()}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <AlertCircle className="h-4 w-4" /> {t("agents.statErrorRate")}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{(errorRate * 100).toFixed(1)}%</div>
            </CardContent>
          </Card>
        </div>

        {/* Search */}
        <div className="flex items-center gap-3">
          <div className="relative flex-1 max-w-sm">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <input
              type="text"
              placeholder={t("agents.searchAgents")}
              className="w-full h-9 pl-9 pr-3 rounded-lg border bg-background text-sm"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          <Button variant="outline" size="sm">
            <RefreshCw className="h-4 w-4 mr-1" /> Refresh
          </Button>
        </div>

        {/* Agent cards grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.map((agent) => {
            const errorPct =
              agent.tasksCompleted > 0
                ? (agent.tasksFailed / (agent.tasksCompleted + agent.tasksFailed)) * 100
                : 0;
            return (
              <Card key={agent.id} className="hover:bg-accent/30 transition-colors">
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <div
                        className={`p-2 rounded-lg ${
                          agent.status === "active"
                            ? "bg-green-100 dark:bg-green-950/30"
                            : agent.status === "error"
                              ? "bg-red-100 dark:bg-red-950/30"
                              : "bg-muted"
                        }`}
                      >
                        <Bot
                          className={`h-5 w-5 ${
                            agent.status === "active"
                              ? "text-green-600"
                              : agent.status === "error"
                                ? "text-red-600"
                                : "text-muted-foreground"
                          }`}
                        />
                      </div>
                      <div>
                        <CardTitle className="text-sm font-medium">{agent.name}</CardTitle>
                        <CardDescription className="text-xs">{agent.role}</CardDescription>
                      </div>
                    </div>
                    {statusBadge(agent.status)}
                  </div>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="flex items-center gap-1 text-muted-foreground">
                      <Cpu className="w-3 h-3" /> Model
                    </div>
                    <div className="font-mono text-right truncate">{agent.model}</div>
                    <div className="flex items-center gap-1 text-muted-foreground">
                      <Zap className="w-3 h-3" /> Tasks
                    </div>
                    <div className="text-right">{agent.tasksCompleted.toLocaleString()}</div>
                    <div className="flex items-center gap-1 text-muted-foreground">
                      <Clock className="w-3 h-3" /> Uptime
                    </div>
                    <div className="text-right">{agent.uptime}</div>
                    <div className="flex items-center gap-1 text-muted-foreground">
                      <AlertCircle className="w-3 h-3" /> Error Rate
                    </div>
                    <div className="text-right">{errorPct.toFixed(1)}%</div>
                  </div>
                  {/* Success bar */}
                  <div className="w-full bg-muted rounded-full h-1.5">
                    <div
                      className={`h-1.5 rounded-full ${
                        errorPct > 10
                          ? "bg-red-500"
                          : errorPct > 3
                            ? "bg-amber-500"
                            : "bg-green-500"
                      }`}
                      style={{ width: `${Math.max(100 - errorPct, 0)}%` }}
                    />
                  </div>
                  <div className="text-xs text-muted-foreground text-right">
                    Last active: {agent.lastActive}
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>
    </OperatorShell>
  );
}

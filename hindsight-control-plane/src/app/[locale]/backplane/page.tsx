"use client";

import { useState, useEffect } from "react";
import { useTranslations } from "next-intl";
import { OperatorShell } from "@/components/operator-shell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Network,
  Bot,
  Cpu,
  Activity,
  MessageSquare,
  Zap,
  RefreshCw,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Clock,
  ArrowRight,
  Server,
  Radio,
  Globe,
  Workflow,
  Users,
  Eye,
} from "lucide-react";
import Link from "next/link";

interface AgentNode {
  id: string;
  name: string;
  role: string;
  status: "active" | "idle" | "error";
  tasksCompleted: number;
  uptime: string;
  model: string;
}

interface BusMessage {
  id: string;
  from: string;
  to: string;
  type: string;
  timestamp: string;
  status: "delivered" | "pending" | "failed";
}

interface TaskQueueItem {
  id: string;
  intent: string;
  assignedTo: string | null;
  priority: "high" | "medium" | "low";
  status: "queued" | "processing" | "completed" | "failed";
  createdAt: string;
}

export default function BackplanePage() {
  const t = useTranslations("operator");

  const [agents, setAgents] = useState<AgentNode[]>([
    {
      id: "agent_001",
      name: "Orchestrator",
      role: "Task Routing & Coordination",
      status: "active",
      tasksCompleted: 1427,
      uptime: "12d 4h",
      model: "gpt-5-mini",
    },
    {
      id: "agent_002",
      name: "Coding Agent",
      role: "Implementation & Refactoring",
      status: "active",
      tasksCompleted: 843,
      uptime: "8d 2h",
      model: "claude-sonnet-4",
    },
    {
      id: "agent_003",
      name: "Researcher",
      role: "Information Retrieval & Analysis",
      status: "active",
      tasksCompleted: 621,
      uptime: "6d 18h",
      model: "gpt-5-mini",
    },
    {
      id: "agent_004",
      name: "Memory Agent",
      role: "Memory Consolidation & Retrieval",
      status: "active",
      tasksCompleted: 2354,
      uptime: "14d 0h",
      model: "gemini-2.0-flash",
    },
    {
      id: "agent_005",
      name: "Security Agent",
      role: "Access Control & Audit",
      status: "idle",
      tasksCompleted: 189,
      uptime: "4d 12h",
      model: "gpt-5-mini",
    },
    {
      id: "agent_006",
      name: "Planner",
      role: "Task Decomposition & Scheduling",
      status: "active",
      tasksCompleted: 456,
      uptime: "7d 6h",
      model: "claude-sonnet-4",
    },
    {
      id: "agent_007",
      name: "Tool Agent",
      role: "External Tool Execution",
      status: "error",
      tasksCompleted: 312,
      uptime: "3d 9h",
      model: "gpt-5-mini",
    },
    {
      id: "agent_008",
      name: "Evaluator",
      role: "Output Validation & Scoring",
      status: "idle",
      tasksCompleted: 567,
      uptime: "5d 22h",
      model: "gpt-5-mini",
    },
  ]);

  const [messages, setMessages] = useState<BusMessage[]>([
    {
      id: "msg_01",
      from: "Orchestrator",
      to: "Coding Agent",
      type: "delegate",
      timestamp: "30s ago",
      status: "delivered",
    },
    {
      id: "msg_02",
      from: "Coding Agent",
      to: "Tool Agent",
      type: "execute",
      timestamp: "45s ago",
      status: "delivered",
    },
    {
      id: "msg_03",
      from: "Researcher",
      to: "Memory Agent",
      type: "store",
      timestamp: "1m ago",
      status: "delivered",
    },
    {
      id: "msg_04",
      from: "Planner",
      to: "Orchestrator",
      type: "report",
      timestamp: "2m ago",
      status: "delivered",
    },
    {
      id: "msg_05",
      from: "Orchestrator",
      to: "Security Agent",
      type: "inspect",
      timestamp: "3m ago",
      status: "pending",
    },
    {
      id: "msg_06",
      from: "Tool Agent",
      to: "Evaluator",
      type: "result",
      timestamp: "4m ago",
      status: "failed",
    },
    {
      id: "msg_07",
      from: "Memory Agent",
      to: "Orchestrator",
      type: "ack",
      timestamp: "5m ago",
      status: "delivered",
    },
    {
      id: "msg_08",
      from: "Evaluator",
      to: "Coding Agent",
      type: "feedback",
      timestamp: "6m ago",
      status: "delivered",
    },
  ]);

  const [tasks, setTasks] = useState<TaskQueueItem[]>([
    {
      id: "task_01",
      intent: "Refactor authentication middleware",
      assignedTo: "Coding Agent",
      priority: "high",
      status: "processing",
      createdAt: "1m ago",
    },
    {
      id: "task_02",
      intent: "Analyze telemetry spike patterns",
      assignedTo: "Researcher",
      priority: "medium",
      status: "processing",
      createdAt: "3m ago",
    },
    {
      id: "task_03",
      intent: "Consolidate daily memory observations",
      assignedTo: "Memory Agent",
      priority: "medium",
      status: "processing",
      createdAt: "5m ago",
    },
    {
      id: "task_04",
      intent: "Audit API key rotation compliance",
      assignedTo: "Security Agent",
      priority: "high",
      status: "queued",
      createdAt: "2m ago",
    },
    {
      id: "task_05",
      intent: "Generate sprint roadmap",
      assignedTo: "Planner",
      priority: "medium",
      status: "queued",
      createdAt: "4m ago",
    },
    {
      id: "task_06",
      intent: "Index new documentation corpus",
      assignedTo: null,
      priority: "low",
      status: "queued",
      createdAt: "10m ago",
    },
    {
      id: "task_07",
      intent: "Run evaluation suite v2.3",
      assignedTo: "Evaluator",
      priority: "low",
      status: "completed",
      createdAt: "15m ago",
    },
    {
      id: "task_08",
      intent: "Fix Tool Agent connection timeout",
      assignedTo: "Coding Agent",
      priority: "high",
      status: "queued",
      createdAt: "30s ago",
    },
  ]);

  const [activeTab, setActiveTab] = useState("topology");

  const activeAgentCount = agents.filter((a) => a.status === "active").length;
  const totalTasksToday = tasks.length;
  const messagesDelivered = messages.filter((m) => m.status === "delivered").length;

  const agentStatusColor = (status: string) => {
    switch (status) {
      case "active":
        return "text-green-500 bg-green-50 dark:bg-green-950/30 border-green-200 dark:border-green-800";
      case "idle":
        return "text-muted-foreground bg-muted border-border";
      case "error":
        return "text-red-500 bg-red-50 dark:bg-red-950/30 border-red-200 dark:border-red-800";
      default:
        return "text-muted-foreground";
    }
  };

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
      case "queued":
        return (
          <Badge variant="outline" className="text-[10px] h-5">
            <Clock className="w-2.5 h-2.5 mr-1" />
            Queued
          </Badge>
        );
      case "processing":
        return (
          <Badge className="bg-blue-500 text-[10px] h-5">
            <Loader2 className="w-2.5 h-2.5 mr-1 animate-spin" />
            Processing
          </Badge>
        );
      case "completed":
        return (
          <Badge
            variant="outline"
            className="border-green-300 text-green-700 bg-green-50 text-[10px] h-5"
          >
            Completed
          </Badge>
        );
      case "failed":
        return (
          <Badge variant="destructive" className="text-[10px] h-5">
            Failed
          </Badge>
        );
      case "delivered":
        return <CheckCircle2 className="w-3 h-3 text-green-500" />;
      case "pending":
        return <Clock className="w-3 h-3 text-amber-500" />;
      default:
        return null;
    }
  };

  const priorityBadge = (p: string) => {
    switch (p) {
      case "high":
        return (
          <Badge
            variant="outline"
            className="border-red-300 text-red-700 bg-red-50 dark:border-red-800 dark:text-red-400 dark:bg-red-950/30 text-[10px] h-5"
          >
            High
          </Badge>
        );
      case "medium":
        return (
          <Badge
            variant="outline"
            className="border-amber-300 text-amber-700 bg-amber-50 dark:border-amber-800 dark:text-amber-400 dark:bg-amber-950/30 text-[10px] h-5"
          >
            Medium
          </Badge>
        );
      case "low":
        return (
          <Badge variant="outline" className="text-[10px] h-5">
            Low
          </Badge>
        );
      default:
        return null;
    }
  };

  return (
    <OperatorShell>
      <div className="p-6 space-y-6">
        {/* Page header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight flex items-center gap-3">
              <Network className="h-6 w-6 text-primary" />
              {t("backplane.title")}
            </h1>
            <p className="text-sm text-muted-foreground mt-1">{t("descriptions.backplane")}</p>
          </div>
          <Button variant="outline" size="sm" onClick={() => {}}>
            <RefreshCw className="h-4 w-4 mr-1" /> Refresh
          </Button>
        </div>

        {/* Stat cards row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Bot className="h-4 w-4" /> {t("backplane.statTotalAgents")}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{agents.length}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Activity className="h-4 w-4" /> {t("backplane.statActiveAgents")}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">{activeAgentCount}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Zap className="h-4 w-4" /> {t("backplane.statTasksToday")}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{totalTasksToday}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <MessageSquare className="h-4 w-4" /> {t("backplane.statAvgResponseTime")}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">1.2s</div>
            </CardContent>
          </Card>
        </div>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
          <TabsList>
            <TabsTrigger value="topology" className="flex items-center gap-2">
              <Workflow className="h-4 w-4" /> Orchestration Topology
            </TabsTrigger>
            <TabsTrigger value="agents" className="flex items-center gap-2">
              <Bot className="h-4 w-4" /> Agent Registry
            </TabsTrigger>
            <TabsTrigger value="bus" className="flex items-center gap-2">
              <Radio className="h-4 w-4" /> Message Bus
            </TabsTrigger>
            <TabsTrigger value="tasks" className="flex items-center gap-2">
              <Zap className="h-4 w-4" /> Task Queue
            </TabsTrigger>
          </TabsList>

          {/* Topology Tab */}
          <TabsContent value="topology" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Globe className="h-5 w-5 text-primary" /> Agent Topology
                </CardTitle>
                <CardDescription>
                  Live orchestration graph showing agent communication pathways
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  {[
                    {
                      name: "Orchestrator",
                      role: "Central Router",
                      connections: 7,
                      color: "border-blue-400 bg-blue-50 dark:bg-blue-950/30",
                    },
                    {
                      name: "Coding Agent",
                      role: "Implementation",
                      connections: 4,
                      color: "border-emerald-400 bg-emerald-50 dark:bg-emerald-950/30",
                    },
                    {
                      name: "Researcher",
                      role: "Analysis",
                      connections: 3,
                      color: "border-violet-400 bg-violet-50 dark:bg-violet-950/30",
                    },
                    {
                      name: "Memory Agent",
                      role: "Storage & Recall",
                      connections: 5,
                      color: "border-amber-400 bg-amber-50 dark:bg-amber-950/30",
                    },
                    {
                      name: "Security Agent",
                      role: "Guard",
                      connections: 2,
                      color: "border-red-400 bg-red-50 dark:bg-red-950/30",
                    },
                    {
                      name: "Planner",
                      role: "Scheduler",
                      connections: 3,
                      color: "border-cyan-400 bg-cyan-50 dark:bg-cyan-950/30",
                    },
                    {
                      name: "Tool Agent",
                      role: "External Executor",
                      connections: 4,
                      color: "border-orange-400 bg-orange-50 dark:bg-orange-950/30",
                    },
                    {
                      name: "Evaluator",
                      role: "Validator",
                      connections: 3,
                      color: "border-pink-400 bg-pink-50 dark:bg-pink-950/30",
                    },
                  ].map((node) => (
                    <div
                      key={node.name}
                      className={`p-4 rounded-lg border-2 ${node.color} relative`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-bold">{node.name}</span>
                        <Badge variant="outline" className="text-[10px]">
                          {node.role}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <Server className="w-3 h-3" />
                        <span>{node.connections} connections</span>
                      </div>
                      {/* Animated pulse for active nodes */}
                      {node.name !== "Tool Agent" && (
                        <span className="absolute top-2 right-2 flex h-2 w-2">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                          <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                        </span>
                      )}
                      {node.name === "Tool Agent" && (
                        <span className="absolute top-2 right-2 flex h-2 w-2">
                          <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Agents Tab */}
          <TabsContent value="agents" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Users className="h-5 w-5 text-primary" /> {t("backplane.agentRegistry")}
                </CardTitle>
                <CardDescription>All registered agents and their current status</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b text-left text-xs text-muted-foreground">
                        <th className="pb-3 font-medium">Agent</th>
                        <th className="pb-3 font-medium">Role</th>
                        <th className="pb-3 font-medium">Model</th>
                        <th className="pb-3 font-medium">Tasks</th>
                        <th className="pb-3 font-medium">Uptime</th>
                        <th className="pb-3 font-medium">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {agents.map((agent) => (
                        <tr key={agent.id} className="border-b last:border-0 hover:bg-accent/30">
                          <td className="py-3 flex items-center gap-2">
                            <Bot className="h-4 w-4 text-primary" />
                            <span className="font-medium">{agent.name}</span>
                          </td>
                          <td className="py-3 text-muted-foreground">{agent.role}</td>
                          <td className="py-3 font-mono text-xs">{agent.model}</td>
                          <td className="py-3">{agent.tasksCompleted.toLocaleString()}</td>
                          <td className="py-3 text-muted-foreground">{agent.uptime}</td>
                          <td className="py-3">{statusBadge(agent.status)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Message Bus Tab */}
          <TabsContent value="bus" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Radio className="h-5 w-5 text-primary" /> {t("backplane.messageBus")}
                </CardTitle>
                <CardDescription>Recent inter-agent communication</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {messages.map((msg) => (
                    <div
                      key={msg.id}
                      className="flex items-center justify-between p-3 rounded-lg border bg-card hover:bg-accent/30 transition-colors"
                    >
                      <div className="flex items-center gap-3 min-w-0">
                        <div className="flex flex-col items-center">
                          <Bot className="w-4 h-4 text-blue-500" />
                          <ArrowRight className="w-3 h-3 text-muted-foreground" />
                          <Bot className="w-4 h-4 text-green-500" />
                        </div>
                        <div>
                          <div className="text-sm">
                            <span className="font-medium">{msg.from}</span>
                            <span className="text-muted-foreground mx-1">→</span>
                            <span className="font-medium">{msg.to}</span>
                          </div>
                          <div className="flex items-center gap-2 text-xs text-muted-foreground">
                            <Badge variant="outline" className="text-[10px] h-5">
                              {msg.type}
                            </Badge>
                            <span>{msg.timestamp}</span>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-1">
                        {msg.status === "delivered" && (
                          <CheckCircle2 className="w-3.5 h-3.5 text-green-500" />
                        )}
                        {msg.status === "pending" && (
                          <Clock className="w-3.5 h-3.5 text-amber-500" />
                        )}
                        {msg.status === "failed" && (
                          <AlertCircle className="w-3.5 h-3.5 text-red-500" />
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Task Queue Tab */}
          <TabsContent value="tasks" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Zap className="h-5 w-5 text-primary" /> {t("backplane.taskQueue")}
                </CardTitle>
                <CardDescription>
                  Current task queue with priorities and assignments
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {tasks.map((task) => (
                    <div
                      key={task.id}
                      className="flex items-center justify-between p-3 rounded-lg border bg-card hover:bg-accent/30 transition-colors"
                    >
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium">{task.intent}</span>
                          {priorityBadge(task.priority)}
                        </div>
                        <div className="flex items-center gap-3 text-xs text-muted-foreground mt-1">
                          <span className="flex items-center gap-1">
                            <Bot className="w-3 h-3" />
                            {task.assignedTo || "Unassigned"}
                          </span>
                          <span>{task.createdAt}</span>
                        </div>
                      </div>
                      <div className="flex-shrink-0 ml-3">{statusBadge(task.status)}</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </OperatorShell>
  );
}

"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { OperatorShell } from "@/components/operator-shell";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Search,
  Wrench,
  CheckCircle2,
  XCircle,
  Activity,
  Code2,
  FileText,
  Globe,
  Database,
  Eye,
  MessageSquare,
  Clock,
} from "lucide-react";

interface StatCardProps {
  label: string;
  value: string;
  icon: React.ReactNode;
}

function StatCard({ label, value, icon }: StatCardProps) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">{label}</CardTitle>
        <div className="text-muted-foreground">{icon}</div>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
      </CardContent>
    </Card>
  );
}

interface Tool {
  id: string;
  name: string;
  category: string;
  description: string;
  status: "active" | "inactive" | "error";
  calls: number;
  lastUsed: string;
  icon: React.ReactNode;
}

const categoryIcons: Record<string, React.ReactNode> = {
  Memory: <Database className="w-3.5 h-3.5" />,
  Code: <Code2 className="w-3.5 h-3.5" />,
  Analysis: <Eye className="w-3.5 h-3.5" />,
  Communication: <MessageSquare className="w-3.5 h-3.5" />,
  Data: <FileText className="w-3.5 h-3.5" />,
  System: <Globe className="w-3.5 h-3.5" />,
};

export default function ToolsPage() {
  const t = useTranslations("operator.tools");
  const [searchQuery, setSearchQuery] = useState("");
  const [categoryFilter, setCategoryFilter] = useState<string>("");
  const [statusFilter, setStatusFilter] = useState<string>("");

  const tools: Tool[] = [
    {
      id: "tool_1",
      name: "retain",
      category: "Memory",
      description: "Store facts to long-term memory with automatic extraction",
      status: "active",
      calls: 14520,
      lastUsed: "1m ago",
      icon: categoryIcons["Memory"],
    },
    {
      id: "tool_2",
      name: "recall",
      category: "Memory",
      description: "Semantic search across stored memories with scoring",
      status: "active",
      calls: 38200,
      lastUsed: "30s ago",
      icon: categoryIcons["Memory"],
    },
    {
      id: "tool_3",
      name: "reflect",
      category: "Memory",
      description: "Reason across memories with bank-contextualized analysis",
      status: "active",
      calls: 8900,
      lastUsed: "2m ago",
      icon: categoryIcons["Memory"],
    },
    {
      id: "tool_4",
      name: "read_file",
      category: "Code",
      description: "Read file contents with line-level granularity",
      status: "active",
      calls: 6700,
      lastUsed: "5m ago",
      icon: categoryIcons["Code"],
    },
    {
      id: "tool_5",
      name: "edit_file",
      category: "Code",
      description: "Apply surgical edits to existing files",
      status: "active",
      calls: 4300,
      lastUsed: "8m ago",
      icon: categoryIcons["Code"],
    },
    {
      id: "tool_6",
      name: "grep_search",
      category: "Code",
      description: "Pattern-matched search across project files",
      status: "active",
      calls: 12100,
      lastUsed: "1m ago",
      icon: categoryIcons["Code"],
    },
    {
      id: "tool_7",
      name: "bash_execute",
      category: "System",
      description: "Execute commands in persistent shell",
      status: "active",
      calls: 8900,
      lastUsed: "3m ago",
      icon: categoryIcons["System"],
    },
    {
      id: "tool_8",
      name: "web_search",
      category: "Analysis",
      description: "Search the web for real-time information",
      status: "active",
      calls: 3400,
      lastUsed: "15m ago",
      icon: categoryIcons["Analysis"],
    },
    {
      id: "tool_9",
      name: "web_fetch",
      category: "Analysis",
      description: "Fetch and render URL content to markdown",
      status: "active",
      calls: 2100,
      lastUsed: "20m ago",
      icon: categoryIcons["Analysis"],
    },
    {
      id: "tool_10",
      name: "list_banks",
      category: "Memory",
      description: "Enumerate all available memory banks",
      status: "active",
      calls: 5600,
      lastUsed: "10m ago",
      icon: categoryIcons["Memory"],
    },
    {
      id: "tool_11",
      name: "create_bank",
      category: "Memory",
      description: "Provision a new memory bank with config",
      status: "active",
      calls: 230,
      lastUsed: "2d ago",
      icon: categoryIcons["Memory"],
    },
    {
      id: "tool_12",
      name: "delete_bank",
      category: "Memory",
      description: "Permanently remove bank and all its data",
      status: "inactive",
      calls: 15,
      lastUsed: "1w ago",
      icon: categoryIcons["Memory"],
    },
    {
      id: "tool_13",
      name: "legacy_transform",
      category: "Data",
      description: "Transform legacy data format to current schema",
      status: "error",
      calls: 89,
      lastUsed: "3d ago",
      icon: categoryIcons["Data"],
    },
    {
      id: "tool_14",
      name: "search_debug",
      category: "System",
      description: "Debug search pipeline with trace logging",
      status: "inactive",
      calls: 45,
      lastUsed: "1w ago",
      icon: categoryIcons["System"],
    },
  ];

  const categories = [...new Set(tools.map((t) => t.category))];
  let filteredTools = tools;
  if (searchQuery)
    filteredTools = filteredTools.filter(
      (t) =>
        t.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        t.description.toLowerCase().includes(searchQuery.toLowerCase())
    );
  if (categoryFilter) filteredTools = filteredTools.filter((t) => t.category === categoryFilter);
  if (statusFilter) filteredTools = filteredTools.filter((t) => t.status === statusFilter);

  return (
    <OperatorShell>
      <div className="p-6 space-y-6">
        <h1 className="text-2xl font-bold tracking-tight">{t("title")}</h1>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <StatCard label={t("statTotal")} value="14" icon={<Wrench className="h-4 w-4" />} />
          <StatCard
            label={t("statActive")}
            value="11"
            icon={<CheckCircle2 className="h-4 w-4" />}
          />
          <StatCard
            label={t("statCalls")}
            value="109,099"
            icon={<Activity className="h-4 w-4" />}
          />
          <StatCard
            label={t("statErrorRate")}
            value="1.2%"
            icon={<XCircle className="h-4 w-4" />}
          />
        </div>

        <div className="flex items-center gap-3">
          <div className="relative flex-1 max-w-sm">
            <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-muted-foreground" />
            <Input
              placeholder={t("searchTools")}
              className="pl-8 h-9 text-xs"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          <select
            className="h-9 text-xs border rounded-md px-2 bg-background text-muted-foreground"
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
          >
            <option value="">All categories</option>
            {categories.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          <select
            className="h-9 text-xs border rounded-md px-2 bg-background text-muted-foreground"
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
          >
            <option value="">All statuses</option>
            <option value="active">Active</option>
            <option value="inactive">Inactive</option>
            <option value="error">Error</option>
          </select>
        </div>

        {filteredTools.length === 0 ? (
          <Card>
            <CardContent className="py-12 text-center text-muted-foreground">
              {t("noTools")}
            </CardContent>
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {filteredTools.map((tool) => (
              <Card key={tool.id} className="hover:bg-accent/30 transition-colors">
                <CardHeader className="pb-2 pt-3 px-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="text-muted-foreground">{tool.icon}</div>
                      <CardTitle className="text-sm font-mono">{tool.name}</CardTitle>
                    </div>
                    <Badge
                      variant={
                        tool.status === "active"
                          ? "default"
                          : tool.status === "inactive"
                            ? "secondary"
                            : "destructive"
                      }
                      className="text-[10px]"
                    >
                      {tool.status}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="px-4 pb-3">
                  <p className="text-xs text-muted-foreground mb-2">{tool.description}</p>
                  <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
                    <Badge variant="outline" className="text-[10px]">
                      {tool.category}
                    </Badge>
                    <span className="flex items-center gap-1">
                      <Activity className="w-2.5 h-2.5" />
                      {tool.calls.toLocaleString()} calls
                    </span>
                    <span className="flex items-center gap-1">
                      <Clock className="w-2.5 h-2.5" />
                      {tool.lastUsed}
                    </span>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </OperatorShell>
  );
}

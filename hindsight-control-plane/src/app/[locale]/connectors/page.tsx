"use client";

import { useState, useEffect } from "react";
import { useTranslations } from "next-intl";
import { OperatorShell } from "@/components/operator-shell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import {
  Plug,
  Plus,
  CheckCircle2,
  AlertCircle,
  Globe,
  Database,
  MessageSquare,
  Code2,
  FileText,
  Webhook,
  Link2,
  HardDrive,
  Bot,
  Zap,
  Clock,
  Server,
  Activity,
  Search,
  Loader2,
  Wrench,
} from "lucide-react";
import { MCPToolsView } from "@/components/mcp-tools-view";

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

interface McpConnector {
  id: string;
  name: string;
  type: string;
  status: "connected" | "disconnected" | "error";
  lastSync: string;
  endpoint: string;
}

interface WebhookEntry {
  id: string;
  url: string;
  events: string[];
  status: "active" | "paused" | "error";
  lastTriggered: string;
}

interface Integration {
  id: string;
  name: string;
  connected: boolean;
  lastSync: string;
}

interface Bridge {
  id: string;
  name: string;
  source: string;
  target: string;
  status: "active" | "paused" | "error";
  lastSync: string;
}

export default function ConnectorsPage() {
  const t = useTranslations("operator");
  const c = useTranslations("operator.connectors");

  const [webhooks, setWebhooks] = useState<WebhookEntry[]>([]);
  const [webhooksLoading, setWebhooksLoading] = useState(true);
  const [webhooksError, setWebhooksError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      setWebhooksLoading(true);
      setWebhooksError(null);
      try {
        const res = await fetch("/api/banks/default/webhooks");
        if (res.ok) {
          const data = await res.json();
          // Hindsight API returns webhooks in the response body
          const list = data.webhooks || data || [];
          setWebhooks(Array.isArray(list) ? list : []);
        } else {
          setWebhooksError(`Failed (${res.status})`);
        }
      } catch {
        setWebhooksError("Network error");
      } finally {
        setWebhooksLoading(false);
      }
    };
    load();
  }, []);

  const mcpConnectors: McpConnector[] = [
    {
      id: "m1",
      name: "File System",
      type: "Files",
      status: "connected",
      lastSync: "Just now",
      endpoint: "mcp://fs.local:9000",
    },
    {
      id: "m2",
      name: "PostgreSQL",
      type: "Database",
      status: "connected",
      lastSync: "2s ago",
      endpoint: "mcp://pg.local:5432",
    },
    {
      id: "m3",
      name: "Web Search",
      type: "Search",
      status: "connected",
      lastSync: "1m ago",
      endpoint: "mcp://search:9002",
    },
    {
      id: "m4",
      name: "Memory Store",
      type: "Memory",
      status: "connected",
      lastSync: "30s ago",
      endpoint: "mcp://memory:9003",
    },
    {
      id: "m5",
      name: "Code Analysis",
      type: "Analysis",
      status: "disconnected",
      lastSync: "5h ago",
      endpoint: "mcp://code:9004",
    },
    {
      id: "m6",
      name: "Slack Gateway",
      type: "Messaging",
      status: "error",
      lastSync: "Failed",
      endpoint: "mcp://slack:9005",
    },
  ];

  const integrations: Integration[] = [
    { id: "i1", name: "Slack", connected: true, lastSync: "1m ago" },
    { id: "i2", name: "Discord", connected: true, lastSync: "5m ago" },
    { id: "i3", name: "GitHub", connected: true, lastSync: "2m ago" },
    { id: "i4", name: "Linear", connected: true, lastSync: "10m ago" },
    { id: "i5", name: "Notion", connected: false, lastSync: "3d ago" },
    { id: "i6", name: "Google Drive", connected: false, lastSync: "1w ago" },
    { id: "i7", name: "Jira", connected: true, lastSync: "1h ago" },
    { id: "i8", name: "Sentry", connected: true, lastSync: "30s ago" },
  ];

  const bridges: Bridge[] = [
    {
      id: "b1",
      name: "Cross-Platform Relay",
      source: "Slack",
      target: "Discord",
      status: "active",
      lastSync: "2m ago",
    },
    {
      id: "b2",
      name: "Issue Sync",
      source: "GitHub",
      target: "Linear",
      status: "active",
      lastSync: "5m ago",
    },
    {
      id: "b3",
      name: "Doc Mirror",
      source: "Notion",
      target: "Google Drive",
      status: "paused",
      lastSync: "1d ago",
    },
  ];

  const [searchQuery, setSearchQuery] = useState("");

  const filteredMcp = mcpConnectors.filter(
    (m) =>
      m.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      m.type.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const statusBadge = (status: string) => {
    const variant =
      status === "connected" || status === "active"
        ? "default"
        : status === "disconnected" || status === "paused"
          ? "secondary"
          : "destructive";
    return (
      <Badge variant={variant} className="text-[10px] capitalize">
        {status}
      </Badge>
    );
  };

  const activeCount = mcpConnectors.filter((c) => c.status === "connected").length;

  return (
    <OperatorShell>
      <div className="p-6 space-y-6">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight flex items-center gap-3">
              <Plug className="h-6 w-6 text-primary" />
              {t("panels.connectors")}
            </h1>
            <p className="text-sm text-muted-foreground mt-1">{t("descriptions.connectors")}</p>
          </div>
          <Button size="sm" className="flex items-center gap-2">
            <Plus className="w-3.5 h-3.5" />
            {c("addConnector")}
          </Button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <StatCard label={c("statTotal")} value="19" icon={<Server className="h-4 w-4" />} />
          <StatCard
            label={c("statActive")}
            value={String(activeCount + 7)}
            icon={<Activity className="h-4 w-4" />}
          />
          <StatCard
            label={c("statDataFlows")}
            value="2.4K/day"
            icon={<Zap className="h-4 w-4" />}
          />
          <StatCard
            label={c("statErrorRate")}
            value="0.8%"
            icon={<AlertCircle className="h-4 w-4" />}
          />
        </div>

        <Tabs defaultValue="mcp" className="space-y-4">
          <TabsList>
            <TabsTrigger value="mcp" className="flex items-center gap-2">
              <Plug className="h-4 w-4" />
              {c("mcpConnections")}
            </TabsTrigger>
            <TabsTrigger value="webhooks" className="flex items-center gap-2">
              <Webhook className="h-4 w-4" />
              {c("webhooks")}
            </TabsTrigger>
            <TabsTrigger value="integrations" className="flex items-center gap-2">
              <Globe className="h-4 w-4" />
              {c("externalIntegrations")}
            </TabsTrigger>
            <TabsTrigger value="bridges" className="flex items-center gap-2">
              <Link2 className="h-4 w-4" />
              {c("platformBridges")}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="mcp" className="space-y-4">
            <MCPToolsView />
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Plug className="h-4 w-4" />
                  <CardTitle className="text-base">{c("mcpConnections")}</CardTitle>
                </div>
                <CardDescription>
                  MCP protocol connections to data sources and services
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="relative max-w-sm">
                  <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-muted-foreground" />
                  <Input
                    placeholder="Search connectors..."
                    className="pl-8 h-9 text-xs"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                </div>

                {filteredMcp.length === 0 ? (
                  <p className="text-sm text-muted-foreground py-4 text-center">
                    {c("noConnectors")}
                  </p>
                ) : (
                  <div className="divide-y rounded-md border">
                    {filteredMcp.map((conn) => (
                      <div
                        key={conn.id}
                        className="flex items-center justify-between px-4 py-3 text-sm hover:bg-accent/30"
                      >
                        <div className="flex items-center gap-3 min-w-0 flex-1">
                          <div>
                            <p className="font-medium">{conn.name}</p>
                            <p className="text-xs text-muted-foreground">{conn.type}</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-4 shrink-0">
                          <code className="text-[10px] text-muted-foreground">{conn.endpoint}</code>
                          {statusBadge(conn.status)}
                          <span className="flex items-center gap-1 text-[10px] text-muted-foreground">
                            <Clock className="w-2.5 h-2.5" />
                            {conn.lastSync}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="webhooks" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">{c("webhooks")}</CardTitle>
                <CardDescription>Outbound webhook notifications for system events</CardDescription>
              </CardHeader>
              <CardContent className="p-0">
                {webhooksLoading ? (
                  <div className="flex items-center justify-center py-8 text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin mr-2" /> Loading webhooks...
                  </div>
                ) : webhooksError ? (
                  <div className="flex items-center justify-center py-8 text-amber-600">
                    <AlertCircle className="h-4 w-4 mr-2" /> {webhooksError}
                  </div>
                ) : webhooks.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground text-sm">
                    No webhooks configured
                  </div>
                ) : (
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b text-muted-foreground text-xs">
                        <th className="text-left font-medium px-6 py-3">URL</th>
                        <th className="text-left font-medium px-6 py-3">Events</th>
                        <th className="text-left font-medium px-6 py-3">{c("status")}</th>
                        <th className="text-left font-medium px-6 py-3">{c("lastSync")}</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y">
                      {webhooks.map((wh: any) => (
                        <tr key={wh.id} className="hover:bg-accent/30">
                          <td className="px-6 py-3">
                            <code className="text-[10px] font-mono">{wh.url}</code>
                          </td>
                          <td className="px-6 py-3">
                            <div className="flex flex-wrap gap-1">
                              {(wh.events || wh.event_types || []).map((ev: string) => (
                                <Badge key={ev} variant="outline" className="text-[10px]">
                                  {ev}
                                </Badge>
                              ))}
                            </div>
                          </td>
                          <td className="px-6 py-3">
                            {statusBadge(wh.status || wh.active ? "active" : "paused")}
                          </td>
                          <td className="px-6 py-3 text-muted-foreground text-xs">
                            {wh.last_triggered_at
                              ? new Date(wh.last_triggered_at).toLocaleString()
                              : wh.lastTriggered || "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="integrations" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {integrations.map((int) => {
                const iconMap: Record<string, React.ReactNode> = {
                  Slack: <MessageSquare className="w-5 h-5" />,
                  Discord: <MessageSquare className="w-5 h-5" />,
                  GitHub: <Code2 className="w-5 h-5" />,
                  Linear: <FileText className="w-5 h-5" />,
                  Notion: <FileText className="w-5 h-5" />,
                  "Google Drive": <HardDrive className="w-5 h-5" />,
                  Jira: <Bot className="w-5 h-5" />,
                  Sentry: <AlertCircle className="w-5 h-5" />,
                };
                return (
                  <Card key={int.id} className="hover:bg-accent/30 transition-colors">
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <div className="text-muted-foreground">{iconMap[int.name]}</div>
                        <Switch checked={int.connected} />
                      </div>
                      <CardTitle className="text-sm mt-2">{int.name}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="flex items-center justify-between text-xs">
                        {int.connected ? (
                          <span className="text-green-600 dark:text-green-400 flex items-center gap-1">
                            <CheckCircle2 className="w-3 h-3" /> Connected
                          </span>
                        ) : (
                          <span className="text-muted-foreground">Disconnected</span>
                        )}
                        <span className="text-muted-foreground">{int.lastSync}</span>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </TabsContent>

          <TabsContent value="bridges" className="space-y-4">
            <div className="space-y-3">
              {bridges.map((br) => (
                <Card key={br.id}>
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-sm font-medium flex items-center gap-2">
                        <Link2 className="h-4 w-4 text-primary" />
                        {br.name}
                      </CardTitle>
                      {statusBadge(br.status)}
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center gap-3 text-xs text-muted-foreground">
                      <span>{br.source}</span>
                      <span>→</span>
                      <span>{br.target}</span>
                      <span className="ml-auto">{br.lastSync}</span>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </OperatorShell>
  );
}

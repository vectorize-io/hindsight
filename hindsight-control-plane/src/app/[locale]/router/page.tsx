"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { OperatorShell } from "@/components/operator-shell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  GitCompare,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  ArrowRight,
  Route,
  Zap,
  BarChart3,
  Server,
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

interface Provider {
  id: string;
  name: string;
  model: string;
  status: "healthy" | "degraded" | "unknown";
  latency: string;
  position: string;
}

interface RouteData {
  id: string;
  path: string;
  target: string;
  provider: string;
  successRate: number;
  avgLatency: string;
  traffic: number;
}

export default function RouterPage() {
  const t = useTranslations("operator");
  const r = useTranslations("operator.router");

  const [providers] = useState<Provider[]>([
    {
      id: "p1",
      name: "OpenAI",
      model: "gpt-4o",
      status: "healthy",
      latency: "120ms",
      position: "Primary",
    },
    {
      id: "p2",
      name: "Anthropic",
      model: "claude-sonnet-4",
      status: "healthy",
      latency: "180ms",
      position: "Failover 1",
    },
    {
      id: "p3",
      name: "Groq",
      model: "llama-3.3-70b",
      status: "healthy",
      latency: "80ms",
      position: "Failover 2",
    },
    {
      id: "p4",
      name: "Ollama",
      model: "mistral:latest",
      status: "degraded",
      latency: "350ms",
      position: "Failover 3",
    },
  ]);

  const [routes] = useState<RouteData[]>([
    {
      id: "r1",
      path: "/api/chat",
      target: "gpt-4o",
      provider: "OpenAI",
      successRate: 99.2,
      avgLatency: "120ms",
      traffic: 45,
    },
    {
      id: "r2",
      path: "/api/embed",
      target: "nomic-embed",
      provider: "Ollama",
      successRate: 99.8,
      avgLatency: "35ms",
      traffic: 30,
    },
    {
      id: "r3",
      path: "/api/rerank",
      target: "cross-encoder",
      provider: "Ollama",
      successRate: 97.5,
      avgLatency: "55ms",
      traffic: 15,
    },
    {
      id: "r4",
      path: "/api/extract",
      target: "gpt-4o-mini",
      provider: "OpenAI",
      successRate: 98.9,
      avgLatency: "90ms",
      traffic: 8,
    },
    {
      id: "r5",
      path: "/api/summarize",
      target: "claude-sonnet-4",
      provider: "Anthropic",
      successRate: 96.7,
      avgLatency: "210ms",
      traffic: 2,
    },
  ]);

  const statusBadge = (status: string) => {
    const variant =
      status === "healthy" ? "default" : status === "degraded" ? "secondary" : "outline";
    return (
      <Badge variant={variant} className="text-[10px]">
        {status}
      </Badge>
    );
  };

  const statusIcon = (status: string) => {
    switch (status) {
      case "healthy":
        return <CheckCircle2 className="w-4 h-4 text-green-500" />;
      case "degraded":
        return <AlertCircle className="w-4 h-4 text-amber-500" />;
      default:
        return <HelpCircle className="w-4 h-4 text-muted-foreground" />;
    }
  };

  return (
    <OperatorShell>
      <div className="p-6 space-y-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-3">
            <GitCompare className="h-6 w-6 text-primary" />
            {t("panels.router")}
          </h1>
          <p className="text-sm text-muted-foreground mt-1">{t("descriptions.router")}</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <StatCard label={r("statProviders")} value="4" icon={<Server className="h-4 w-4" />} />
          <StatCard label={r("statActiveRoutes")} value="12" icon={<Route className="h-4 w-4" />} />
          <StatCard label={r("statAvgLatency")} value="142ms" icon={<Zap className="h-4 w-4" />} />
          <StatCard
            label={r("statFailoverCount")}
            value="3"
            icon={<BarChart3 className="h-4 w-4" />}
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <GitCompare className="h-4 w-4 text-primary" />
                {r("failoverChain")}
              </CardTitle>
              <CardDescription>{r("strategy")}: Latency-based routing</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {providers.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground text-sm">
                  {r("noProviders")}
                </div>
              ) : (
                providers.map((p, i) => (
                  <div
                    key={p.id}
                    className="flex items-center justify-between p-3 rounded-lg border bg-card hover:bg-accent/30 transition-colors"
                  >
                    <div className="flex items-center gap-3">
                      {statusIcon(p.status)}
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium">{p.name}</span>
                          <Badge variant="outline" className="text-[10px] h-5">
                            {p.model}
                          </Badge>
                        </div>
                        <div className="flex items-center gap-2 text-[10px] text-muted-foreground mt-0.5">
                          <span>{p.position}</span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="text-right">
                        <div className="text-xs font-mono">{p.latency}</div>
                        {statusBadge(p.status)}
                      </div>
                      {i < providers.length - 1 && (
                        <ArrowRight className="w-3 h-3 text-muted-foreground" />
                      )}
                    </div>
                  </div>
                ))
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Route className="h-4 w-4 text-primary" />
                {r("routePreview")}
              </CardTitle>
              <CardDescription>Active API routes and their routing targets</CardDescription>
            </CardHeader>
            <CardContent className="p-0">
              {routes.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground text-sm px-6">
                  {r("noRoutes")}
                </div>
              ) : (
                <div className="divide-y">
                  {routes.map((route) => (
                    <div
                      key={route.id}
                      className="flex items-center justify-between px-6 py-3 text-sm"
                    >
                      <div className="flex items-center gap-3 min-w-0">
                        <code className="text-xs font-mono bg-muted px-1.5 py-0.5 rounded truncate">
                          {route.path}
                        </code>
                        <span className="text-muted-foreground text-xs">{route.target}</span>
                      </div>
                      <div className="flex items-center gap-4 text-xs text-muted-foreground flex-shrink-0">
                        <span className="text-green-600 dark:text-green-400">
                          {route.successRate}%
                        </span>
                        <span>{route.avgLatency}</span>
                        <span className="font-mono">{route.traffic}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-primary" />
              {r("latencyHeatmap")}
            </CardTitle>
            <CardDescription>Provider latency over the last 24 hours</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {providers.map((p) => {
                const bars = [
                  65, 70, 55, 80, 60, 75, 85, 50, 45, 60, 90, 40, 55, 70, 65, 80, 75, 60, 55, 70,
                  65, 75, 85, 50,
                ];
                const baseWidth =
                  p.status === "degraded" ? 350 : p.status === "healthy" ? 120 : 200;
                return (
                  <div key={p.id} className="flex items-center gap-3">
                    <span className="text-xs font-medium w-20 truncate">{p.name}</span>
                    <div className="flex-1 flex gap-0.5 h-6 items-end">
                      {bars.map((h, i) => (
                        <div
                          key={i}
                          className="flex-1 rounded-sm"
                          style={{
                            height: `${Math.max(h * 0.3, 4)}px`,
                            backgroundColor:
                              h > 80
                                ? "#ef4444"
                                : h > 60
                                  ? "#f59e0b"
                                  : h > 40
                                    ? "#22c55e"
                                    : "#86efac",
                          }}
                        />
                      ))}
                    </div>
                    <span className="text-xs text-muted-foreground w-16 text-right">
                      {p.latency}
                    </span>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      </div>
    </OperatorShell>
  );
}

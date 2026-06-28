"use client";

import { useState, useEffect, useCallback } from "react";
import { useTranslations } from "next-intl";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Database,
  Server,
  HardDrive,
  RefreshCw,
  Wifi,
  WifiOff,
  AlertCircle,
  Loader2,
  ExternalLink,
} from "lucide-react";

interface Connection {
  id: string;
  name: string;
  type: string;
  status: "connected" | "disconnected" | "error";
  stats?: { documents: number; namespaces: number; totalVectors: number };
}

interface Health {
  status: string;
  version: string;
}

interface Stats {
  totalConnections: number;
  connectedCount: number;
  totalDocuments: number;
  totalVectors: number;
}

export function VectorAdminView() {
  const t = useTranslations("vectorAdmin");
  const [connections, setConnections] = useState<Connection[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const loadAll = useCallback(() => {
    setLoading(true);
    Promise.all([
      fetch("/api/vector-admin/connections").then((r) => r.json()),
      fetch("/api/vector-admin/health").then((r) => r.json()),
      fetch("/api/vector-admin/stats").then((r) => r.json()),
    ])
      .then(([conns, h, s]) => {
        setConnections(Array.isArray(conns) ? conns : []);
        setHealth(h);
        setStats(s);
        setError("");
      })
      .catch(() => setError("Failed to connect to VectorAdmin backend"))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    loadAll();
  }, [loadAll]);

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">{t("title")}</h1>
          <p className="text-muted-foreground text-sm mt-1">
            {t("description")}
            {health && (
              <span className="ml-2 text-xs">
                v{health.version} · {health.status}
              </span>
            )}
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={loadAll} disabled={loading}>
            <RefreshCw className={`h-4 w-4 mr-1.5 ${loading ? "animate-spin" : ""}`} />
            {t("refresh")}
          </Button>
          <Button variant="outline" size="sm" asChild>
            <a href="http://localhost:3000" target="_blank" rel="noopener noreferrer">
              <ExternalLink className="h-4 w-4 mr-1.5" />
              {t("openFullUI")}
            </a>
          </Button>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 dark:bg-red-950/10 dark:border-red-900 p-4 text-sm text-red-700 dark:text-red-400">
          {t("errorLoading")}
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">{t("databases")}</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {loading ? (
                <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
              ) : (
                stats?.totalConnections ?? "-"
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              {stats?.connectedCount ?? 0} {t("connected")}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">{t("connected")}</CardTitle>
            <Wifi className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600 dark:text-green-400">
              {loading ? (
                <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
              ) : (
                stats?.connectedCount ?? 0
              )}
            </div>
            <p className="text-xs text-muted-foreground">{t("connectionStatus")}</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">{t("documents")}</CardTitle>
            <HardDrive className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {loading ? (
                <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
              ) : (
                stats?.totalDocuments?.toLocaleString() ?? "-"
              )}
            </div>
            <p className="text-xs text-muted-foreground">{t("overview")}</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">{t("vectors")}</CardTitle>
            <Server className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {loading ? (
                <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
              ) : (
                stats?.totalVectors?.toLocaleString() ?? "-"
              )}
            </div>
            <p className="text-xs text-muted-foreground">{t("vectorCount")}</p>
          </CardContent>
        </Card>
      </div>

      <div>
        <h2 className="text-lg font-semibold mb-3">{t("collections")}</h2>
        {loading && connections.length === 0 ? (
          <div className="flex items-center gap-2 text-muted-foreground text-sm py-8">
            <Loader2 className="h-4 w-4 animate-spin" />
            {t("loadingConnections")}
          </div>
        ) : connections.length === 0 ? (
          <div className="text-muted-foreground text-sm py-8 text-center border rounded-lg">
            {t("noConnections")}
          </div>
        ) : (
          <div className="space-y-3">
            {connections.map((conn) => (
              <Card key={conn.id} className="hover:bg-accent/30 transition-colors">
                <CardContent className="flex items-center justify-between p-4">
                  <div className="flex items-center gap-3">
                    {conn.status === "connected" ? (
                      <Wifi className="h-5 w-5 text-green-500" />
                    ) : conn.status === "error" ? (
                      <AlertCircle className="h-5 w-5 text-red-500" />
                    ) : (
                      <WifiOff className="h-5 w-5 text-muted-foreground" />
                    )}
                    <div>
                      <div className="font-medium">{conn.name}</div>
                      <div className="text-xs text-muted-foreground">
                        {conn.type}
                        {conn.stats && (
                          <>
                            {" · "}
                            {conn.stats.documents} {t("documents")}
                            {" · "}
                            {conn.stats.namespaces} {t("collections")}
                            {" · "}
                            {conn.stats.totalVectors.toLocaleString()} {t("vectors")}
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                  <Badge
                    variant={
                      conn.status === "connected"
                        ? "default"
                        : conn.status === "error"
                          ? "destructive"
                          : "secondary"
                    }
                  >
                    {conn.status === "connected"
                      ? t("healthOk")
                      : conn.status === "error"
                        ? t("healthError")
                        : t("disconnected")}
                  </Badge>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

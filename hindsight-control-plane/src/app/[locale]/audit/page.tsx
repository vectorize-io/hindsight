"use client";

import { useState, useEffect, useCallback } from "react";
import { useTranslations } from "next-intl";
import { OperatorShell } from "@/components/operator-shell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Shield,
  Download,
  Search,
  Loader2,
  CheckCircle2,
  AlertCircle,
  XCircle,
  User,
  Activity,
  FileText,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";

interface AuditEvent {
  id: string;
  timestamp: string;
  action: string;
  actor: string;
  resource: string;
  status: "success" | "failure" | "pending";
  details: string;
}

interface AuditStats {
  total: number;
  failures: number;
  actors: number;
  period_days?: number;
}

const actionColors: Record<string, string> = {
  bank_create: "bg-blue-100 text-blue-700 dark:bg-blue-950/30 dark:text-blue-400",
  bank_update: "bg-indigo-100 text-indigo-700 dark:bg-indigo-950/30 dark:text-indigo-400",
  bank_delete: "bg-red-100 text-red-700 dark:bg-red-950/30 dark:text-red-400",
  memory_retain: "bg-green-100 text-green-700 dark:bg-green-950/30 dark:text-green-400",
  memory_update: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950/30 dark:text-emerald-400",
  memory_delete: "bg-red-100 text-red-700 dark:bg-red-950/30 dark:text-red-400",
  memory_recall: "bg-teal-100 text-teal-700 dark:bg-teal-950/30 dark:text-teal-400",
  config_update: "bg-amber-100 text-amber-700 dark:bg-amber-950/30 dark:text-amber-400",
  directive_create: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950/30 dark:text-emerald-400",
  directive_update: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950/30 dark:text-emerald-400",
  directive_delete: "bg-red-100 text-red-700 dark:bg-red-950/30 dark:text-red-400",
  webhook_create: "bg-purple-100 text-purple-700 dark:bg-purple-950/30 dark:text-purple-400",
  webhook_delete: "bg-red-100 text-red-700 dark:bg-red-950/30 dark:text-red-400",
  webhook_delivery: "bg-sky-100 text-sky-700 dark:bg-sky-950/30 dark:text-sky-400",
  mental_model_create: "bg-violet-100 text-violet-700 dark:bg-violet-950/30 dark:text-violet-400",
  mental_model_refresh: "bg-violet-100 text-violet-700 dark:bg-violet-950/30 dark:text-violet-400",
  llm_request: "bg-orange-100 text-orange-700 dark:bg-orange-950/30 dark:text-orange-400",
  consolidation: "bg-slate-100 text-slate-700 dark:bg-slate-950/30 dark:text-slate-400",
  operation_retry: "bg-rose-100 text-rose-700 dark:bg-rose-950/30 dark:text-rose-400",
  default: "bg-gray-100 text-gray-700 dark:bg-gray-950/30 dark:text-gray-400",
};

function getActionColor(action: string): string {
  const key = Object.keys(actionColors).find((k) => action?.toLowerCase().includes(k));
  return actionColors[key || "default"];
}

function formatTimestamp(ts: string): string {
  try {
    const d = new Date(ts);
    return d.toLocaleString("en-US", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return ts;
  }
}

export default function AuditPage() {
  const t = useTranslations("operator");
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [stats, setStats] = useState<AuditStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [page, setPage] = useState(0);
  const pageSize = 20;

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      // Fetch audit logs
      const logRes = await fetch(
        `/api/banks/default/audit-logs?limit=${pageSize + 1}&offset=${page * pageSize}`
      );
      if (!logRes.ok) throw new Error("Failed to fetch audit logs");
      const logData = await logRes.json();
      const items: AuditEvent[] = (logData.items || logData.audit_logs || logData.events || []).map(
        (e: any) => ({
          id: e.id || e.audit_log_id,
          timestamp: e.timestamp || e.created_at || e.occurred_at,
          action: e.action || e.event_type || "unknown",
          actor: e.actor || e.agent_id || "system",
          resource: e.resource || e.resource_id || "—",
          status:
            e.status === "success" || e.status === "failure"
              ? e.status
              : e.level === "error"
                ? "failure"
                : "success",
          details: e.details || e.description || "",
        })
      );
      setEvents(items);

      // Fetch stats
      const statsRes = await fetch("/api/banks/default/audit-logs/stats");
      if (statsRes.ok) {
        const statsData = await statsRes.json();
        setStats({
          total: statsData.total || statsData.count || 0,
          failures: statsData.failures || statsData.error_count || statsData.critical || 0,
          actors: statsData.actors || statsData.unique_actors || statsData.unique_agents || 0,
          period_days: statsData.period_days,
        });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load audit data");
    } finally {
      setLoading(false);
    }
  }, [page]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Client-side filtering for search
  const filtered = searchQuery
    ? events.filter(
        (e) =>
          e.action.toLowerCase().includes(searchQuery.toLowerCase()) ||
          e.actor.toLowerCase().includes(searchQuery.toLowerCase()) ||
          e.resource.toLowerCase().includes(searchQuery.toLowerCase()) ||
          e.details.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : events;

  const showNext = events.length > pageSize;

  return (
    <OperatorShell>
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight flex items-center gap-3">
              <Shield className="h-6 w-6 text-primary" />
              {t("audit.title")}
            </h1>
            <p className="text-sm text-muted-foreground mt-1">{t("descriptions.audit")}</p>
          </div>
          <div className="flex items-center gap-2">
            {stats?.period_days && (
              <span className="text-[11px] text-muted-foreground">
                Last {stats.period_days} days
              </span>
            )}
            <Button variant="outline" size="sm" onClick={fetchData} disabled={loading}>
              <RefreshCw className={`h-4 w-4 mr-1 ${loading ? "animate-spin" : ""}`} />
              Refresh
            </Button>
          </div>
        </div>

        {/* Stats from real API */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Activity className="h-4 w-4" /> {t("audit.statEvents")}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {stats ? (
                  stats.total.toLocaleString()
                ) : (
                  <Loader2 className="h-4 w-4 animate-spin" />
                )}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <AlertCircle className="h-4 w-4" /> {t("audit.statCritical")}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-red-600">
                {stats ? (
                  stats.failures.toLocaleString()
                ) : (
                  <Loader2 className="h-4 w-4 animate-spin" />
                )}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <User className="h-4 w-4" /> {t("audit.statActors")}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {stats ? stats.actors : <Loader2 className="h-4 w-4 animate-spin" />}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <FileText className="h-4 w-4" /> Bank
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Badge variant="outline" className="font-mono text-xs">
                default
              </Badge>
            </CardContent>
          </Card>
        </div>

        {/* Search + Filters */}
        <div className="flex items-center gap-3 flex-wrap">
          <div className="relative flex-1 max-w-sm">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <input
              type="text"
              placeholder={t("audit.searchLogs")}
              className="w-full h-9 pl-9 pr-3 rounded-lg border bg-background text-sm"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          {error && (
            <span className="text-xs text-red-500 flex items-center gap-1">
              <AlertCircle className="w-3 h-3" /> {error}
            </span>
          )}
        </div>

        {/* Audit log table */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-base flex items-center gap-2">
                  <Shield className="h-5 w-5 text-primary" /> Event Log
                </CardTitle>
                <CardDescription>Immutable audit trail of all system actions</CardDescription>
              </div>
              {/* Pagination */}
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  disabled={page === 0}
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                >
                  <ChevronLeft className="h-3 w-3" />
                </Button>
                <span className="text-xs text-muted-foreground min-w-[4rem] text-center">
                  Page {page + 1}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={!showNext}
                  onClick={() => setPage((p) => p + 1)}
                >
                  <ChevronRight className="h-3 w-3" />
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {loading && events.length === 0 ? (
              <div className="flex items-center justify-center py-12 text-muted-foreground">
                <Loader2 className="h-5 w-5 animate-spin mr-2" /> Loading audit events...
              </div>
            ) : filtered.length === 0 ? (
              <div className="text-sm text-muted-foreground text-center py-12">
                {searchQuery ? "No events match your search" : "No audit events found"}
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b text-left text-xs text-muted-foreground">
                      <th className="pb-3 font-medium pr-4">{t("audit.timestamp")}</th>
                      <th className="pb-3 font-medium pr-4">{t("audit.action")}</th>
                      <th className="pb-3 font-medium pr-4">{t("audit.actor")}</th>
                      <th className="pb-3 font-medium pr-4">{t("audit.resource")}</th>
                      <th className="pb-3 font-medium pr-4">{t("audit.status")}</th>
                      <th className="pb-3 font-medium">{t("audit.details")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.slice(0, pageSize).map((event) => (
                      <tr key={event.id} className="border-b last:border-0 hover:bg-accent/30">
                        <td className="py-3 pr-4 font-mono text-xs text-muted-foreground whitespace-nowrap">
                          {formatTimestamp(event.timestamp)}
                        </td>
                        <td className="py-3 pr-4">
                          <Badge
                            variant="outline"
                            className={`text-[10px] h-5 font-mono ${getActionColor(event.action)}`}
                          >
                            {event.action}
                          </Badge>
                        </td>
                        <td className="py-3 pr-4 font-mono text-xs">{event.actor}</td>
                        <td className="py-3 pr-4 font-mono text-xs text-muted-foreground max-w-[200px] truncate">
                          {event.resource}
                        </td>
                        <td className="py-3 pr-4">
                          {event.status === "success" ? (
                            <Badge
                              variant="outline"
                              className="border-green-300 text-green-700 bg-green-50 text-[10px] h-5"
                            >
                              <CheckCircle2 className="w-2.5 h-2.5 mr-1" />
                              Success
                            </Badge>
                          ) : event.status === "failure" ? (
                            <Badge variant="destructive" className="text-[10px] h-5">
                              <XCircle className="w-2.5 h-2.5 mr-1" />
                              Failure
                            </Badge>
                          ) : (
                            <Badge
                              variant="outline"
                              className="border-amber-300 text-amber-700 bg-amber-50 text-[10px] h-5"
                            >
                              Pending
                            </Badge>
                          )}
                        </td>
                        <td className="py-3 text-xs text-muted-foreground max-w-[300px] truncate">
                          {event.details || "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </OperatorShell>
  );
}

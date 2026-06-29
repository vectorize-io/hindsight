"use client";

import { useState } from "react";
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
  Clock,
  User,
  Activity,
  FileText,
  RefreshCw,
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

export default function AuditPage() {
  const t = useTranslations("operator");
  const [searchQuery, setSearchQuery] = useState("");
  const [actionFilter, setActionFilter] = useState<string>("all");

  const events: AuditEvent[] = [
    { id: "aud_001", timestamp: "2026-06-28 14:23:01", action: "bank.create", actor: "oliver", resource: "bank::opencode", status: "success", details: "Created new memory bank 'opencode'" },
    { id: "aud_002", timestamp: "2026-06-28 14:22:45", action: "memory.retain", actor: "system", resource: "memory::fact_8902", status: "success", details: "Retained 3 facts from conversation" },
    { id: "aud_003", timestamp: "2026-06-28 14:20:12", action: "config.update", actor: "oliver", resource: "config::llm", status: "success", details: "Updated LLM model to claude-sonnet-4" },
    { id: "aud_004", timestamp: "2026-06-28 14:15:33", action: "agent.deploy", actor: "oliver", resource: "agent::tool-agent", status: "success", details: "Deployed Tool Agent v2.1.0" },
    { id: "aud_005", timestamp: "2026-06-28 14:10:00", action: "auth.login", actor: "oliver", resource: "session::abc123", status: "success", details: "Login from 192.168.1.100" },
    { id: "aud_006", timestamp: "2026-06-28 14:05:22", action: "memory.recall", actor: "agent::researcher", resource: "bank::default", status: "success", details: "Recalled 15 facts for analysis" },
    { id: "aud_007", timestamp: "2026-06-28 14:00:45", action: "system.backup", actor: "system", resource: "database::pg0", status: "success", details: "Automatic backup completed (2.3GB)" },
    { id: "aud_008", timestamp: "2026-06-28 13:55:10", action: "api.key_rotation", actor: "oliver", resource: "key::ollama-embeddings", status: "success", details: "Rotated API key for embeddings provider" },
    { id: "aud_009", timestamp: "2026-06-28 13:50:00", action: "worker.scale", actor: "system", resource: "worker::pool", status: "success", details: "Scaled worker pool from 2 to 4" },
    { id: "aud_010", timestamp: "2026-06-28 13:45:30", action: "memory.retain", actor: "agent::memory-agent", resource: "memory::fact_8890", status: "failure", details: "Failed to retain: LLM timeout" },
    { id: "aud_011", timestamp: "2026-06-28 13:40:00", action: "auth.login_failed", actor: "unknown", resource: "session::null", status: "failure", details: "Failed login attempt from 10.0.0.55" },
    { id: "aud_012", timestamp: "2026-06-28 13:35:15", action: "directive.create", actor: "oliver", resource: "directive::pri-003", status: "success", details: "Created prime directive: security-first" },
  ];

  const eventCount = events.length;
  const criticalCount = events.filter(e => e.status === "failure").length;
  const uniqueActors = [...new Set(events.map(e => e.actor))].length;

  const filtered = events.filter(e => {
    if (actionFilter !== "all" && e.action !== actionFilter) return false;
    if (searchQuery && !e.action.toLowerCase().includes(searchQuery.toLowerCase()) && !e.actor.toLowerCase().includes(searchQuery.toLowerCase()) && !e.resource.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  });

  const actionColors: Record<string, string> = {
    "bank.create": "bg-blue-100 text-blue-700 dark:bg-blue-950/30 dark:text-blue-400",
    "memory.retain": "bg-green-100 text-green-700 dark:bg-green-950/30 dark:text-green-400",
    "config.update": "bg-amber-100 text-amber-700 dark:bg-amber-950/30 dark:text-amber-400",
    "agent.deploy": "bg-purple-100 text-purple-700 dark:bg-purple-950/30 dark:text-purple-400",
    "auth.login": "bg-sky-100 text-sky-700 dark:bg-sky-950/30 dark:text-sky-400",
    "memory.recall": "bg-teal-100 text-teal-700 dark:bg-teal-950/30 dark:text-teal-400",
    "system.backup": "bg-slate-100 text-slate-700 dark:bg-slate-950/30 dark:text-slate-400",
    "api.key_rotation": "bg-red-100 text-red-700 dark:bg-red-950/30 dark:text-red-400",
    "worker.scale": "bg-indigo-100 text-indigo-700 dark:bg-indigo-950/30 dark:text-indigo-400",
    "auth.login_failed": "bg-red-100 text-red-700 dark:bg-red-950/30 dark:text-red-400",
    "directive.create": "bg-emerald-100 text-emerald-700 dark:bg-emerald-950/30 dark:text-emerald-400",
  };

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
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-1" /> {t("audit.exportCSV")}
          </Button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2"><Activity className="h-4 w-4" /> {t("audit.statEvents")}</CardTitle>
            </CardHeader>
            <CardContent><div className="text-2xl font-bold">{eventCount}</div></CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2"><AlertCircle className="h-4 w-4" /> {t("audit.statCritical")}</CardTitle>
            </CardHeader>
            <CardContent><div className="text-2xl font-bold text-red-600">{criticalCount}</div></CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2"><User className="h-4 w-4" /> {t("audit.statActors")}</CardTitle>
            </CardHeader>
            <CardContent><div className="text-2xl font-bold">{uniqueActors}</div></CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2"><FileText className="h-4 w-4" /> {t("audit.statExportCount")}</CardTitle>
            </CardHeader>
            <CardContent><div className="text-2xl font-bold">3</div></CardContent>
          </Card>
        </div>

        {/* Search */}
        <div className="flex items-center gap-3 flex-wrap">
          <div className="relative flex-1 max-w-sm">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <input type="text" placeholder={t("audit.searchLogs")} className="w-full h-9 pl-9 pr-3 rounded-lg border bg-background text-sm" value={searchQuery} onChange={e => setSearchQuery(e.target.value)} />
          </div>
          <Button variant="outline" size="sm"><RefreshCw className="h-4 w-4 mr-1" /> Refresh</Button>
        </div>

        {/* Audit log table */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2"><Shield className="h-5 w-5 text-primary" /> Event Log</CardTitle>
            <CardDescription>Immutable audit trail of all system actions</CardDescription>
          </CardHeader>
          <CardContent>
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
                  {filtered.map((event) => (
                    <tr key={event.id} className="border-b last:border-0 hover:bg-accent/30">
                      <td className="py-3 pr-4 font-mono text-xs text-muted-foreground whitespace-nowrap">{event.timestamp}</td>
                      <td className="py-3 pr-4">
                        <Badge variant="outline" className={`text-[10px] h-5 font-mono ${actionColors[event.action] || ""}`}>
                          {event.action}
                        </Badge>
                      </td>
                      <td className="py-3 pr-4 font-mono text-xs">{event.actor}</td>
                      <td className="py-3 pr-4 font-mono text-xs text-muted-foreground max-w-[200px] truncate">{event.resource}</td>
                      <td className="py-3 pr-4">
                        {event.status === "success"
                          ? <Badge variant="outline" className="border-green-300 text-green-700 bg-green-50 text-[10px] h-5"><CheckCircle2 className="w-2.5 h-2.5 mr-1" />Success</Badge>
                          : <Badge variant="destructive" className="text-[10px] h-5">Failure</Badge>
                        }
                      </td>
                      <td className="py-3 text-xs text-muted-foreground max-w-[300px] truncate">{event.details}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </div>
    </OperatorShell>
  );
}

"use client";

import { useState } from "react";
import {
  CheckCircle2,
  AlertCircle,
  XCircle,
  Minus,
  Search,
  Server,
  Database,
  Cpu,
  HardDrive,
  Monitor,
  Activity,
  Network,
  Box,
  Container,
  Router,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";

interface ServiceRow {
  name: string;
  port: number;
  status: "running" | "stopped" | "error";
  category: string;
  health?: string;
  pid?: number;
  uptime?: string;
  cpu?: number;
  memory?: number;
}

const statusIcon = (status: string) => {
  switch (status) {
    case "running":
      return <CheckCircle2 className="w-3.5 h-3.5 text-green-500" />;
    case "stopped":
      return <XCircle className="w-3.5 h-3.5 text-red-400" />;
    case "error":
      return <AlertCircle className="w-3.5 h-3.5 text-amber-500" />;
    default:
      return <Minus className="w-3.5 h-3.5 text-gray-400" />;
  }
};

const categoryIcon = (cat: string) => {
  switch (cat) {
    case "core":
      return Server;
    case "database":
      return Database;
    case "llm":
      return Cpu;
    case "storage":
      return HardDrive;
    case "ui":
      return Monitor;
    case "workers":
      return Activity;
    case "network":
      return Network;
    case "docker":
      return Container;
    case "infra":
      return Box;
    default:
      return Router;
  }
};

interface ServiceHealthTableProps {
  services: ServiceRow[];
  loading?: boolean;
  compact?: boolean;
}

export function ServiceHealthTable({
  services,
  loading = false,
  compact = false,
}: ServiceHealthTableProps) {
  const [filter, setFilter] = useState("");
  const [categoryFilter, setCategoryFilter] = useState<string | null>(null);

  const categories = [...new Set(services.map((s) => s.category))].sort();

  const filtered = services.filter((s) => {
    if (filter && !s.name.toLowerCase().includes(filter.toLowerCase())) return false;
    if (categoryFilter && s.category !== categoryFilter) return false;
    return true;
  });

  const running = services.filter((s) => s.status === "running").length;
  const stopped = services.filter((s) => s.status === "stopped").length;
  const errored = services.filter((s) => s.status === "error").length;

  if (compact) {
    // Compact mode: just a mini summary with dots
    return (
      <div className="space-y-2">
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <CheckCircle2 className="w-3 h-3 text-green-500" /> {running} running
          </span>
          {stopped > 0 && (
            <span className="flex items-center gap-1">
              <XCircle className="w-3 h-3 text-red-400" /> {stopped} stopped
            </span>
          )}
          {errored > 0 && (
            <span className="flex items-center gap-1">
              <AlertCircle className="w-3 h-3 text-amber-500" /> {errored} degraded
            </span>
          )}
        </div>
        <div className="flex flex-wrap gap-1.5">
          {services.map((s) => {
            const CatIcon = categoryIcon(s.category);
            return (
              <div
                key={s.name}
                className={`flex items-center gap-1 px-2 py-0.5 rounded text-[10px] border ${
                  s.status === "running"
                    ? "border-green-200 bg-green-50 text-green-700 dark:border-green-800 dark:bg-green-950/30 dark:text-green-400"
                    : s.status === "error"
                      ? "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-800 dark:bg-amber-950/30 dark:text-amber-400"
                      : "border-red-200 bg-red-50 text-red-700 dark:border-red-800 dark:bg-red-950/30 dark:text-red-400"
                }`}
              >
                {statusIcon(s.status)}
                <CatIcon className="w-2.5 h-2.5 opacity-60" />
                <span className="font-medium">{s.name}</span>
                <span className="opacity-60">:{s.port}</span>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Summary bar */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3 text-xs">
          <span className="flex items-center gap-1 text-green-600 dark:text-green-400">
            <CheckCircle2 className="w-3.5 h-3.5" /> {running} running
          </span>
          {stopped > 0 && (
            <span className="flex items-center gap-1 text-red-500">
              <XCircle className="w-3.5 h-3.5" /> {stopped} stopped
            </span>
          )}
          {errored > 0 && (
            <span className="flex items-center gap-1 text-amber-500">
              <AlertCircle className="w-3.5 h-3.5" /> {errored} degraded
            </span>
          )}
          <span className="text-muted-foreground">· {services.length} total</span>
        </div>

        {/* Search */}
        <div className="relative">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3 h-3 text-muted-foreground" />
          <input
            type="text"
            placeholder="Filter services..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="h-7 pl-7 pr-2 text-xs border rounded bg-transparent w-44 focus:outline-none focus:ring-1 focus:ring-primary"
          />
        </div>
      </div>

      {/* Category pills */}
      <div className="flex gap-1.5 flex-wrap">
        <button
          onClick={() => setCategoryFilter(null)}
          className={`px-2 py-0.5 rounded text-[10px] font-medium border transition-colors ${
            categoryFilter === null
              ? "bg-primary text-primary-foreground border-primary"
              : "bg-transparent text-muted-foreground border-border hover:bg-accent"
          }`}
        >
          All
        </button>
        {categories.map((cat) => (
          <button
            key={cat}
            onClick={() => setCategoryFilter(cat)}
            className={`px-2 py-0.5 rounded text-[10px] font-medium border capitalize transition-colors ${
              categoryFilter === cat
                ? "bg-primary text-primary-foreground border-primary"
                : "bg-transparent text-muted-foreground border-border hover:bg-accent"
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      {/* Table */}
      <div className="border rounded-lg overflow-hidden">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b bg-muted/50">
              <th className="text-left py-2 px-3 font-medium text-muted-foreground">Status</th>
              <th className="text-left py-2 px-3 font-medium text-muted-foreground">Service</th>
              <th className="text-left py-2 px-3 font-medium text-muted-foreground hidden sm:table-cell">
                Port
              </th>
              <th className="text-left py-2 px-3 font-medium text-muted-foreground hidden md:table-cell">
                Category
              </th>
              <th className="text-right py-2 px-3 font-medium text-muted-foreground hidden lg:table-cell">
                CPU
              </th>
              <th className="text-right py-2 px-3 font-medium text-muted-foreground hidden lg:table-cell">
                Mem
              </th>
              <th className="text-right py-2 px-3 font-medium text-muted-foreground hidden md:table-cell">
                Uptime
              </th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={7} className="text-center py-6 text-muted-foreground">
                  No matching services
                </td>
              </tr>
            ) : (
              filtered.map((s) => {
                const CatIcon = categoryIcon(s.category);
                return (
                  <tr
                    key={s.name}
                    className="border-b last:border-0 hover:bg-muted/30 transition-colors"
                  >
                    <td className="py-2 px-3">{statusIcon(s.status)}</td>
                    <td className="py-2 px-3">
                      <div className="flex items-center gap-2">
                        <CatIcon className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
                        <div>
                          <div className="font-medium">{s.name}</div>
                          {s.health && s.health !== "unknown" && (
                            <div className="text-[10px] text-muted-foreground">{s.health}</div>
                          )}
                        </div>
                      </div>
                    </td>
                    <td className="py-2 px-3 font-mono text-muted-foreground hidden sm:table-cell">
                      :{s.port}
                    </td>
                    <td className="py-2 px-3 hidden md:table-cell">
                      <Badge variant="outline" className="text-[10px] capitalize">
                        {s.category}
                      </Badge>
                    </td>
                    <td className="py-2 px-3 text-right font-mono hidden lg:table-cell">
                      {s.cpu !== undefined ? `${s.cpu.toFixed(1)}%` : "—"}
                    </td>
                    <td className="py-2 px-3 text-right font-mono hidden lg:table-cell">
                      {s.memory !== undefined ? `${s.memory.toFixed(1)}%` : "—"}
                    </td>
                    <td className="py-2 px-3 text-right font-mono text-muted-foreground hidden md:table-cell">
                      {s.uptime || "—"}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// Map services API response to categorized rows
export function categorizeServices(services: any[]): ServiceRow[] {
  const categoryMap: Record<string, string[]> = {
    core: ["hindsight api", "control plane"],
    llm: ["ollama", "lm studio", "llm", "tei"],
    database: ["postgresql", "pg0", "pgvector"],
    workers: ["worker"],
    storage: ["memlord", "qdrant"],
    ui: ["grafana", "langfuse", "vector-admin"],
    network: ["cloudflare", "tunnel", "nginx"],
    docker: ["docker"],
    infra: ["jaeger", "prometheus", "loki", "tempo", "mimir", "lgtm"],
  };

  return services.map((s) => {
    const name = (s.name || "").toLowerCase();
    let category = "other";
    for (const [cat, keywords] of Object.entries(categoryMap)) {
      if (keywords.some((kw) => name.includes(kw))) {
        category = cat;
        break;
      }
    }
    return {
      name: s.name,
      port: s.port,
      status: s.status,
      category,
      health: s.health,
      pid: s.pid,
      uptime: s.uptime,
      cpu: s.cpu,
      memory: s.memory,
    };
  });
}

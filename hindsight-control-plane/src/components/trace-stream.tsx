"use client";

import { useState, useEffect, useRef } from "react";
import { Loader2, ArrowRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import Link from "next/link";

interface TraceEvent {
  id: string;
  task_type: string;
  status: string;
  items_count: number;
  created_at: string;
  retry_count: number;
  error_message: string | null;
  duration_ms?: number;
}

type AgeBand = "fresh" | "recent" | "stale" | "stuck";

function getAge(createdAt: string): { seconds: number; band: AgeBand; label: string } {
  const seconds = Math.floor((Date.now() - new Date(createdAt).getTime()) / 1000);
  if (seconds < 30) return { seconds, band: "fresh", label: `${seconds}s` };
  if (seconds < 120)
    return { seconds, band: "recent", label: `${Math.floor(seconds / 60)}m ${seconds % 60}s` };
  if (seconds < 300) return { seconds, band: "stale", label: `${Math.floor(seconds / 60)}m` };
  return {
    seconds,
    band: "stuck",
    label: `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`,
  };
}

const typeColors: Record<string, string> = {
  retain: "border-l-blue-500 bg-blue-50/50 dark:bg-blue-950/20",
  consolidate: "border-l-purple-500 bg-purple-50/50 dark:bg-purple-950/20",
  reflect: "border-l-emerald-500 bg-emerald-50/50 dark:bg-emerald-950/20",
  search: "border-l-cyan-500 bg-cyan-50/50 dark:bg-cyan-950/20",
  delete: "border-l-red-500 bg-red-50/50 dark:bg-red-950/20",
  mental_model: "border-l-amber-500 bg-amber-50/50 dark:bg-amber-950/20",
  default: "border-l-gray-400 bg-gray-50/50 dark:bg-gray-800/20",
};

function getTypeColor(type: string): string {
  const key = Object.keys(typeColors).find((k) => type?.toLowerCase().includes(k));
  return typeColors[key || "default"];
}

const statusBadge: Record<string, { label: string; className: string }> = {
  completed: { label: "Completed", className: "bg-green-500 text-white" },
  failed: { label: "Failed", className: "bg-red-500 text-white" },
  processing: { label: "Processing", className: "bg-blue-500 text-white" },
  pending: {
    label: "Pending",
    className: "bg-amber-500/20 text-amber-700 dark:text-amber-400 border border-amber-300",
  },
};

function TraceItem({ event, isNew }: { event: TraceEvent; isNew: boolean }) {
  const age = getAge(event.created_at);
  const borderColor = getTypeColor(event.task_type);
  const statusInfo = statusBadge[event.status] || {
    label: event.status,
    className: "bg-gray-500/20 text-gray-700 dark:text-gray-400",
  };
  const showError = event.status === "failed" && event.error_message;

  return (
    <div
      className={`border-l-2 pl-3 py-2 space-y-1 transition-all duration-500 ${
        isNew ? "animate-in slide-in-from-left-2" : ""
      } ${borderColor} rounded-r`}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <Badge variant="outline" className="text-[10px] font-mono h-5 px-1.5 shrink-0 uppercase">
            {event.task_type}
          </Badge>
          <span className={`text-[11px] font-medium px-1.5 py-0.5 rounded ${statusInfo.className}`}>
            {statusInfo.label}
          </span>
        </div>
        <span
          className={`text-[10px] font-mono shrink-0 ${
            age.band === "stuck"
              ? "text-red-500 font-bold animate-pulse"
              : age.band === "stale"
                ? "text-amber-500"
                : "text-muted-foreground"
          }`}
        >
          {age.label}
        </span>
      </div>
      <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
        <span className="font-mono">{event.id.substring(0, 8)}</span>
        <span>·</span>
        <span>{event.items_count} items</span>
        {event.retry_count > 0 && (
          <>
            <span>·</span>
            <span className="text-amber-500">retry {event.retry_count}</span>
          </>
        )}
      </div>
      {showError && (
        <p className="text-[10px] text-red-500 truncate leading-tight" title={event.error_message!}>
          {event.error_message}
        </p>
      )}
    </div>
  );
}

interface TraceStreamProps {
  events: TraceEvent[];
  loading?: boolean;
  maxItems?: number;
  showViewAll?: boolean;
}

export function TraceStream({
  events,
  loading = false,
  maxItems = 8,
  showViewAll = true,
}: TraceStreamProps) {
  const prevEventCount = useRef(events.length);
  const [newIds, setNewIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (events.length > prevEventCount.current) {
      const added = events.slice(0, events.length - prevEventCount.current);
      const ids = new Set(added.map((e) => e.id));
      setNewIds(ids);
      const timer = setTimeout(() => setNewIds(new Set()), 1500);
      prevEventCount.current = events.length;
      return () => clearTimeout(timer);
    }
  }, [events.length]);

  if (loading && events.length === 0) {
    return (
      <div className="flex items-center justify-center py-8 text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin mr-2" /> Loading trace...
      </div>
    );
  }

  if (events.length === 0) {
    return (
      <div className="text-sm text-muted-foreground text-center py-8">
        No events yet — operations will appear here in real time
      </div>
    );
  }

  const displayEvents = events.slice(0, maxItems);

  return (
    <div className="space-y-1">
      {displayEvents.map((event) => (
        <TraceItem key={event.id} event={event} isNew={newIds.has(event.id)} />
      ))}
      {showViewAll && events.length > maxItems && (
        <div className="pt-2">
          <Link href="/runs">
            <Button variant="ghost" size="sm" className="w-full text-xs">
              View all {events.length} operations <ArrowRight className="w-3 h-3 ml-1" />
            </Button>
          </Link>
        </div>
      )}
    </div>
  );
}

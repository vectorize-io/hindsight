"use client";

import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2, TrendingUp, TrendingDown } from "lucide-react";

// ── Trend data store (localStorage-backed) ──────────────────────────

const STORAGE_KEY = "cockpit-metric-history";

interface MetricSnapshot {
  timestamp: number;
  values: Record<string, number>;
}

function loadHistory(): MetricSnapshot[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function appendSnapshot(snapshot: MetricSnapshot) {
  const history = loadHistory();
  history.push(snapshot);
  // Keep last 60 snapshots (~10 minutes at 10s intervals)
  while (history.length > 60) history.shift();
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
  } catch {
    // localStorage may be full
  }
}

// ── CSS sparkline (pure div bars) ──────────────────────────────────

function CssSparkline({
  data,
  height = 24,
  width = 64,
  color = "var(--primary)",
}: {
  data: number[];
  height?: number;
  width?: number;
  color?: string;
}) {
  if (data.length < 2) return null;

  const max = Math.max(...data, 1);
  const barW = Math.max(2, Math.floor(width / data.length) - 1);
  const bars = data.map((v, i) => {
    const h = Math.max(2, (v / max) * height);
    return (
      <div
        key={i}
        className="rounded-sm transition-all duration-300"
        style={{
          width: barW,
          height,
          background: `linear-gradient(to top, ${color}22, ${color})`,
          borderRadius: 1,
        }}
      >
        <div
          className="rounded-sm"
          style={{
            width: "100%",
            height,
            background: `linear-gradient(to top, ${color}66, ${color})`,
            clipPath: `inset(${height - h}px 0 0 0)`,
            borderRadius: 1,
          }}
        />
      </div>
    );
  });

  return (
    <div className="flex items-end gap-px" style={{ height, width }}>
      {bars}
    </div>
  );
}

// ── Metric Tile ────────────────────────────────────────────────────

interface MetricTileProps {
  title: string;
  value: string | number;
  unit?: string;
  icon?: React.ReactNode;
  trend?: number; // percentage change
  history: number[];
  loading?: boolean;
  color?: string;
}

export function MetricTile({
  title,
  value,
  unit,
  icon,
  trend,
  history,
  loading = false,
  color,
}: MetricTileProps) {
  const isUp = trend !== undefined && trend > 0;
  const isDown = trend !== undefined && trend < 0;

  return (
    <Card>
      <CardHeader className="pb-1 pt-3 px-3">
        <CardTitle className="text-[11px] font-medium text-muted-foreground flex items-center gap-1.5">
          {icon}
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="px-3 pb-3 space-y-1.5">
        <div className="flex items-baseline gap-1.5">
          {loading ? (
            <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
          ) : (
            <>
              <span className="text-xl font-bold tabular-nums">{value}</span>
              {unit && <span className="text-[11px] text-muted-foreground">{unit}</span>}
            </>
          )}
        </div>
        <div className="flex items-center justify-between">
          {history.length > 0 && (
            <CssSparkline data={history} height={20} width={56} color={color} />
          )}
          {trend !== undefined && trend !== 0 && (
            <span
              className={`flex items-center gap-0.5 text-[10px] font-medium ${
                isUp ? "text-green-600" : isDown ? "text-red-600" : "text-muted-foreground"
              }`}
            >
              {isUp ? (
                <TrendingUp className="w-2.5 h-2.5" />
              ) : (
                <TrendingDown className="w-2.5 h-2.5" />
              )}
              {Math.abs(trend).toFixed(0)}%
            </span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

// ── History Accumulator Hook ────────────────────────────────────────

export function useMetricHistory(key: string, currentValue: number, intervalMs = 10000) {
  const [history, setHistory] = useState<number[]>(() => {
    const saved = loadHistory();
    return saved.map((s) => s.values[key]).filter((v): v is number => v !== undefined);
  });

  useEffect(() => {
    const snapshot: MetricSnapshot = {
      timestamp: Date.now(),
      values: { [key]: currentValue },
    };
    appendSnapshot(snapshot);
    setHistory((prev) => {
      const next = [...prev, currentValue];
      return next.length > 60 ? next.slice(-60) : next;
    });
  }, [currentValue, key]);

  // Refresh from storage on mount if current value is 0 (no data yet)
  useEffect(() => {
    if (currentValue === 0 && history.length === 0) {
      const saved = loadHistory();
      const vals = saved.map((s) => s.values[key]).filter((v): v is number => v !== undefined);
      if (vals.length > 0) setHistory(vals);
    }
  }, []);

  return history;
}

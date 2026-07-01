"use client";

import { Cpu, HardDrive, Activity, Zap } from "lucide-react";

interface EngineMetric {
  label: string;
  value: number;
  max: number;
  unit: string;
  color: "green" | "blue" | "amber" | "purple";
  icon?: "cpu" | "memory" | "activity" | "zap";
}

const colorMap = {
  green: { bg: "bg-green-500", track: "bg-green-100 dark:bg-green-950/50" },
  blue: { bg: "bg-blue-500", track: "bg-blue-100 dark:bg-blue-950/50" },
  amber: { bg: "bg-amber-500", track: "bg-amber-100 dark:bg-amber-950/50" },
  purple: { bg: "bg-purple-500", track: "bg-purple-100 dark:bg-purple-950/50" },
};

const iconMap = {
  cpu: Cpu,
  memory: HardDrive,
  activity: Activity,
  zap: Zap,
};

function EngineBar({ metric }: { metric: EngineMetric }) {
  const pct = metric.max > 0 ? Math.min((metric.value / metric.max) * 100, 100) : 0;
  const IconComp = metric.icon ? iconMap[metric.icon] : null;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="flex items-center gap-1.5 text-muted-foreground">
          {IconComp && <IconComp className="w-3 h-3" />}
          {metric.label}
        </span>
        <span className="font-mono text-xs font-medium">
          {metric.value}
          {metric.unit}
        </span>
      </div>
      <div className={`h-2 rounded-full ${colorMap[metric.color].track} overflow-hidden`}>
        <div
          className={`h-full rounded-full ${colorMap[metric.color].bg} transition-all duration-500 ease-out`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

interface EngineLoadBarProps {
  cpu?: number;
  memory?: number;
  workers?: { active: number; total: number };
  llmLoad?: { active: number; max: number };
}

export function EngineLoadBar({ cpu, memory, workers, llmLoad }: EngineLoadBarProps) {
  const metrics: EngineMetric[] = [];

  if (cpu !== undefined) {
    metrics.push({
      label: "CPU",
      value: cpu,
      max: 100,
      unit: "%",
      color: cpu > 80 ? "amber" : "green",
      icon: "cpu",
    });
  }

  if (memory !== undefined) {
    metrics.push({
      label: "Memory",
      value: memory,
      max: 100,
      unit: "%",
      color: memory > 80 ? "amber" : "blue",
      icon: "memory",
    });
  }

  if (workers) {
    metrics.push({
      label: "Workers",
      value: workers.active,
      max: workers.total,
      unit: `/${workers.total}`,
      color: workers.active > 0 ? "green" : "amber",
      icon: "activity",
    });
  }

  if (llmLoad) {
    metrics.push({
      label: "LLM",
      value: llmLoad.active,
      max: llmLoad.max,
      unit: `/${llmLoad.max}`,
      color: llmLoad.active >= llmLoad.max ? "amber" : "purple",
      icon: "zap",
    });
  }

  if (metrics.length === 0) return null;

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {metrics.map((m) => (
        <EngineBar key={m.label} metric={m} />
      ))}
    </div>
  );
}

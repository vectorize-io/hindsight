"use client";

import { Brain, Search, GitMerge, Network, Database, Filter } from "lucide-react";

interface Strategy {
  id: string;
  name: string;
  description: string;
  percentage: number;
  color: string;
  icon: "brain" | "search" | "merge" | "network" | "database" | "filter";
  active: boolean;
}

const iconMap = {
  brain: Brain,
  search: Search,
  merge: GitMerge,
  network: Network,
  database: Database,
  filter: Filter,
};

const defaultStrategies: Strategy[] = [
  {
    id: "semantic",
    name: "Semantic Search",
    description: "Vector embedding similarity",
    percentage: 40,
    color: "#8b5cf6",
    icon: "brain",
    active: true,
  },
  {
    id: "keyword",
    name: "Keyword BM25",
    description: "Full-text lexical matching",
    percentage: 25,
    color: "#06b6d4",
    icon: "search",
    active: true,
  },
  {
    id: "graph",
    name: "Graph Traversal",
    description: "Entity-relation links",
    percentage: 15,
    color: "#10b981",
    icon: "network",
    active: true,
  },
  {
    id: "temporal",
    name: "Temporal Decay",
    description: "Recency-weighted ranking",
    percentage: 10,
    color: "#f59e0b",
    icon: "merge",
    active: true,
  },
  {
    id: "entity",
    name: "Entity Resolution",
    description: "Canonical entity linking",
    percentage: 7,
    color: "#ef4444",
    icon: "database",
    active: false,
  },
  {
    id: "reranker",
    name: "Cross-Encoder Rerank",
    description: "Neural relevance scoring",
    percentage: 3,
    color: "#ec4899",
    icon: "filter",
    active: true,
  },
];

interface FusionStrategyMixProps {
  strategies?: Strategy[];
  compact?: boolean;
}

export function FusionStrategyMix({
  strategies = defaultStrategies,
  compact = false,
}: FusionStrategyMixProps) {
  const active = strategies.filter((s) => s.active);
  const totalPct = active.reduce((sum, s) => sum + s.percentage, 0);

  if (compact) {
    return (
      <div className="space-y-2">
        <div className="flex h-2 rounded-full overflow-hidden">
          {active.map((s) => (
            <div
              key={s.id}
              className="transition-all duration-500"
              style={{
                width: `${(s.percentage / totalPct) * 100}%`,
                backgroundColor: s.color,
              }}
            />
          ))}
        </div>
        <div className="flex flex-wrap gap-x-3 gap-y-1">
          {active.map((s) => {
            const Icon = iconMap[s.icon];
            return (
              <div key={s.id} className="flex items-center gap-1 text-[10px] text-muted-foreground">
                <Icon className="w-2.5 h-2.5" style={{ color: s.color }} />
                <span>{s.name}</span>
                <span className="font-mono font-medium">
                  {Math.round((s.percentage / totalPct) * 100)}%
                </span>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Stacked horizontal bar */}
      <div className="h-4 rounded-full overflow-hidden flex">
        {active.map((s) => (
          <div
            key={s.id}
            className="relative transition-all duration-500 first:rounded-l-full last:rounded-r-full"
            style={{
              width: `${(s.percentage / totalPct) * 100}%`,
              backgroundColor: s.color,
            }}
            title={`${s.name}: ${Math.round((s.percentage / totalPct) * 100)}%`}
          />
        ))}
      </div>

      {/* Legend */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
        {active.map((s) => {
          const Icon = iconMap[s.icon];
          const pct = Math.round((s.percentage / totalPct) * 100);
          return (
            <div key={s.id} className="flex items-start gap-2 p-1.5 rounded">
              <div
                className="w-2 h-2 rounded-full mt-1 shrink-0"
                style={{ backgroundColor: s.color }}
              />
              <div className="min-w-0">
                <div className="flex items-center gap-1.5">
                  <Icon className="w-3 h-3 text-muted-foreground shrink-0" />
                  <span className="text-xs font-medium truncate">{s.name}</span>
                </div>
                <div className="flex items-center gap-1.5 mt-0.5">
                  <span className="text-[11px] font-mono font-bold">{pct}%</span>
                  <span className="text-[10px] text-muted-foreground truncate">
                    {s.description}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

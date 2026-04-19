"use client";

import { cn } from "@/lib/utils";

export type FactType = "world" | "experience" | "observation";

export const ALL_FACT_TYPES: FactType[] = ["world", "experience", "observation"];

const FACT_TYPE_CONFIG: Record<
  FactType,
  { label: string; active: string; inactive: string; dot: string }
> = {
  // Aligned with the stats-chart palette (bank-stats-view.tsx CHART_COLORS).
  world: {
    label: "World",
    active:
      "bg-violet-500/15 text-violet-700 border-violet-400 dark:text-violet-300 dark:border-violet-500",
    inactive:
      "border-border text-muted-foreground hover:border-violet-300 hover:text-violet-600 dark:hover:text-violet-400",
    dot: "bg-violet-500",
  },
  experience: {
    label: "Experience",
    active: "bg-pink-500/15 text-pink-700 border-pink-400 dark:text-pink-300 dark:border-pink-500",
    inactive:
      "border-border text-muted-foreground hover:border-pink-300 hover:text-pink-600 dark:hover:text-pink-400",
    dot: "bg-pink-500",
  },
  observation: {
    label: "Observation",
    active:
      "bg-indigo-500/15 text-indigo-700 border-indigo-400 dark:text-indigo-300 dark:border-indigo-500",
    inactive:
      "border-border text-muted-foreground hover:border-indigo-300 hover:text-indigo-600 dark:hover:text-indigo-400",
    dot: "bg-indigo-500",
  },
};

function FactTypePill({
  ft,
  active,
  onToggle,
}: {
  ft: FactType;
  active: boolean;
  onToggle: () => void;
}) {
  const cfg = FACT_TYPE_CONFIG[ft];
  return (
    <button
      type="button"
      onClick={onToggle}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium transition-all",
        active ? cfg.active : cfg.inactive
      )}
    >
      <span
        className={cn("h-1.5 w-1.5 rounded-full", active ? cfg.dot : "bg-muted-foreground/50")}
      />
      {cfg.label}
    </button>
  );
}

/**
 * Inline pill-toggle fact-type filter for filter bars.
 * An empty selection means "all types included".
 */
export function FactTypeFilter({
  value,
  onChange,
  label = "Fact types:",
}: {
  value: FactType[];
  onChange: (next: FactType[]) => void;
  label?: string;
}) {
  const toggle = (ft: FactType) =>
    onChange(value.includes(ft) ? value.filter((f) => f !== ft) : [...value, ft]);

  return (
    <div className="flex items-center gap-2">
      {label && <span className="text-sm font-medium text-muted-foreground">{label}</span>}
      <div className="flex gap-1.5">
        {ALL_FACT_TYPES.map((ft) => (
          <FactTypePill key={ft} ft={ft} active={value.includes(ft)} onToggle={() => toggle(ft)} />
        ))}
      </div>
    </div>
  );
}

/**
 * Pill-toggle group for use inside forms/dialogs.
 */
export function FactTypeCheckboxGroup({
  value,
  onChange,
}: {
  value: FactType[];
  onChange: (next: FactType[]) => void;
}) {
  const toggle = (ft: FactType) =>
    onChange(value.includes(ft) ? value.filter((f) => f !== ft) : [...value, ft]);

  return (
    <div className="flex flex-wrap gap-1.5">
      {ALL_FACT_TYPES.map((ft) => (
        <FactTypePill key={ft} ft={ft} active={value.includes(ft)} onToggle={() => toggle(ft)} />
      ))}
    </div>
  );
}

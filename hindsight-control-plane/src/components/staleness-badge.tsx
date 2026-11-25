"use client";

import { useTranslations } from "next-intl";
import { cn } from "@/lib/utils";
import type { RefreshAttempt } from "@/lib/use-refresh-attempts";

type TriggerLike = {
  refresh_after_consolidation?: boolean;
  refresh_cron?: string | null;
};

/**
 * The one staleness indicator, shared by pages and mental models.
 *
 * They are the same object — a page is a mental model with a place in the tree —
 * so they get the same answer rendered the same way, rather than a badge here and
 * a differently-worded dot there.
 *
 * `is_stale` is the API's per-model answer: a memory in *this* model's scope has
 * been written since it last read the memories. It is the same check that decides
 * whether a scheduled refresh does any work, so a flagged model is one a refresh
 * would actually rewrite (#3291).
 *
 * Staleness is two states, two colours. An earlier revision split "stale" into two
 * colours — one for models that refresh themselves and one for manual ones — but
 * every surface that shows this badge already shows "Next refresh: …" beside it,
 * so the colour restated a fact that was on screen anyway and could only be
 * decoded by hovering. The distinction lives in the tooltip instead. A refresh in
 * flight that never failed is blue, so it cannot be read as a warning.
 *
 * `variant` picks the presentation, not the meaning: compact rows (the pages tree,
 * the mental-model list) have no room for a label and use the dot; headers and
 * detail panes use the labelled badge.
 *
 * A refresh in flight (`attempt`) outranks both, and says which kind it is:
 * queued, refreshing, or retrying. Only one that already failed once is a retry
 * (#6864); any of them means the model is not paused yet, whatever the failure
 * stamp says — the worker has not given up on it.
 *
 * `refreshFailedAt` outranks staleness, in red: once a refresh has failed, the
 * automatic triggers leave the model alone until one succeeds (#4532), so "stale,
 * refreshes itself" would be a promise nothing is going to keep. It is the same
 * signal on every surface — the list, the pages tree and the detail header — so a
 * model that has stopped refreshing itself cannot be read as merely behind.
 */
type State = "queued" | "refreshing" | "retrying" | "paused" | "stale" | "inSync";

const DOT: Record<State, string> = {
  queued: "bg-sky-500",
  refreshing: "bg-sky-500",
  retrying: "bg-amber-500",
  paused: "bg-red-500",
  stale: "bg-amber-500",
  inSync: "bg-emerald-500",
};

const TEXT: Record<State, string> = {
  queued: "text-sky-600 dark:text-sky-400",
  refreshing: "text-sky-600 dark:text-sky-400",
  retrying: "text-amber-600 dark:text-amber-400",
  paused: "text-red-600 dark:text-red-400",
  stale: "text-amber-600 dark:text-amber-400",
  inSync: "text-emerald-600 dark:text-emerald-400",
};

const CHIP: Record<State, string> = {
  queued: "bg-sky-500/15 text-sky-700 dark:text-sky-400",
  refreshing: "bg-sky-500/15 text-sky-700 dark:text-sky-400",
  retrying: "bg-amber-500/15 text-amber-700 dark:text-amber-400",
  paused: "bg-red-500/15 text-red-700 dark:text-red-400",
  stale: "bg-amber-500/15 text-amber-700 dark:text-amber-400",
  inSync: "bg-green-500/15 text-green-700 dark:text-green-400",
};

export function StalenessBadge({
  isStale,
  trigger,
  refreshFailedAt,
  attempt,
  variant = "badge",
  className,
}: {
  isStale: boolean | null | undefined;
  trigger?: TriggerLike | null;
  /** ISO time of the last failed refresh, or null when the last one succeeded. */
  refreshFailedAt?: string | null;
  /** A refresh that is queued or running for this model, if any. */
  attempt?: RefreshAttempt | null;
  variant?: "badge" | "dot" | "inline";
  className?: string;
}) {
  const t = useTranslations("staleness");

  const state: State | null = attempt
    ? attempt.retry
      ? "retrying"
      : attempt.running
        ? "refreshing"
        : "queued"
    : refreshFailedAt
      ? "paused"
      : // Null/undefined means the surface did not ask for staleness — render
        // nothing rather than guessing, so "unknown" never reads as "in sync".
        isStale === null || isStale === undefined
        ? null
        : isStale
          ? "stale"
          : "inSync";
  if (!state) return null;

  const selfHealing = Boolean(
    trigger?.refresh_cron?.trim() || trigger?.refresh_after_consolidation
  );
  // Literal keys, not `${state}Title`: the used-keys test can only check those.
  const titles: Record<State, string> = {
    queued: t("queuedTitle"),
    refreshing: t("refreshingTitle"),
    retrying: t("retryingTitle"),
    paused: t("pausedTitle"),
    stale: selfHealing ? t("staleAutoTitle") : t("staleManualTitle"),
    inSync: t("inSyncTitle"),
  };
  const labels: Record<State, string> = {
    queued: t("queued"),
    refreshing: t("refreshing"),
    retrying: t("retrying"),
    paused: t("paused"),
    stale: t("stale"),
    inSync: t("inSync"),
  };
  const title = titles[state];
  const label = labels[state];

  if (variant === "inline") {
    // Dot + label with no chip around it, for the freshness line: there the status
    // is one clause in a sentence, and a bordered pill would out-shout the rest.
    return (
      <span
        className={cn("inline-flex items-center gap-1.5 font-medium", TEXT[state], className)}
        title={title}
      >
        <span className={cn("w-1.5 h-1.5 rounded-full", DOT[state])} />
        {label}
      </span>
    );
  }

  if (variant === "dot") {
    return (
      <span
        className={cn("w-1.5 h-1.5 rounded-full flex-shrink-0", DOT[state], className)}
        title={title}
      />
    );
  }

  return (
    <span
      className={cn(
        "px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide whitespace-nowrap",
        CHIP[state],
        className
      )}
      title={title}
    >
      {label}
    </span>
  );
}

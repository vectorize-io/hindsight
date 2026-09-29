"use client";

import { useTranslations } from "next-intl";
import { cn } from "@/lib/utils";

type TriggerLike = {
  refresh_after_consolidation?: boolean;
  refresh_cron?: string | null;
};

/**
 * The one staleness indicator, shared by knowledge pages and mental models.
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
 * Two states, two colours. An earlier revision split "stale" into two colours —
 * one for models that refresh themselves and one for manual ones — but every
 * surface that shows this badge already shows "Next refresh: …" beside it, so the
 * colour restated a fact that was on screen anyway and could only be decoded by
 * hovering. The distinction lives in the tooltip instead.
 *
 * `variant` picks the presentation, not the meaning: compact rows (the pages tree,
 * the mental-model list) have no room for a label and use the dot; headers and
 * detail panes use the labelled badge.
 *
 * `retrying` outranks both: while the worker still has attempts left, the model
 * is not paused yet, whatever the failure stamp says.
 *
 * `refreshFailedAt` outranks staleness, in red: once a refresh has failed, the
 * automatic triggers leave the model alone until one succeeds (#4532), so "stale,
 * refreshes itself" would be a promise nothing is going to keep. It is the same
 * signal on every surface — the list, the pages tree and the detail header — so a
 * model that has stopped refreshing itself cannot be read as merely behind.
 */
export function StalenessBadge({
  isStale,
  trigger,
  refreshFailedAt,
  retrying = false,
  variant = "badge",
  className,
}: {
  isStale: boolean | null | undefined;
  trigger?: TriggerLike | null;
  /** ISO time of the last failed refresh, or null when the last one succeeded. */
  refreshFailedAt?: string | null;
  /** True while a refresh attempt for this model is queued or running. */
  retrying?: boolean;
  variant?: "badge" | "dot" | "inline";
  className?: string;
}) {
  const t = useTranslations("staleness");

  const paused = Boolean(refreshFailedAt) && !retrying;

  // Null/undefined means the surface did not ask for staleness — render nothing
  // rather than guessing, so "unknown" never reads as "in sync". A paused model
  // still shows: that one is about the refresh, not about the data.
  if (!paused && !retrying && (isStale === null || isStale === undefined)) return null;

  const selfHealing = Boolean(
    trigger?.refresh_cron?.trim() || trigger?.refresh_after_consolidation
  );
  const title = retrying
    ? t("retryingTitle")
    : paused
      ? t("pausedTitle")
      : !isStale
        ? t("inSyncTitle")
        : selfHealing
          ? t("staleAutoTitle")
          : t("staleManualTitle");

  if (variant === "inline") {
    // Dot + label with no chip around it, for the freshness line: there the status
    // is one clause in a sentence, and a bordered pill would out-shout the rest.
    return (
      <span
        className={cn(
          "inline-flex items-center gap-1.5 font-medium",
          retrying
            ? "text-amber-600 dark:text-amber-400"
            : paused
              ? "text-red-600 dark:text-red-400"
              : isStale
                ? "text-amber-600 dark:text-amber-400"
                : "text-emerald-600 dark:text-emerald-400",
          className
        )}
        title={title}
      >
        <span
          className={cn(
            "w-1.5 h-1.5 rounded-full",
            retrying
              ? "bg-amber-500"
              : paused
                ? "bg-red-500"
                : isStale
                  ? "bg-amber-500"
                  : "bg-emerald-500"
          )}
        />
        {retrying ? t("retrying") : paused ? t("paused") : isStale ? t("stale") : t("inSync")}
      </span>
    );
  }

  if (variant === "dot") {
    return (
      <span
        className={cn(
          "w-1.5 h-1.5 rounded-full flex-shrink-0",
          retrying
            ? "bg-amber-500"
            : paused
              ? "bg-red-500"
              : isStale
                ? "bg-amber-500"
                : "bg-emerald-500",
          className
        )}
        title={title}
      />
    );
  }

  return (
    <span
      className={cn(
        "px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide whitespace-nowrap",
        retrying || (!paused && isStale)
          ? "bg-amber-500/15 text-amber-700 dark:text-amber-400"
          : paused
            ? "bg-red-500/15 text-red-700 dark:text-red-400"
            : isStale
              ? "bg-amber-500/15 text-amber-700 dark:text-amber-400"
              : "bg-green-500/15 text-green-700 dark:text-green-400",
        className
      )}
      title={title}
    >
      {retrying ? t("retrying") : paused ? t("paused") : isStale ? t("stale") : t("inSync")}
    </span>
  );
}

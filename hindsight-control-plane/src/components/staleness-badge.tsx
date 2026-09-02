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
 */
export function StalenessBadge({
  isStale,
  trigger,
  variant = "badge",
  className,
}: {
  isStale: boolean | null | undefined;
  trigger?: TriggerLike | null;
  variant?: "badge" | "dot" | "inline";
  className?: string;
}) {
  const t = useTranslations("staleness");

  // Null/undefined means the surface did not ask for staleness — render nothing
  // rather than guessing, so "unknown" never reads as "in sync".
  if (isStale === null || isStale === undefined) return null;

  const selfHealing = Boolean(
    trigger?.refresh_cron?.trim() || trigger?.refresh_after_consolidation
  );
  const title = !isStale
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
          isStale ? "text-amber-600 dark:text-amber-400" : "text-emerald-600 dark:text-emerald-400",
          className
        )}
        title={title}
      >
        <span
          className={cn("w-1.5 h-1.5 rounded-full", isStale ? "bg-amber-500" : "bg-emerald-500")}
        />
        {isStale ? t("stale") : t("inSync")}
      </span>
    );
  }

  if (variant === "dot") {
    return (
      <span
        className={cn(
          "w-1.5 h-1.5 rounded-full flex-shrink-0",
          isStale ? "bg-amber-500" : "bg-emerald-500",
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
        isStale
          ? "bg-amber-500/15 text-amber-700 dark:text-amber-400"
          : "bg-green-500/15 text-green-700 dark:text-green-400",
        className
      )}
      title={title}
    >
      {isStale ? t("stale") : t("inSync")}
    </span>
  );
}

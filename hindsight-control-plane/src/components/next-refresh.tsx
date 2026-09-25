"use client";

import { useTranslations } from "next-intl";
import { nextCronRun } from "@/lib/cron";
import { formatRelativeTime } from "@/lib/relative-time";

type TriggerLike = {
  refresh_after_consolidation?: boolean;
  refresh_cron?: string | null;
};

/**
 * Inline "next refresh" value derived from a mental model's trigger:
 * - cron schedule -> relative time of the next UTC run (absolute UTC on hover)
 * - after consolidation -> "after consolidation"
 * - otherwise (manual) -> "on demand"
 *
 * Renders only the value; call sites supply their own "Next refresh" label.
 */
export function NextRefresh({
  trigger,
  refreshFailedAt,
  attempt,
  className,
}: {
  trigger?: TriggerLike | null;
  /** ISO time of the last failed refresh, or null when the last one succeeded. */
  refreshFailedAt?: string | null;
  /** A refresh that is queued or running for this model, if any. */
  attempt?: { nextAttemptAt: string | null } | null;
  className?: string;
}) {
  const t = useTranslations("mentalModels");
  const cron = trigger?.refresh_cron?.trim();

  // An attempt that has not finished outranks everything else: the failure stamp
  // is written on the first failed attempt, while the worker still has retries
  // left, so the next thing to happen is that retry — and its time is known.
  if (attempt) {
    return (
      <span className={className}>
        {attempt.nextAttemptAt
          ? t("nextRefreshRetrying", { time: formatRelativeTime(attempt.nextAttemptAt) })
          : t("nextRefreshRunning")}
      </span>
    );
  }
  // A failed refresh pauses the automatic triggers until one succeeds (#4532), so
  // naming the next scheduled run here would promise something nothing will do.
  if (refreshFailedAt) return <span className={className}>{t("nextRefreshPaused")}</span>;

  if (cron) {
    const next = nextCronRun(cron);
    if (!next) return <span className={className}>{t("nextRefreshInvalid")}</span>;
    const utc = next.toLocaleString("en-GB", {
      timeZone: "UTC",
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
    return (
      <span className={className} title={`${utc} UTC`}>
        {formatRelativeTime(next.toISOString())}
      </span>
    );
  }

  return (
    <span className={className}>
      {trigger?.refresh_after_consolidation ? t("nextRefreshAuto") : t("nextRefreshManual")}
    </span>
  );
}

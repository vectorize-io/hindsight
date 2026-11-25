"use client";

import { useTranslations } from "next-intl";
import { nextCronRun } from "@/lib/cron";
import { formatRelativeTime } from "@/lib/relative-time";
import type { RefreshAttempt } from "@/lib/use-refresh-attempts";

type TriggerLike = {
  refresh_after_consolidation?: boolean;
  refresh_cron?: string | null;
};

/**
 * Inline "next refresh" value. A refresh in flight wins: queued (with its start
 * time), refreshing now, or retrying — the last only after a failed attempt.
 * Otherwise it is derived from the mental model's trigger:
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
  attempt?: RefreshAttempt | null;
  className?: string;
}) {
  const t = useTranslations("mentalModels");
  const cron = trigger?.refresh_cron?.trim();

  // A refresh in flight outranks everything else: it is the next thing that will
  // happen, and its time is known. Only one that already failed is a retry
  // (#6864) — a refresh held back by its minimum interval is merely scheduled.
  if (attempt) {
    const due =
      attempt.nextAttemptAt && new Date(attempt.nextAttemptAt).getTime() > Date.now()
        ? formatRelativeTime(attempt.nextAttemptAt)
        : null;
    const text = attempt.running
      ? attempt.retry
        ? t("nextRefreshRetryingNow")
        : t("nextRefreshRunning")
      : due
        ? attempt.retry
          ? t("nextRefreshRetrying", { time: due })
          : t("nextRefreshQueued", { time: due })
        : attempt.retry
          ? t("nextRefreshRetryingSoon")
          : t("nextRefreshQueuedSoon");
    return <span className={className}>{text}</span>;
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

"use client";

import { useTranslations } from "next-intl";
import { formatAbsoluteDateTime, formatRelativeTime, formatWatermark } from "@/lib/relative-time";
import { cn } from "@/lib/utils";
import { NextRefresh } from "./next-refresh";
import { StalenessBadge } from "./staleness-badge";

type TriggerLike = {
  refresh_after_consolidation?: boolean;
  refresh_cron?: string | null;
  fact_types?: string[];
};

function Sep() {
  return (
    <span aria-hidden className="text-muted-foreground/40">
      ·
    </span>
  );
}

/**
 * A model's freshness as one quiet line, rather than a row of competing pills.
 *
 * This replaced a header that carried five separate bordered chips — status,
 * refreshed-at, memories-read-to, scope and next-refresh — which wrapped onto two
 * rows, used three different chip styles, and gave the least important fact (the
 * next scheduled run, in blue) the loudest treatment. Everything here is one size
 * and one colour except the status dot, so the eye lands on the status first and
 * the supporting detail reads as a sentence.
 *
 * `read to` is shown whenever the model has a watermark, and omitted only when it
 * genuinely has none — a model no refresh has stamped yet. An earlier revision also
 * hid it when it matched `refreshed`, to avoid saying the same thing twice, but that
 * made an absent clause mean two different things ("same as the refresh" or "never
 * stamped") with no way to tell them apart, and made the line look arbitrarily
 * different from one row to the next.
 *
 * The redundancy that hiding was working around is better solved by formatting: the
 * refresh is an *age* ("4 hours ago") and the watermark is a *position*
 * (`formatWatermark`), so the two never collapse to the same string the way
 * "refreshed 4 hours ago · read to 4 hours ago" did for timestamps two hours apart.
 */
export function FreshnessLine({
  isStale,
  trigger,
  lastRefreshedAt,
  lastMemorySeenAt,
  className,
}: {
  isStale: boolean | null | undefined;
  trigger?: TriggerLike | null;
  lastRefreshedAt: string | null;
  lastMemorySeenAt?: string | null;
  className?: string;
}) {
  const t = useTranslations("mentalModels");

  const factTypes = trigger?.fact_types?.length ? trigger.fact_types.join(", ") : null;

  return (
    <div
      className={cn(
        "flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground",
        className
      )}
    >
      <StalenessBadge isStale={isStale} trigger={trigger} variant="inline" />
      {lastRefreshedAt && (
        <>
          {isStale !== null && isStale !== undefined && <Sep />}
          <span title={formatAbsoluteDateTime(lastRefreshedAt)}>
            {t("freshnessRefreshed", { time: formatRelativeTime(lastRefreshedAt) })}
          </span>
        </>
      )}
      {lastMemorySeenAt && (
        <>
          <Sep />
          <span title={formatAbsoluteDateTime(lastMemorySeenAt)}>
            {t("freshnessReadTo", { time: formatWatermark(lastMemorySeenAt) })}
          </span>
        </>
      )}
      <Sep />
      <span>
        {t("freshnessNext")} <NextRefresh trigger={trigger} />
      </span>
      {factTypes && (
        <>
          <Sep />
          <span title={t("scopeFactTypesTitle")}>{factTypes}</span>
        </>
      )}
    </div>
  );
}

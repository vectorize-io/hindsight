"use client";

import { useTranslations } from "next-intl";
import cronstrue from "cronstrue";
import { CronExpressionParser } from "cron-parser";
import { formatRelativeTime } from "@/lib/relative-time";

/**
 * Live preview for a cron expression: a human-readable description plus the next
 * few run times. Schedules are evaluated in UTC (matching the API, which runs
 * croniter in UTC), and each run is also shown in the viewer's local time so the
 * UTC offset is obvious.
 */
export function CronSchedulePreview({ cron }: { cron: string }) {
  const t = useTranslations("cronPreview");
  const expr = cron.trim();
  if (!expr) return null;

  let human = "";
  const nextRuns: Date[] = [];
  try {
    human = cronstrue.toString(expr, { throwExceptionOnParseError: true });
    const it = CronExpressionParser.parse(expr, { tz: "UTC" });
    for (let i = 0; i < 3; i++) nextRuns.push(it.next().toDate());
  } catch {
    return (
      <div className="rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive">
        {t("invalid")}
      </div>
    );
  }

  const fmtUtc = (d: Date) =>
    d.toLocaleString("en-GB", {
      timeZone: "UTC",
      day: "2-digit",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
  const fmtLocal = (d: Date) =>
    d.toLocaleString(undefined, {
      day: "2-digit",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });

  const next = nextRuns[0];

  return (
    <div className="rounded-md border bg-muted/30 px-3 py-2.5 text-xs space-y-2">
      <div className="font-medium text-foreground">{human}</div>
      <div className="text-muted-foreground">
        {t("nextRun")}:{" "}
        <span className="text-foreground">{formatRelativeTime(next.toISOString())}</span>
        {" — "}
        {fmtUtc(next)} {t("utc")}
      </div>
      <div className="text-muted-foreground">
        <div className="mb-0.5">{t("upcoming")}</div>
        <ul className="space-y-0.5">
          {nextRuns.map((d) => (
            <li key={d.toISOString()} className="font-mono">
              {fmtUtc(d)} {t("utc")}
              <span className="opacity-70">
                {" · "}
                {fmtLocal(d)} {t("local")}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

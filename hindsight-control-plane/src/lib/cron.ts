import { CronExpressionParser } from "cron-parser";

/**
 * Next scheduled run for a cron expression, evaluated in UTC (matching the API,
 * which runs croniter in UTC). Returns null for an empty or invalid expression.
 */
export function nextCronRun(cron: string | null | undefined): Date | null {
  const expr = cron?.trim();
  if (!expr) return null;
  try {
    return CronExpressionParser.parse(expr, { tz: "UTC" }).next().toDate();
  } catch {
    return null;
  }
}

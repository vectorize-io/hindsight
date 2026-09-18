/**
 * The "last hour / 24 hours / 7 days / 30 days" preset shared by every list
 * view that filters on time.
 *
 * Kept out of the components because the same eight lines were already copied
 * into `audit-logs-view` and `llm-requests-view`, and the documents list made a
 * third. The presets are the vocabulary those two views established; this only
 * gives them one home so a new preset appears everywhere at once.
 *
 * Returns ISO-8601 with an offset, which is what every endpoint taking
 * `start_date`/`end_date` expects. Only the lower bound is set: these presets
 * all mean "since then", never a closed range.
 */
export type DateRangePreset = "all" | "1h" | "1d" | "7d" | "30d";

export const DATE_RANGE_PRESETS: DateRangePreset[] = ["all", "1h", "1d", "7d", "30d"];

export interface DateRangeBounds {
  start_date?: string;
  end_date?: string;
}

export function resolveDateRangePreset(range: string, now: Date = new Date()): DateRangeBounds {
  if (range === "all") return {};

  const start = new Date(now);
  if (range === "1h") start.setHours(now.getHours() - 1);
  else if (range === "1d") start.setDate(now.getDate() - 1);
  else if (range === "7d") start.setDate(now.getDate() - 7);
  else if (range === "30d") start.setDate(now.getDate() - 30);
  // An unknown preset falls through with `start` still at `now`, which would be
  // an empty window — treat it as no filter instead, the way "all" does.
  else return {};

  return { start_date: start.toISOString() };
}

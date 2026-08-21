/**
 * Timeline date utilities for memory rows, replicating the backend's temporal
 * coalescing policy.
 *
 * The backend derives a unit's effective time via
 * `COALESCE(occurred_start, mentioned_at, occurred_end)`
 * (see `_coalesce_date` in hindsight-api-slim/hindsight_api/engine/search/
 * retrieval.py, and the same order for recency in engine/search/reranking.py).
 * `occurred_*` dates are content dates extracted by the LLM and are missing
 * on most rows, while `mentioned_at` is the system-set mention time and
 * `created_at` the ingest time. `effectiveDate` extends that chain with the
 * always-populated system fields (mirroring the per-row `COALESCE` fallback
 * to `created_at` in `get_memories_timeseries`) plus the legacy
 * `date`/`event_date` column, so rows without any content date still plot
 * instead of being dropped from the timeline.
 *
 * The partition and grouping helpers below are the pure date logic behind the
 * TimelineView, extracted so it can be unit-tested without rendering.
 */

// The graph endpoint serializes its legacy `date` column as
// "YYYY-MM-DD HH:MM" - no `T`, no zone. That form is outside ECMA-262's date
// grammar (browsers parse it inconsistently), so it is matched explicitly and
// interpreted as UTC, matching the UTC timestamps the backend emits.
const SPACE_FORMAT = /^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})$/;

function parseCandidate(raw: string): Date | null {
  const spaceMatch = SPACE_FORMAT.exec(raw.trim());
  if (spaceMatch) {
    const [, year, month, day, hour, minute] = spaceMatch;
    return new Date(
      Date.UTC(Number(year), Number(month) - 1, Number(day), Number(hour), Number(minute))
    );
  }
  const parsed = new Date(raw);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

export function effectiveDate(row: {
  occurred_start?: string | null;
  mentioned_at?: string | null;
  occurred_end?: string | null;
  created_at?: string | null;
  date?: string | null;
}): Date | null {
  for (const raw of [
    row.occurred_start,
    row.mentioned_at,
    row.occurred_end,
    row.created_at,
    row.date,
  ]) {
    if (!raw) continue;
    const parsed = parseCandidate(raw);
    if (parsed) return parsed;
  }
  return null;
}

export interface EffectiveDateEntry<T> {
  row: T;
  date: Date;
}

/**
 * Split rows into date-bearing entries (sorted ascending by effective date;
 * ties keep input order, as `Array.sort` is stable) and rows with no usable
 * date at all.
 */
export function partitionByEffectiveDate<T extends object>(
  rows: T[]
): {
  sortedItems: EffectiveDateEntry<T>[];
  itemsWithoutDates: T[];
} {
  const withDates: EffectiveDateEntry<T>[] = [];
  const withoutDates: T[] = [];
  for (const row of rows) {
    const date = effectiveDate(row);
    if (date) withDates.push({ row, date });
    else withoutDates.push(row);
  }
  withDates.sort((a, b) => a.date.getTime() - b.date.getTime());
  return { sortedItems: withDates, itemsWithoutDates: withoutDates };
}

export type TimelineGranularity = "year" | "month" | "week" | "day";

/**
 * Bucket key for a date at the given granularity. Week buckets start on
 * Sunday (`getDay() === 0`); the `Wxx` component is a week index within the
 * month (ceil of the start day / 7) and only needs to be unique per bucket,
 * mirroring the timeline's historical key format.
 */
export function getTimelineGroupKey(date: Date, granularity: TimelineGranularity): string {
  const year = date.getFullYear();
  const month = date.getMonth();
  const day = date.getDate();

  switch (granularity) {
    case "year":
      return `${year}`;
    case "month":
      return `${year}-${String(month + 1).padStart(2, "0")}`;
    case "week": {
      const startOfWeek = new Date(date);
      startOfWeek.setDate(day - date.getDay());
      return `${startOfWeek.getFullYear()}-W${String(Math.ceil(startOfWeek.getDate() / 7)).padStart(2, "0")}-${String(startOfWeek.getMonth() + 1).padStart(2, "0")}-${String(startOfWeek.getDate()).padStart(2, "0")}`;
    }
    case "day":
      return `${year}-${String(month + 1).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
  }
}

export function getTimelineGroupLabel(
  key: string,
  date: Date,
  granularity: TimelineGranularity
): string {
  switch (granularity) {
    case "year":
      return key;
    case "month":
      return date.toLocaleDateString("en-US", { year: "numeric", month: "short" });
    case "week": {
      const endOfWeek = new Date(date);
      endOfWeek.setDate(date.getDate() + 6);
      return `${date.toLocaleDateString("en-US", { month: "short", day: "numeric" })} - ${endOfWeek.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}`;
    }
    case "day":
      return date.toLocaleDateString("en-US", {
        weekday: "short",
        month: "short",
        day: "numeric",
        year: "numeric",
      });
  }
}

export interface TimelineGroup<T> {
  key: string;
  label: string;
  date: Date;
  items: EffectiveDateEntry<T>[];
}

/**
 * Group date-bearing entries by granularity bucket, ordered ascending by
 * bucket date. Input must already be sorted ascending by effective date.
 */
export function groupTimelineItems<T>(
  sortedItems: EffectiveDateEntry<T>[],
  granularity: TimelineGranularity
): TimelineGroup<T>[] {
  if (sortedItems.length === 0) return [];

  const groups = new Map<string, { items: EffectiveDateEntry<T>[]; date: Date }>();
  for (const entry of sortedItems) {
    const key = getTimelineGroupKey(entry.date, granularity);
    let group = groups.get(key);
    if (!group) {
      // Week keys embed the week-start date; recover it for labels/ordering.
      let groupDate = entry.date;
      if (granularity === "week") {
        const parts = key.split("-");
        groupDate = new Date(Number(parts[0]), Number(parts[2]) - 1, Number(parts[3]));
      }
      group = { items: [], date: groupDate };
      groups.set(key, group);
    }
    group.items.push(entry);
  }

  return [...groups.entries()]
    .sort(([, a], [, b]) => a.date.getTime() - b.date.getTime())
    .map(([key, { items, date }]) => ({
      key,
      label: getTimelineGroupLabel(key, date, granularity),
      items,
      date,
    }));
}

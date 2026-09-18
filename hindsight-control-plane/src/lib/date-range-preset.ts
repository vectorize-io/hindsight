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
export type DateRangePreset = "all" | "1h" | "1d" | "7d" | "30d" | "custom";

export interface DateRangeBounds {
  start_date?: string;
  end_date?: string;
}

export interface CustomRangeState {
  /** The bounds to send. Empty while the range is unusable, or when both ends are blank. */
  bounds: DateRangeBounds;
  /** Both ends are set and the end precedes the start — nothing is sent. */
  reversed: boolean;
}

/**
 * Resolve an explicit from/to pair, as typed into two `datetime-local` inputs.
 *
 * Either end alone is a valid filter here — the presets are one-sided too, and
 * the endpoints accept one bound — so unlike `resolveTemporalWindow` (which
 * backs recall's window and needs a closed range) this does not demand both.
 *
 * `datetime-local` has no offset, and the API reads an offset-less datetime as
 * UTC. A value typed into that input is a LOCAL wall-clock time, so converting
 * through `Date` is what makes "from 09:00" mean the user's 09:00 rather than
 * 09:00 UTC. `new Date("YYYY-MM-DDTHH:mm")` parses as local; `toISOString()`
 * then carries the correct instant with an offset.
 */
export function resolveCustomRange(from: string, to: string): CustomRangeState {
  const start = toIsoOrNull(from);
  const end = toIsoOrNull(to);

  // Compared as the converted instants, not as the typed strings. The two
  // disagree inside a daylight-saving gap: 02:30 on a spring-forward morning
  // does not exist and normalises *forward*, so "02:30 -> 03:00" reads as
  // ascending text but is a reversed hour. Comparing what is actually sent also
  // keeps this rule identical to the server's (`end_date <= start_date` is a
  // 400), so the inline message replaces an error toast rather than racing it.
  if (start && end && end <= start) return { bounds: {}, reversed: true };

  const bounds: DateRangeBounds = {};
  if (start) bounds.start_date = start;
  if (end) bounds.end_date = end;
  return { bounds, reversed: false };
}

/**
 * A `datetime-local` value that is not yet a complete instant, as ISO — or null.
 *
 * Mid-typing the input hands back half a value (a year with no month, say), and
 * `new Date()` of that is an Invalid Date whose `toISOString()` THROWS
 * `RangeError` rather than returning anything. Unguarded, that took the whole
 * documents page down on the first keystroke. An incomplete value is simply not
 * a bound yet, so it contributes nothing and the list stays unfiltered until the
 * user finishes typing.
 */
function toIsoOrNull(value: string): string | null {
  if (!value) return null;
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? null : parsed.toISOString();
}

export function resolveDateRangePreset(range: string, now: Date = new Date()): DateRangeBounds {
  if (range === "all") return {};
  // "custom" carries its bounds in its own inputs, not in the preset.
  if (range === "custom") return {};

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

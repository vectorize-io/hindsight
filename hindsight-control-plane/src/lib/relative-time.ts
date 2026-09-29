export function formatRelativeTime(dateStr: string): string {
  const then = new Date(dateStr).getTime();
  const diffSec = Math.round((Date.now() - then) / 1000);
  const abs = Math.abs(diffSec);
  if (abs < 60) return diffSec >= 0 ? "just now" : "in a moment";
  // value starts in seconds; each step divides by `factor` to reach `nextUnit`.
  // (Previously the first pair was mislabeled [60, "second"], which left every
  // unit one step too low — e.g. 8 minutes rendered as "in 8 seconds".)
  const units: [number, Intl.RelativeTimeFormatUnit][] = [
    [60, "minute"],
    [60, "hour"],
    [24, "day"],
    [7, "week"],
    [4.345, "month"],
    [12, "year"],
  ];
  let value = diffSec;
  let unit: Intl.RelativeTimeFormatUnit = "second";
  for (const [factor, nextUnit] of units) {
    if (Math.abs(value) < factor) break;
    value = value / factor;
    unit = nextUnit;
  }
  return new Intl.RelativeTimeFormat("en", { numeric: "auto" }).format(-Math.round(value), unit);
}

export function formatAbsoluteDateTime(dateStr: string): string {
  const date = new Date(dateStr);
  return `${date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  })} at ${date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  })}`;
}

/**
 * A timestamp that is a *position* rather than an age — a read watermark, say.
 *
 * Relative rendering ("4 hours ago") is wrong for these: shown next to a relative
 * age it collapses to the same string even when the two moments are hours apart,
 * which reads as the same fact printed twice. So this is always absolute, and only
 * as long as it needs to be:
 *
 * - today        -> `14:38`
 * - this year    -> `19 Aug 14:38`
 * - earlier year -> `19 Aug 2025`
 *
 * The full date and time is expected to be on the element's `title`, so the label
 * can stay short without losing anything.
 */
export function formatWatermark(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const time = date.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
  if (date.toDateString() === now.toDateString()) return time;
  const dayMonth = date.toLocaleDateString(undefined, { day: "numeric", month: "short" });
  if (date.getFullYear() === now.getFullYear()) return `${dayMonth} ${time}`;
  return date.toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

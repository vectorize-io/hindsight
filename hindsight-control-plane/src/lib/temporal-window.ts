/**
 * Resolve the recall temporal window from the two form inputs.
 *
 * Kept out of the component so the rules are testable: a window needs *both*
 * ends (one alone is an incomplete range, not a filter), and a range whose end
 * precedes its start is rejected here rather than sent for the API to 422.
 *
 * `datetime-local` values are already `YYYY-MM-DDTHH:mm`, which sorts
 * lexicographically in chronological order, so comparing the raw strings is
 * exact — no Date parsing, and no local-timezone reinterpretation on the way.
 * The API reads a datetime with no offset as UTC.
 */
export interface TemporalWindowState {
  /** Both ends are set, so a window could be sent. */
  complete: boolean;
  /** Both ends are set but the end precedes the start. */
  reversed: boolean;
  /** The window to send, or undefined when there is nothing valid to send. */
  value?: { start: string; end: string };
}

export function resolveTemporalWindow(start: string, end: string): TemporalWindowState {
  const complete = Boolean(start && end);
  const reversed = complete && end < start;

  if (!complete || reversed) {
    return { complete, reversed };
  }
  return { complete, reversed, value: { start, end } };
}

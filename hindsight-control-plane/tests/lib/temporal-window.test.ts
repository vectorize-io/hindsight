import { describe, expect, it } from "vitest";
import { resolveTemporalWindow } from "@/lib/temporal-window";

describe("resolveTemporalWindow", () => {
  it("sends the window when both ends are set", () => {
    const state = resolveTemporalWindow("2023-04-01T00:00", "2023-06-30T23:59");

    expect(state).toEqual({
      complete: true,
      reversed: false,
      value: { start: "2023-04-01T00:00", end: "2023-06-30T23:59" },
    });
  });

  it("sends nothing until both ends are set", () => {
    // One end alone is an incomplete range, not a half-open filter.
    for (const [start, end] of [
      ["2023-04-01T00:00", ""],
      ["", "2023-06-30T23:59"],
      ["", ""],
    ]) {
      const state = resolveTemporalWindow(start, end);

      expect(state.complete).toBe(false);
      expect(state.reversed).toBe(false);
      expect(state.value).toBeUndefined();
    }
  });

  it("flags a reversed range and withholds it", () => {
    const state = resolveTemporalWindow("2023-06-30T23:59", "2023-04-01T00:00");

    expect(state.reversed).toBe(true);
    expect(state.value).toBeUndefined();
  });

  it("accepts an instant, where both ends are equal", () => {
    const state = resolveTemporalWindow("2023-04-01T00:00", "2023-04-01T00:00");

    expect(state.reversed).toBe(false);
    expect(state.value).toEqual({ start: "2023-04-01T00:00", end: "2023-04-01T00:00" });
  });

  it("orders by calendar date, not by string length or day-of-month", () => {
    // A naive comparison of the rendered dd/mm/yyyy form would call this
    // reversed; the ISO value the input actually holds does not.
    const state = resolveTemporalWindow("2023-04-30T00:00", "2023-06-01T00:00");

    expect(state.reversed).toBe(false);
  });
});

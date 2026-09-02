import { describe, expect, it } from "vitest";
import { groupFailures, type HistoryEntry } from "@/components/mental-model-detail-modal";

/**
 * A failed mental-model refresh is retried by the worker, and every attempt
 * records its own history row. Grouping is what stops one outage rendering as N
 * identical events — and what keeps two outages from rendering as one.
 */

const failure = (
  changed_at: string,
  error_message = "recall failed",
  failure_reason: HistoryEntry["failure_reason"] = "retrieval_failed"
): HistoryEntry => ({
  previous_content: null,
  changed_at,
  kind: "refresh_failed",
  failure_reason,
  error_message,
});

const version = (changed_at: string): HistoryEntry => ({
  previous_content: "# Team\n\nold\n",
  changed_at,
});

describe("groupFailures", () => {
  it("returns nothing for a history with no failures", () => {
    expect(groupFailures([version("3"), version("2")])).toEqual([]);
  });

  it("collapses consecutive identical failures into one event", () => {
    const groups = groupFailures([failure("4"), failure("3"), failure("2")]);
    expect(groups).toHaveLength(1);
    expect(groups[0].attempts).toBe(3);
    // Newest first, so the group is stamped with the newest attempt and remembers
    // the oldest — the two ends of the episode.
    expect(groups[0].entry.changed_at).toBe("4");
    expect(groups[0].oldest).toBe("2");
  });

  it("splits identical failures that a successful refresh separates", () => {
    // It broke, recovered, and broke again: two episodes. Merging them would
    // under-report how often the model has been failing.
    const groups = groupFailures([failure("5"), failure("4"), version("3"), failure("2")]);
    expect(groups.map((g) => g.attempts)).toEqual([2, 1]);
    expect(groups[1].entry.changed_at).toBe("2");
  });

  it("splits on a different reason", () => {
    const groups = groupFailures([failure("3"), failure("2", "no answer", "no_answer")]);
    expect(groups.map((g) => g.entry.failure_reason)).toEqual(["retrieval_failed", "no_answer"]);
  });

  it("splits on a different message under the same reason", () => {
    // Same class of failure, different cause — worth seeing separately.
    const groups = groupFailures([
      failure("3", "the embedder is down"),
      failure("2", "the store is down"),
    ]);
    expect(groups).toHaveLength(2);
  });

  it("ignores versions between failures of different kinds", () => {
    const groups = groupFailures([
      failure("5"),
      version("4"),
      failure("3", "no answer", "no_answer"),
      failure("2", "no answer", "no_answer"),
    ]);
    expect(groups.map((g) => g.attempts)).toEqual([1, 2]);
  });
});

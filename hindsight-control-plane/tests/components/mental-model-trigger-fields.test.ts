import { describe, expect, it } from "vitest";
import { triggerFormFromTrigger, triggerFromForm } from "@/components/mental-model-trigger-fields";

/**
 * The server MERGES a trigger over the stored one (and over the knowledge-page
 * default), so the form must state what it turned off rather than omit it — an
 * omitted field silently keeps its old value.
 */

describe("triggerFromForm", () => {
  it("clears the cron, fact types and exclude flag explicitly", () => {
    const form = triggerFormFromTrigger({
      refresh_cron: "0 * * * *",
      fact_types: ["observation"],
      exclude_mental_models: true,
    });
    const trigger = triggerFromForm({
      ...form,
      refreshTrigger: "manual",
      factTypes: [],
      excludeMentalModels: false,
    });

    expect(trigger).toMatchObject({
      refresh_after_consolidation: false,
      refresh_cron: null,
      fact_types: null,
      exclude_mental_models: false,
    });
  });

  it("round-trips a scheduled trigger", () => {
    const original = {
      mode: "delta" as const,
      refresh_after_consolidation: false,
      refresh_cron: "0 3 * * *",
      min_refresh_interval_seconds: 600,
      fact_types: ["observation" as const],
      exclude_mental_models: true,
      tags_match: "any" as const,
      recall_max_tokens: 2048,
      keep_trace: true,
    };

    expect(triggerFromForm(triggerFormFromTrigger(original))).toMatchObject(original);
  });

  it("sends refresh_after_consolidation for the auto option and no cron", () => {
    const trigger = triggerFromForm(triggerFormFromTrigger({ refresh_after_consolidation: true }));

    expect(trigger?.refresh_after_consolidation).toBe(true);
    expect(trigger?.refresh_cron).toBeNull();
  });

  it("returns null when the tag groups are not valid JSON", () => {
    expect(triggerFromForm({ ...triggerFormFromTrigger(), tagGroups: "[{" })).toBeNull();
  });
});

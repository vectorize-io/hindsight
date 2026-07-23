import { describe, expect, it } from "vitest";

import {
  mergeObservationsOverrides,
  mergeResolvedObservations,
  observationsSlice,
  reconcileObservationsEdits,
} from "../../src/lib/observations-config";

describe("observations config state", () => {
  it.each([true, false])(
    "keeps a resolved %s value inherited when the bank has no override",
    (resolvedValue) => {
      const edits = observationsSlice(
        {
          enable_observations: resolvedValue,
          consolidation_llm_batch_size: 12,
          observations_mission: "Resolve from the parent configuration",
        },
        {}
      );

      expect(edits).toMatchObject({
        enable_observations: null,
        consolidation_llm_batch_size: 12,
        observations_mission: "Resolve from the parent configuration",
      });
    }
  );

  it.each([true, false])("preserves an explicit %s bank override", (overrideValue) => {
    const edits = observationsSlice(
      { enable_observations: overrideValue },
      { enable_observations: overrideValue }
    );

    expect(edits.enable_observations).toBe(overrideValue);
  });

  it("uses the PATCH response layers when resetting the saved state", () => {
    const edits = observationsSlice(
      {
        enable_observations: false,
        consolidation_llm_batch_size: 24,
        consolidation_source_facts_max_tokens: 10_000,
        consolidation_source_facts_max_tokens_per_observation: 2_000,
        observations_mission: "Returned resolved state",
        max_observations_per_scope: 8,
      },
      { enable_observations: true, consolidation_llm_batch_size: 6 }
    );

    expect(edits).toEqual({
      enable_observations: true,
      consolidation_llm_batch_size: 24,
      consolidation_source_facts_max_tokens: 10_000,
      consolidation_source_facts_max_tokens_per_observation: 2_000,
      observations_mission: "Returned resolved state",
      max_observations_per_scope: 8,
    });
  });

  it("updates only observation baselines from a PATCH response", () => {
    const submitted = observationsSlice(
      { enable_observations: false, consolidation_llm_batch_size: 24 },
      { enable_observations: false }
    );
    const nextConfig = mergeResolvedObservations(
      {
        retain_chunk_size: 3_000,
        audit_log_enabled: true,
        enable_observations: true,
        consolidation_llm_batch_size: 12,
      },
      submitted,
      {
        // Unrelated response values must not move another section's baseline.
        enable_observations: false,
        consolidation_llm_batch_size: 24,
        retain_chunk_size: 9_000,
        audit_log_enabled: false,
      }
    );

    expect(nextConfig).toMatchObject({
      retain_chunk_size: 3_000,
      audit_log_enabled: true,
      enable_observations: false,
      consolidation_llm_batch_size: 24,
    });
  });

  it("uses accepted values for permission-filtered response fields", () => {
    const submitted = observationsSlice(
      {
        enable_observations: true,
        consolidation_llm_batch_size: 24,
        observations_mission: "Accepted mission",
      },
      {}
    );
    const nextConfig = mergeResolvedObservations(
      {
        enable_observations: true,
        consolidation_llm_batch_size: 12,
        observations_mission: "Old mission",
      },
      submitted,
      { enable_observations: true }
    );

    expect(nextConfig).toMatchObject({
      enable_observations: true,
      consolidation_llm_batch_size: 24,
      observations_mission: "Accepted mission",
    });
  });

  it("drops a stale resolved boolean when the cleared parent value is filtered", () => {
    const submitted = observationsSlice({ enable_observations: false }, {});
    const nextConfig = mergeResolvedObservations({ enable_observations: false }, submitted, {});

    expect(nextConfig).not.toHaveProperty("enable_observations");
  });

  it("preserves unrelated overrides while applying the boolean tombstone", () => {
    const currentOverrides = {
      audit_log_enabled: true,
      enable_observations: false,
    };

    expect(mergeObservationsOverrides(currentOverrides, {})).toEqual({
      audit_log_enabled: true,
    });
    expect(mergeObservationsOverrides(currentOverrides, { enable_observations: true })).toEqual({
      audit_log_enabled: true,
      enable_observations: true,
    });
  });

  it("does not discard edits made while a PATCH is in flight", () => {
    const submitted = observationsSlice(
      {
        enable_observations: true,
        consolidation_llm_batch_size: 12,
        observations_mission: "Submitted mission",
      },
      {}
    );
    const editedDuringSave = { ...submitted, observations_mission: "Newer mission" };
    const responseConfig = {
      enable_observations: true,
      consolidation_llm_batch_size: 10,
      observations_mission: "Submitted mission",
    };

    expect(reconcileObservationsEdits(submitted, submitted, responseConfig, {})).toEqual({
      ...submitted,
      consolidation_llm_batch_size: 10,
    });
    expect(reconcileObservationsEdits(editedDuringSave, submitted, responseConfig, {})).toEqual({
      ...editedDuringSave,
      consolidation_llm_batch_size: 10,
    });
  });
});

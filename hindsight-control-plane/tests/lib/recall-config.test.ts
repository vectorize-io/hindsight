import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { RECALL_FIELDS, recallSlice } from "../../src/lib/recall-config";

describe("recall config fields", () => {
  it("includes the decision-model reranker's prune and dedup", () => {
    const fields = RECALL_FIELDS.map(([field]) => field);
    expect(fields).toContain("recall_decision_model_prune");
    expect(fields).toContain("recall_decision_model_dedup");
  });

  it("only offers fields the API accepts as per-bank config", () => {
    // The API's bank-config schema lists every configurable field; a toggle for
    // anything else would fail to save.
    const spec = JSON.parse(
      readFileSync(resolve(__dirname, "../../../hindsight-docs/static/openapi.json"), "utf8")
    );
    const configurable = Object.keys(spec.components.schemas.BankTemplateConfig.properties);
    for (const [field] of RECALL_FIELDS) expect(configurable).toContain(field);
  });

  it("has a label and description for every field in the default locale", () => {
    const en = JSON.parse(
      readFileSync(resolve(__dirname, "../../src/messages/en.json"), "utf8")
    ).bankConfig;
    for (const [, key] of RECALL_FIELDS) {
      expect(en[`${key}Label`]).toBeTruthy();
      expect(en[`${key}Description`]).toBeTruthy();
    }
  });
});

describe("recallSlice", () => {
  it("inherits (null) every field the bank does not override", () => {
    const edits = recallSlice({});
    for (const [field] of RECALL_FIELDS) expect(edits[field]).toBeNull();
  });

  it("keeps the bank's own true and false overrides", () => {
    const edits = recallSlice({
      recall_decision_model_prune: true,
      recall_decision_model_dedup: false,
      enable_reranking: false,
    });
    expect(edits.recall_decision_model_prune).toBe(true);
    expect(edits.recall_decision_model_dedup).toBe(false);
    expect(edits.enable_reranking).toBe(false);
    expect(edits.recall_prefer_observations).toBeNull();
  });

  it("ignores fields outside the recall section", () => {
    expect(Object.keys(recallSlice({ reflect_mission: "x" })).sort()).toEqual(
      RECALL_FIELDS.map(([field]) => field).sort()
    );
  });
});

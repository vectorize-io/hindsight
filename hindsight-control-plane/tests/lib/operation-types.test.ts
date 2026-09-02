import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";

/**
 * Static guard for dataplane -> control-plane operation-type parity.
 *
 * An async operation type is declared once in the engine (`operation_type="..."`)
 * and then has to be repeated in this UI: once in `OPERATION_TYPE_VALUES` so the
 * operations list can filter on it, and once in the `operationTypeLabels` map so
 * it renders as a name rather than a raw snake_case string. Nothing links the
 * two sides, so the failure mode is silent — the operations view keeps working,
 * it just shows `vector_index_maintenance` and offers no filter for it.
 *
 * The Python side already has the mirror of this guard
 * (`test_all_operation_types_have_slot_reservation_config` in api-slim, which
 * asserts every operation type has worker slot-reservation config). This is the
 * half that was missing: it caught the Python omission when
 * `vector_index_maintenance` was added and said nothing about the UI.
 *
 * Parsed rather than imported because the source of truth is a Python file and
 * a React component; a runtime import would need the whole app.
 */

const REPO_ROOT = join(__dirname, "..", "..", "..");
const ENGINE = join(REPO_ROOT, "hindsight-api-slim", "hindsight_api", "engine", "memory_engine.py");
const VIEW = join(REPO_ROOT, "hindsight-control-plane", "src", "components", "bank-operations-view.tsx");

/**
 * Operation types the UI carries that the engine never emits from
 * `_submit_async_operation`. These are real rows users can see, produced by
 * paths that build the operation record directly, so the UI must keep them —
 * the parity check only runs in the dataplane -> UI direction.
 */
const UI_ONLY = new Set(["all", "retain", "webhook_delivery"]);

function engineOperationTypes(): Set<string> {
  const src = readFileSync(ENGINE, "utf-8");
  return new Set([...src.matchAll(/operation_type=["']([a-z_]+)["']/g)].map((m) => m[1]));
}

function uiFilterValues(): Set<string> {
  const src = readFileSync(VIEW, "utf-8");
  const block = src.match(/const OPERATION_TYPE_VALUES = \[([\s\S]*?)\] as const/);
  expect(block, "OPERATION_TYPE_VALUES not found — did the constant get renamed?").toBeTruthy();
  return new Set([...block![1].matchAll(/"([a-z_]+)"/g)].map((m) => m[1]));
}

function uiLabelledTypes(): Set<string> {
  const src = readFileSync(VIEW, "utf-8");
  const block = src.match(/const operationTypeLabels: Record<string, string> = \{([\s\S]*?)\n {2}\}/);
  expect(block, "operationTypeLabels not found — did the map get renamed?").toBeTruthy();
  return new Set([...block![1].matchAll(/^\s+([a-z_]+):/gm)].map((m) => m[1]));
}

describe("operation-type parity between the engine and this UI", () => {
  it("every operation type the engine submits is filterable here", () => {
    const missing = [...engineOperationTypes()].filter((t) => !uiFilterValues().has(t)).sort();

    expect(
      missing,
      `Operation types ${JSON.stringify(missing)} are submitted by memory_engine.py but missing from ` +
        `OPERATION_TYPE_VALUES in bank-operations-view.tsx. Users cannot filter the operations list by them. ` +
        `Add them there (and to operationTypeLabels + every src/messages/*.json).`,
    ).toEqual([]);
  });

  it("every operation type the engine submits has a display label", () => {
    const missing = [...engineOperationTypes()].filter((t) => !uiLabelledTypes().has(t)).sort();

    expect(
      missing,
      `Operation types ${JSON.stringify(missing)} have no entry in operationTypeLabels, so they render as ` +
        `raw snake_case in the operations list. Add a t("operationType.<camelCase>") entry.`,
    ).toEqual([]);
  });

  it("the filter list and the label map stay in step", () => {
    // Two hand-maintained lists of the same thing: a type added to one and not
    // the other is filterable-but-unnamed, or named-but-unfilterable.
    expect([...uiFilterValues()].sort()).toEqual([...uiLabelledTypes()].sort());
  });

  it("declares which UI-only types are deliberate", () => {
    // Keeps the exemption list honest: an entry that the engine started
    // emitting, or that the UI dropped, should not sit here unnoticed.
    const unexplained = [...uiFilterValues()].filter(
      (t) => !engineOperationTypes().has(t) && !UI_ONLY.has(t),
    );

    expect(unexplained, `UI carries operation types the engine never emits: ${JSON.stringify(unexplained)}`).toEqual(
      [],
    );
  });
});

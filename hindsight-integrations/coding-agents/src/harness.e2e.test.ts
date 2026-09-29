import { describe, expect, it } from "vitest";
import { ALL_HARNESS_SETUPS } from "./e2e/harnesses";
import {
  e2eEnabled,
  e2ePromptMarker,
  getRetainedDocument,
  harnessCredentialStatus,
  runHarnessE2e,
  seededDecisionStatuses,
} from "./e2e/harness";

/**
 * One real run per harness: install the CLI and the published Hindsight package in a container,
 * point it at a bank seeded with a decision the prompt deliberately does not contain, and require
 * that the agent's answer carries that decision back out. Passing means the harness's actual hook
 * lifecycle both INJECTED memory and RETAINED the session — neither can be faked by a stub.
 *
 * Harnesses authenticate with subscription logins, so each is skipped when its credentials are not
 * on this machine; a skip is reported with its reason so it can't be mistaken for a pass.
 */
describe.runIf(e2eEnabled)("Docker harness E2E", () => {
  for (const harness of ALL_HARNESS_SETUPS) {
    const { available, reason } = harnessCredentialStatus(harness);
    const injects = harness.injectsIntoModel !== false;
    it.runIf(available)(
      `${harness.name}: installs the CLI and Hindsight, ${injects ? "injects a seeded decision, and retains" : "and retains"} the real session`,
      async () => {
        const run = await runHarnessE2e(harness);
        // The seeded commit says only 429 and 408 are retryable; the prompt never mentions either.
        // On failure the hook diagnostics are the only way to tell "memory never reached the model"
        // apart from "the CLI's answer never reached us", so surface them with the assertion.
        const context = `--- ${harness.name} output ---\n${run.output}\n--- diagnostics ---\n${run.diagnostics || "(none written)"}`;
        // Skipped for hosts that cannot inject at all (see injectsIntoModel) — retention below is
        // still asserted for every harness, so a passive host is covered, not waved through.
        for (const status of injects ? seededDecisionStatuses : []) {
          expect(run.output, context).toContain(status);
        }
        expect(JSON.stringify(await getRetainedDocument(run))).toContain(e2ePromptMarker);
      },
      720_000
    );
    it.skipIf(available)(`${harness.name}: skipped — ${reason}`, () => {});
  }
});

describe.runIf(!e2eEnabled)("Docker harness E2E", () => {
  it.skip("set HINDSIGHT_HARNESS_E2E=1 to run locally with host credentials", () => {});
});

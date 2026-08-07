/**
 * `retain apply` — ingest the bound documents into a NEW versioned bank with the current missions.
 * `observe apply` — clear observations on the current bank and re-consolidate with the current
 *                   observation mission (in place; no new version).
 */

import { collectDocuments } from "./docs.js";
import { resolveApiKey, SandboxApi } from "./hindsight.js";
import { provisionAndIngest } from "./pipeline.js";
import { Project } from "./store.js";
import type { ProgressFn } from "./types.js";

export interface ApplyParams {
  projectDir: string;
  apiKey?: string;
}

export interface RetainApplyResult {
  version: number;
  bank: string;
  observationCount: number;
}

/** Ingest documents into a fresh `<name>-vN` bank with the current retain + observation missions. */
export async function runRetainApply(
  params: ApplyParams,
  onProgress: ProgressFn
): Promise<RetainApplyResult> {
  const proj = await Project.load(params.projectDir);
  const documents = await collectDocuments(proj.documents);
  if (documents.length === 0) throw new Error(`No .txt/.md documents found at ${proj.documents}`);

  // Attach the retain feedback that accumulated since the previous version.
  const consumed = proj.versions.reduce((sum, v) => sum + v.feedback.length, 0);
  const version = proj.addVersion({
    retainMission: proj.retain.mission,
    observeMission: proj.observe.mission,
    feedback: proj.retain.feedback.slice(consumed),
  });
  await proj.save();

  const api = new SandboxApi(proj.apiUrl, resolveApiKey(params.apiKey ?? proj.apiKey));
  onProgress(`Applying retain → version ${version.n} (bank ${version.bank})`);
  const { observationCount } = await provisionAndIngest(
    api,
    version.bank,
    { retainMission: proj.retain.mission, observationsMission: proj.observe.mission },
    documents,
    onProgress
  );

  proj.addStep(
    "retain apply",
    `v${version.n} → ${version.bank} (${observationCount} observations)`,
    version.retainMission
  );
  await proj.save();
  onProgress(
    `\nVersion ${version.n} ready — bank: ${version.bank} (${observationCount} observations).`
  );
  onProgress("Point your validator at this bank, then feed failures back via `retain mission`.");
  return { version: version.n, bank: version.bank, observationCount };
}

export interface ObserveApplyResult {
  bank: string;
  observationCount: number;
}

/** Re-consolidate observations on the current bank with the current observation mission. */
export async function runObserveApply(
  params: ApplyParams,
  onProgress: ProgressFn
): Promise<ObserveApplyResult> {
  const proj = await Project.load(params.projectDir);
  const bank = proj.currentBank();
  if (!bank) throw new Error("No current version — run `retain apply` first.");

  const api = new SandboxApi(proj.apiUrl, resolveApiKey(params.apiKey ?? proj.apiKey));
  onProgress(`Applying observation mission to ${bank}…`);
  await api.updateObservationsMission(bank, proj.observe.mission);

  const cleared = await api.clearObservations(bank);
  onProgress(`Cleared ${cleared} observation(s); re-consolidating…`);
  await api.triggerConsolidation(bank);
  await api.waitForConsolidation(bank, { onProgress });

  // Record the observation mission against the current version.
  const current = proj.versions.find((v) => v.n === proj.currentVersion);
  if (current) current.observeMission = proj.observe.mission;
  await proj.save();

  const observationCount = (await api.getStats(bank)).totalObservations;
  proj.addStep(
    "observe apply",
    `re-consolidated ${bank} (${observationCount} observations)`,
    proj.observe.mission
  );
  await proj.save();
  onProgress(`\nDone — bank ${bank} now has ${observationCount} observations.`);
  return { bank, observationCount };
}

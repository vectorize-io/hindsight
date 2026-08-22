/** `retain mission` / `observe mission` — refine a mission from feedback (+ optional examples). */

import { MissionLlm } from "./llm.js";
import { Project } from "./store.js";
import type { MissionKind, ProgressFn } from "./types.js";

export interface MissionParams {
  projectDir: string;
  kind: MissionKind;
  feedback: string;
  examples?: string[];
  model?: string;
  apiKey?: string;
}

export interface MissionResult {
  kind: MissionKind;
  mission: string;
}

export async function runMission(
  params: MissionParams,
  onProgress: ProgressFn
): Promise<MissionResult> {
  const proj = await Project.load(params.projectDir);
  const state = params.kind === "retain" ? proj.retain : proj.observe;

  const llm = new MissionLlm({ apiKey: params.apiKey, model: params.model });
  onProgress(`Refining ${params.kind} mission from feedback (${llm.model})…`);
  const mission = await llm.refineMission(
    params.kind,
    state.mission,
    params.feedback,
    params.examples ?? []
  );

  state.mission = mission;
  state.feedback.push(params.feedback);
  proj.addStep(
    `${params.kind} mission`,
    `refined from feedback`,
    `feedback: ${params.feedback}\n\nnew mission: ${mission}`
  );
  await proj.save();

  onProgress(`\nNew ${params.kind} mission:\n${mission}`);
  onProgress(
    params.kind === "retain"
      ? "\nNext: `retain apply` to ingest into a new version bank."
      : "\nNext: `observe apply` to re-consolidate the current bank."
  );
  return { kind: params.kind, mission };
}

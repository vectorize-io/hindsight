/** `status` — read-only project state: bound docs, missions, versions, golden + curations + check. */

import { Project } from "./store.js";
import type { CheckResult, CurationEvent, ProjectVersion, StepEntry } from "./types.js";

export interface ProjectStatus {
  name: string;
  dir: string;
  documents: string;
  apiUrl: string;
  retainMission: string | null;
  observeMission: string | null;
  retainFeedback: string[];
  observeFeedback: string[];
  versions: ProjectVersion[];
  currentVersion: number | null;
  currentBank: string | null;
  goldenCount: number;
  goldenAt: string | null;
  curations: CurationEvent[];
  lastCheck: CheckResult | null;
  steps: StepEntry[];
}

export async function readStatus(projectDir: string): Promise<ProjectStatus> {
  const proj = await Project.load(projectDir);
  return {
    name: proj.name,
    dir: proj.dir,
    documents: proj.documents,
    apiUrl: proj.apiUrl,
    retainMission: proj.retain.mission,
    observeMission: proj.observe.mission,
    retainFeedback: proj.retain.feedback,
    observeFeedback: proj.observe.feedback,
    versions: proj.versions,
    currentVersion: proj.currentVersion,
    currentBank: proj.currentBank(),
    goldenCount: proj.golden.length,
    goldenAt: proj.goldenAt,
    curations: proj.curations,
    lastCheck: proj.lastCheck,
    steps: proj.steps,
  };
}

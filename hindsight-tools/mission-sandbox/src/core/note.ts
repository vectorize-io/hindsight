/** `note` — set free-text notes on a version (e.g. validator results). */

import { Project } from "./store.js";
import type { ProgressFn } from "./types.js";

export interface NoteParams {
  projectDir: string;
  notes: string;
  /** Version to annotate; defaults to the current version. */
  version?: number;
}

export interface NoteResult {
  version: number;
  bank: string;
}

export async function runNote(params: NoteParams, onProgress: ProgressFn): Promise<NoteResult> {
  const proj = await Project.load(params.projectDir);
  const version = proj.setVersionNotes(params.notes, params.version);
  proj.addStep("note", `v${version.n}`, params.notes);
  await proj.save();
  onProgress(`Noted v${version.n} (${version.bank}).`);
  return { version: version.n, bank: version.bank };
}

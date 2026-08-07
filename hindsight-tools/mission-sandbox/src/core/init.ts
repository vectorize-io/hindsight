/** `init` — bind a project to its documents path + API config. No ingestion. */

import { promises as fs } from "node:fs";
import path from "node:path";

import { Project } from "./store.js";
import type { ProgressFn } from "./types.js";

export interface InitParams {
  projectDir: string;
  /** Path to the documents directory (or file) to ingest on each `retain apply`. */
  documents: string;
  apiUrl: string;
  apiKey?: string;
  /** Bank-id prefix; defaults to the project directory name. */
  name?: string;
}

export interface InitResult {
  name: string;
  documents: string;
}

export async function runInit(params: InitParams, onProgress: ProgressFn): Promise<InitResult> {
  const documents = path.resolve(params.documents);
  await fs.stat(documents).catch(() => {
    throw new Error(`Documents path not found: ${params.documents}`);
  });
  const name = params.name ?? path.basename(path.resolve(params.projectDir));

  const proj = await Project.create(params.projectDir, {
    name,
    documents,
    apiUrl: params.apiUrl,
    apiKey: params.apiKey,
  });
  proj.addStep("init", `bound to ${documents}`, `api: ${proj.apiUrl}`);
  await proj.save();
  onProgress(`Initialized project '${name}' bound to ${documents} (api: ${proj.apiUrl}).`);
  onProgress("Next: set a mission with `retain mission --feedback`, then `retain apply`.");
  return { name, documents };
}

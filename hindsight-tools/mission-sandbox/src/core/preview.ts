/** `preview` — dry-run extraction: show what a retain mission extracts from text, without ingesting. */

import { resolveApiKey, SandboxApi } from "./hindsight.js";
import { Project } from "./store.js";

export interface PreviewFact {
  text: string;
  factType: string;
  occurredStart: string | null;
  occurredEnd: string | null;
  entities: string[];
}

export interface PreviewParams {
  projectDir: string;
  content: string;
  /** Mission to test; defaults to the project's current working retain mission. */
  retainMission?: string | null;
  apiKey?: string;
}

export async function runExtractPreview(params: PreviewParams): Promise<PreviewFact[]> {
  const proj = await Project.load(params.projectDir);
  const bank = proj.currentBank();
  if (!bank)
    throw new Error("No current bank — run `retain apply` first (extraction config lives on it).");

  const api = new SandboxApi(proj.apiUrl, resolveApiKey(params.apiKey ?? proj.apiKey));
  const mission = params.retainMission !== undefined ? params.retainMission : proj.retain.mission;
  const items = await api.dryRunExtractItems(bank, params.content, { retainMission: mission });
  return items.map((m) => ({
    text: String(m.text ?? ""),
    factType: String(m.fact_type ?? ""),
    occurredStart: m.occurred_start != null ? String(m.occurred_start) : null,
    occurredEnd: m.occurred_end != null ? String(m.occurred_end) : null,
    entities: Array.isArray(m.entities) ? (m.entities as string[]) : [],
  }));
}

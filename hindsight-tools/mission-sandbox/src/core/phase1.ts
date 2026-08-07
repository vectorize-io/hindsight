/**
 * Phase 1 — curate the current bank to a GOLDEN snapshot (in place, no re-ingest).
 *   inspect : find memories (by doc / text) to trace an eval failure.
 *   curate  : edit / invalidate / revert a memory (PATCH).
 *   snapshot: freeze the current memory set as the golden target.
 */

import { resolveApiKey, SandboxApi, type FactRow } from "./hindsight.js";
import { Project } from "./store.js";
import type { CurationKind, ProgressFn } from "./types.js";

function bankOrThrow(proj: Project): string {
  const bank = proj.currentBank();
  if (!bank) throw new Error("No current bank — run `retain apply` first.");
  return bank;
}

export interface InspectParams {
  projectDir: string;
  doc?: string;
  grep?: string;
  apiKey?: string;
}

export async function runInspect(params: InspectParams): Promise<FactRow[]> {
  const proj = await Project.load(params.projectDir);
  const api = new SandboxApi(proj.apiUrl, resolveApiKey(params.apiKey ?? proj.apiKey));
  const rows = await api.listFacts(bankOrThrow(proj), { docId: params.doc, q: params.grep });
  return rows;
}

export interface TraceParams {
  projectDir: string;
  query: string;
  /** Evidence document ids the answer should come from (e.g. mapped from a benchmark's evidence). */
  docs?: string[];
  apiKey?: string;
}

export interface TraceResult {
  retrieved: FactRow[];
  evidence: { docId: string; facts: FactRow[] }[];
}

/**
 * Evidence-based tracing: recall what the bank retrieves for a failing question, and list the
 * memories in the evidence document(s) — so you can see whether the answer-fact is present but
 * not retrieved (→ curate for retrieval) or missing entirely (→ fix the mission).
 */
export async function runTrace(params: TraceParams): Promise<TraceResult> {
  const proj = await Project.load(params.projectDir);
  const bank = bankOrThrow(proj);
  const api = new SandboxApi(proj.apiUrl, resolveApiKey(params.apiKey ?? proj.apiKey));
  const retrieved = await api.recall(bank, params.query);
  const evidence = [];
  for (const docId of params.docs ?? []) {
    evidence.push({ docId, facts: await api.listFacts(bank, { docId }) });
  }
  const detail =
    `retrieved:\n${retrieved
      .slice(0, 6)
      .map((r) => `- [${r.docId ?? "?"}] ${r.text}`)
      .join("\n")}` +
    evidence
      .map(
        (e) => `\n\nevidence ${e.docId}:\n${e.facts.map((f) => `- ${f.id} ${f.text}`).join("\n")}`
      )
      .join("");
  proj.addStep("trace", `"${params.query}"`, detail);
  await proj.save();
  return { retrieved, evidence };
}

/** Record a free-form step — e.g. an external eval result ("eval summer → FAIL"). */
export async function runLog(params: {
  projectDir: string;
  kind: string;
  summary: string;
  detail?: string;
}): Promise<void> {
  const proj = await Project.load(params.projectDir);
  proj.addStep(params.kind, params.summary, params.detail);
  await proj.save();
}

export interface CurateParams {
  projectDir: string;
  memoryId: string;
  kind: CurationKind;
  text?: string;
  reason?: string;
  apiKey?: string;
}

export async function runCurate(params: CurateParams, onProgress: ProgressFn): Promise<void> {
  const proj = await Project.load(params.projectDir);
  const bank = bankOrThrow(proj);
  const api = new SandboxApi(proj.apiUrl, resolveApiKey(params.apiKey ?? proj.apiKey));

  if (params.kind === "edit" && !params.text) throw new Error("--edit requires --text");
  const before = await api.getMemoryText(bank, params.memoryId).catch(() => "");

  const body =
    params.kind === "edit"
      ? { text: params.text, reason: params.reason }
      : {
          state: params.kind === "invalidate" ? ("invalidated" as const) : ("valid" as const),
          reason: params.reason,
        };
  await api.updateMemory(bank, params.memoryId, body);

  const stepDetail =
    params.kind === "edit"
      ? `"${before.slice(0, 80)}" → "${params.text}"${params.reason ? `\nreason: ${params.reason}` : ""}`
      : `"${before.slice(0, 100)}"${params.reason ? `\nreason: ${params.reason}` : ""}`;
  proj.addStep("curate", `${params.kind} ${params.memoryId.slice(0, 8)}`, stepDetail);
  proj.curations.push({
    id: `c${proj.curations.length + 1}`,
    memoryId: params.memoryId,
    kind: params.kind,
    before,
    after: params.kind === "edit" ? (params.text ?? null) : null,
    reason: params.reason ?? null,
    at: new Date().toISOString(),
  });
  await proj.save();
  onProgress(`${params.kind} ${params.memoryId} on ${bank}.`);
}

export interface SnapshotResult {
  total: number;
  docs: number;
}

export async function runSnapshot(
  params: { projectDir: string; apiKey?: string },
  onProgress: ProgressFn
): Promise<SnapshotResult> {
  const proj = await Project.load(params.projectDir);
  const bank = bankOrThrow(proj);
  const api = new SandboxApi(proj.apiUrl, resolveApiKey(params.apiKey ?? proj.apiKey));

  onProgress(`Freezing golden memory set from ${bank}…`);
  const rows = await api.listFacts(bank);
  proj.golden = rows.map((r) => {
    // Attach curation provenance: the latest edit applied to this memory, if any.
    const edit = [...proj.curations]
      .reverse()
      .find((c) => c.memoryId === r.id && c.kind === "edit");
    return {
      docId: r.docId ?? "",
      text: r.text,
      factType: r.factType,
      curatedFrom: edit?.before ?? null,
      curateReason: edit?.reason ?? null,
    };
  });
  proj.goldenAt = new Date().toISOString();
  const docCount = new Set(proj.golden.map((g) => g.docId)).size;
  proj.addStep("snapshot", `froze ${proj.golden.length} golden memories across ${docCount} docs`);
  await proj.save();

  const docs = new Set(proj.golden.map((g) => g.docId)).size;
  onProgress(`Golden snapshot: ${proj.golden.length} memories across ${docs} doc(s).`);
  return { total: proj.golden.length, docs };
}

/**
 * Phase 2 — does the current retain mission reproduce the GOLDEN snapshot?
 * For each doc, run a **dry-run extraction** (the API extracts at chunk level and returns candidate
 * facts WITHOUT persisting), then score coverage of that doc's golden memories with one LLM call.
 * No re-ingest, no scratch bank, no recall — just extraction fidelity vs the golden target.
 */

import { collectDocuments } from "./docs.js";
import { resolveApiKey, SandboxApi } from "./hindsight.js";
import { MissionLlm, type GoldenForCoverage } from "./llm.js";
import { documentId } from "./pipeline.js";
import { Project } from "./store.js";
import type { CheckResult, ProgressFn } from "./types.js";

export interface CheckParams {
  projectDir: string;
  /** Limit the check to specific docIds (defaults to every doc that has golden memories). */
  docs?: string[];
  model?: string;
  apiKey?: string;
}

export interface DocCoverage {
  docId: string;
  covered: number;
  total: number;
  missing: string[];
}

export interface CheckSummary {
  result: CheckResult;
  perDoc: DocCoverage[];
}

export async function runRetainCheck(
  params: CheckParams,
  onProgress: ProgressFn
): Promise<CheckSummary> {
  const proj = await Project.load(params.projectDir);
  if (proj.golden.length === 0) throw new Error("No golden snapshot — run `snapshot` first.");
  if (!proj.retain.mission) throw new Error("No retain mission set — run `retain mission` first.");
  const bank = proj.currentBank();
  if (!bank) throw new Error("No current bank — run `retain apply` first.");

  // Golden grouped by doc (carrying curation provenance for the coverage judge).
  const goldenByDoc = new Map<string, GoldenForCoverage[]>();
  for (const g of proj.golden) {
    const list = goldenByDoc.get(g.docId) ?? [];
    list.push({ text: g.text, curatedFrom: g.curatedFrom, curateReason: g.curateReason });
    goldenByDoc.set(g.docId, list);
  }
  // Documents on disk, keyed by their docId.
  const docs = await collectDocuments(proj.documents);
  const contentByDoc = new Map(docs.map((d) => [documentId(d.name), d.content]));

  const targetDocs = (params.docs ?? [...goldenByDoc.keys()]).filter((d) => goldenByDoc.has(d));
  if (targetDocs.length === 0) throw new Error("No matching docs to check.");

  const api = new SandboxApi(proj.apiUrl, resolveApiKey(params.apiKey ?? proj.apiKey));
  const llm = new MissionLlm({ apiKey: params.apiKey, model: params.model });

  const perDoc: DocCoverage[] = [];
  for (const docId of targetDocs) {
    const content = contentByDoc.get(docId);
    if (!content) {
      onProgress(`  ${docId}: no document content on disk — skipped`);
      continue;
    }
    onProgress(`  ${docId}: dry-run extracting + scoring coverage…`);
    // Dry-run extraction with the CURRENT working mission — no persistence, no bank mutation.
    const rows = await api.dryRunExtract(bank, content, { retainMission: proj.retain.mission });
    const candidate = rows.map((r) => r.text);

    const golden = goldenByDoc.get(docId) ?? [];
    const cov = await llm.coverage(golden, candidate);
    perDoc.push({
      docId,
      covered: cov.coveredIndices.length,
      total: golden.length,
      missing: cov.missing,
    });
    onProgress(`  ${docId}: ${cov.coveredIndices.length}/${golden.length} golden covered`);
  }

  const covered = perDoc.reduce((s, d) => s + d.covered, 0);
  const total = perDoc.reduce((s, d) => s + d.total, 0);
  const result: CheckResult = {
    coverage: total ? covered / total : 0,
    covered,
    total,
    docs: perDoc.length,
    at: new Date().toISOString(),
  };
  proj.lastCheck = result;
  const missing = perDoc.flatMap((d) => d.missing);
  proj.addStep(
    "retain check",
    `${(result.coverage * 100).toFixed(0)}% coverage (${covered}/${total} golden, ${perDoc.length} doc${perDoc.length === 1 ? "" : "s"})`,
    missing.length
      ? `missing:\n${missing.map((m) => `- ${m}`).join("\n")}`
      : "all golden reproduced"
  );
  await proj.save();

  onProgress(
    `\nCoverage: ${covered}/${total} golden memories (${(result.coverage * 100).toFixed(0)}%) across ${perDoc.length} doc(s).`
  );
  return { result, perDoc };
}

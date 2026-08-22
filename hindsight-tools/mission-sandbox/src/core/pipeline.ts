/** Shared ingest pipeline: provision a versioned bank, retain documents, consolidate. */

import type { LoadedDocument } from "./docs.js";
import { SandboxApi } from "./hindsight.js";
import type { ProgressFn } from "./types.js";

/** Derive a document_id from a filename (drop the extension). */
export function documentId(filename: string): string {
  const base = filename.replace(/^.*[/\\]/, "");
  return base.replace(/\.[^.]+$/, "");
}

/**
 * Create the bank with the given missions, ingest every document, then run consolidation.
 * Returns the resulting observation count. The bank is assumed fresh (a new version).
 */
export async function provisionAndIngest(
  api: SandboxApi,
  bankId: string,
  missions: { retainMission?: string | null; observationsMission?: string | null },
  documents: LoadedDocument[],
  onProgress: ProgressFn
): Promise<{ observationCount: number }> {
  onProgress(`Creating bank ${bankId}…`);
  await api.createBank(bankId, missions);

  onProgress(`Ingesting ${documents.length} document(s)…`);
  for (const doc of documents) {
    onProgress(`  Retaining: ${doc.name} (${doc.content.length} chars)`);
    await api.retain(bankId, doc.content, documentId(doc.name));
  }

  onProgress("Triggering consolidation…");
  await api.triggerConsolidation(bankId);
  await api.waitForConsolidation(bankId, { onProgress });
  onProgress("Consolidation complete.");

  return { observationCount: (await api.getStats(bankId)).totalObservations };
}

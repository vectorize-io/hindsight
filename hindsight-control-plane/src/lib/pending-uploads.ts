// Build the Documents view's in-flight/failed upload rows from the server's
// operations, so the status survives reloads, tabs and devices.
//
// An upload runs as two operations: file_convert_retain converts the file, then
// queues a separate retain for the same document_id and marks itself completed.
// The conversion finishing says nothing about the retain, so a completed
// conversion takes its status from that follow-up retain — otherwise an upload
// whose extraction is still running, or failed, silently drops off the list.

const PENDING_MAX_AGE_MS = 24 * 60 * 60 * 1000;
// An operation flips to "completed" a couple of seconds before the document
// becomes visible in listDocuments. Keep showing the pending row for that long
// so it stays on screen until the real document row takes over (dedup by
// document_id) — no flicker. Past it, older completed uploads (whose row simply
// sits on another page) don't resurface as phantom "Processing" rows.
export const PENDING_BRIDGE_MS = 30000;

export type PendingUpload = {
  operationId: string;
  id: string;
  filename: string | null;
  status: "processing" | "failed";
  error?: string | null;
  createdAt: string;
};

export type UploadOperation = {
  id: string;
  document_id: string | null;
  filename?: string | null;
  created_at: string;
  updated_at?: string | null;
  status: string;
  error_message: string | null;
};

export function pendingUploadsFromOperations(
  conversions: UploadOperation[],
  retains: UploadOperation[],
  now: number
): PendingUpload[] {
  const cutoff = now - PENDING_MAX_AGE_MS;
  const uploads: PendingUpload[] = [];
  for (const conversion of conversions) {
    const created = Date.parse(conversion.created_at);
    if (!Number.isFinite(created) || created < cutoff) continue;
    // The retain this conversion queued: the earliest one for the same document
    // created after the conversion was (a later one is a separate re-retain).
    const followUp =
      conversion.status === "completed" && conversion.document_id
        ? retains
            .filter(
              (op) =>
                op.document_id === conversion.document_id && Date.parse(op.created_at) >= created
            )
            .sort((a, b) => Date.parse(a.created_at) - Date.parse(b.created_at))[0]
        : undefined;
    const op = followUp ?? conversion;
    if (op.status === "completed") {
      const settled = Date.parse(op.updated_at || op.created_at);
      if (!Number.isFinite(settled) || now - settled >= PENDING_BRIDGE_MS) continue;
    } else if (op.status !== "pending" && op.status !== "processing" && op.status !== "failed") {
      continue;
    }
    uploads.push({
      // The failed operation, so deleting the row deletes what failed.
      operationId: op.id,
      id: conversion.document_id || conversion.id,
      filename: conversion.filename ?? null,
      status: op.status === "failed" ? "failed" : "processing",
      error: op.error_message,
      createdAt: conversion.created_at,
    });
  }
  return uploads;
}

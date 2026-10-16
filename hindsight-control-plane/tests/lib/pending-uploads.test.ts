import { describe, expect, it } from "vitest";
import {
  PENDING_BRIDGE_MS,
  pendingUploadsFromOperations,
  type UploadOperation,
} from "@/lib/pending-uploads";

const NOW = Date.parse("2026-10-16T12:00:00Z");
const ago = (ms: number) => new Date(NOW - ms).toISOString();

function op(overrides: Partial<UploadOperation>): UploadOperation {
  return {
    id: "op",
    document_id: "doc-1",
    filename: "report.pdf",
    created_at: ago(10 * 60_000),
    updated_at: ago(10 * 60_000),
    status: "completed",
    error_message: null,
    ...overrides,
  };
}

// The conversion finished long ago (past the bridge); the retain it queued is what matters.
const conversion = op({ id: "convert", created_at: ago(10 * 60_000), updated_at: ago(9 * 60_000) });

describe("pendingUploadsFromOperations", () => {
  it("keeps an upload failed when its follow-up retain fails after the bridge (#5310)", () => {
    const retain = op({
      id: "retain",
      filename: null,
      created_at: ago(9 * 60_000),
      updated_at: ago(5 * 60_000),
      status: "failed",
      error_message: "Fact extraction failed: 1/1 chunks failed.",
    });
    expect(pendingUploadsFromOperations([conversion], [retain], NOW)).toEqual([
      {
        operationId: "retain",
        id: "doc-1",
        filename: "report.pdf",
        status: "failed",
        error: "Fact extraction failed: 1/1 chunks failed.",
        createdAt: conversion.created_at,
      },
    ]);
  });

  it("keeps an upload processing while its follow-up retain runs", () => {
    const retain = op({ id: "retain", created_at: ago(9 * 60_000), status: "processing" });
    const [row] = pendingUploadsFromOperations([conversion], [retain], NOW);
    expect(row.status).toBe("processing");
  });

  it("drops the upload once its retain completed past the bridge", () => {
    const retain = op({ id: "retain", created_at: ago(9 * 60_000), updated_at: ago(PENDING_BRIDGE_MS + 1) });
    expect(pendingUploadsFromOperations([conversion], [retain], NOW)).toEqual([]);
  });

  it("bridges a just-completed retain until the document row takes over", () => {
    const retain = op({ id: "retain", created_at: ago(9 * 60_000), updated_at: ago(1000) });
    expect(pendingUploadsFromOperations([conversion], [retain], NOW)).toHaveLength(1);
  });

  it("ignores retains for the same document from before the conversion", () => {
    const older = op({ id: "old", created_at: ago(60 * 60_000), status: "failed" });
    expect(pendingUploadsFromOperations([conversion], [older], NOW)).toEqual([]);
  });

  it("still reports a failed conversion itself", () => {
    const failed = op({ id: "convert", status: "failed", error_message: "parse error" });
    const [row] = pendingUploadsFromOperations([failed], [], NOW);
    expect(row).toMatchObject({ operationId: "convert", status: "failed", error: "parse error" });
  });
});

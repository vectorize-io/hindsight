"use client";

const STORAGE_KEY = "hindsight.pendingDocuments.v1";
const CHANGE_EVENT = "hindsight:pending-documents-changed";
const MAX_AGE_MS = 24 * 60 * 60 * 1000;

export type PendingDocumentStatus = "processing" | "failed";

export interface PendingDocument {
  id: string;
  bankId: string;
  filename: string;
  size: number;
  tags: string[];
  operationId?: string;
  createdAt: string;
  status: PendingDocumentStatus;
  error?: string;
}

function canUseStorage() {
  return typeof window !== "undefined" && typeof window.sessionStorage !== "undefined";
}

function readAll(): PendingDocument[] {
  if (!canUseStorage()) return [];

  try {
    const raw = window.sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];

    const now = Date.now();
    return parsed.filter((item): item is PendingDocument => {
      if (!item || typeof item !== "object") return false;
      if (typeof item.id !== "string" || typeof item.bankId !== "string") return false;
      const createdAt = typeof item.createdAt === "string" ? Date.parse(item.createdAt) : NaN;
      return Number.isFinite(createdAt) && now - createdAt <= MAX_AGE_MS;
    });
  } catch {
    return [];
  }
}

function writeAll(items: PendingDocument[]) {
  if (!canUseStorage()) return;
  window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(items));
  window.dispatchEvent(new Event(CHANGE_EVENT));
}

export function getPendingDocuments(bankId: string): PendingDocument[] {
  return readAll().filter((item) => item.bankId === bankId);
}

export function upsertPendingDocuments(items: PendingDocument[]) {
  if (items.length === 0) return;

  const nextByKey = new Map(readAll().map((item) => [`${item.bankId}:${item.id}`, item]));
  for (const item of items) {
    nextByKey.set(`${item.bankId}:${item.id}`, item);
  }
  writeAll([...nextByKey.values()]);
}

export function removePendingDocuments(bankId: string, documentIds: string[]) {
  if (documentIds.length === 0) return;

  const ids = new Set(documentIds);
  writeAll(readAll().filter((item) => item.bankId !== bankId || !ids.has(item.id)));
}

export function updatePendingDocumentStatus(
  bankId: string,
  documentId: string,
  status: PendingDocumentStatus,
  error?: string
) {
  let changed = false;
  const next = readAll().map((item) => {
    if (item.bankId !== bankId || item.id !== documentId) return item;
    changed = true;
    return { ...item, status, error };
  });

  if (changed) writeAll(next);
}

export function subscribePendingDocuments(listener: () => void) {
  if (typeof window === "undefined") return () => {};

  const onStorage = (event: StorageEvent) => {
    if (event.key === STORAGE_KEY) listener();
  };

  window.addEventListener(CHANGE_EVENT, listener);
  window.addEventListener("storage", onStorage);

  return () => {
    window.removeEventListener(CHANGE_EVENT, listener);
    window.removeEventListener("storage", onStorage);
  };
}

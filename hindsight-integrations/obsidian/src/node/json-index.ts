/**
 * JSON-file persistence for the sync index, the CLI's equivalent of the plugin's
 * `data.json`. Kept out of the vault by default (see {@link defaultIndexPath})
 * so Obsidian Sync never propagates it and it can't collide with the plugin's
 * own index on another machine.
 *
 * The index is *bound* to the destination it was built against — API origin,
 * bank, vault, and the document-id namespace (`vaultName` + `prefixDocId`).
 * Reusing a vault-name-only index across banks/targets would silently classify
 * the old target's files as "already synced" against the new one (leaving it
 * incomplete) and could authorize prune DELETEs against a bank this ingester
 * never wrote to. So the default path is target-scoped and {@link loadIndex}
 * fails closed when the persisted destination doesn't match (issue #3257).
 *
 * Include/exclude scope is deliberately *not* bound: on the same destination,
 * narrowing scope legitimately prunes the newly-excluded docs (this ingester
 * owns them there). The described cross-target harms all require a destination
 * change, which is what this binding refuses.
 */

import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import type { SyncIndex } from "../sync";

/**
 * The destination an index is bound to. Two indexes are interchangeable only
 * when every field here matches; any difference means a distinct target whose
 * sync state must not be shared. Scope (include/exclude) is intentionally absent
 * — see the module header.
 */
export interface IndexIdentity {
  /** Canonical, non-secret API origin (protocol + host + port), no path/token. */
  apiOrigin: string;
  bankId: string;
  /** Resolved absolute vault path — the on-disk source of truth. */
  vaultPath: string;
  /** With {@link prefixDocId}, part of the document-id namespace. */
  vaultName: string;
  prefixDocId: boolean;
}

interface IndexFile {
  /** Envelope schema version; bump when the on-disk shape changes. */
  version: number;
  /** Destination this index was built against (see {@link IndexIdentity}). */
  identity: IndexIdentity;
  syncIndex: SyncIndex;
  lastSyncAt: string | null;
}

const ENVELOPE_VERSION = 1;

/** Raised when an existing index belongs to a different (or unknown) target. */
export class IndexIdentityError extends Error {}

/** Canonical origin of an API base URL — drops path, query, and any credentials. */
export function canonicalApiOrigin(apiUrl: string): string {
  try {
    return new URL(apiUrl).origin;
  } catch {
    // Not a parseable URL (e.g. a bare host in a test); normalize trailing slashes.
    return apiUrl.replace(/\/+$/, "");
  }
}

function sanitize(part: string): string {
  return part.replace(/[^A-Za-z0-9._-]+/g, "_") || "x";
}

/**
 * Short, stable fingerprint of the destination identity. Two targets that differ
 * in any bound field get different fingerprints, so their default index files
 * never collide even when the vault name is identical.
 */
export function identityFingerprint(identity: IndexIdentity): string {
  const canonical = JSON.stringify([
    identity.apiOrigin,
    identity.bankId,
    identity.vaultPath,
    identity.vaultName,
    identity.prefixDocId,
  ]);
  return createHash("sha256").update(canonical).digest("hex").slice(0, 12);
}

/**
 * Default out-of-vault index location, target-scoped so a different bank/API/
 * vault lands in its own file: `~/.hindsight/obsidian/<vault>-<bank>-<fp>.json`.
 */
export function defaultIndexPath(identity: IndexIdentity): string {
  const name = `${sanitize(identity.vaultName)}-${sanitize(identity.bankId)}-${identityFingerprint(identity)}`;
  return join(homedir(), ".hindsight", "obsidian", `${name}.json`);
}

/** Fields whose divergence between two identities makes them incompatible. */
function identityMismatches(want: IndexIdentity, have: IndexIdentity): string[] {
  const keys: (keyof IndexIdentity)[] = [
    "apiOrigin",
    "bankId",
    "vaultPath",
    "vaultName",
    "prefixDocId",
  ];
  return keys
    .filter((key) => want[key] !== have[key])
    .map((key) => `${key}: ${JSON.stringify(have[key])} → ${JSON.stringify(want[key])}`);
}

/**
 * Load a previously-persisted index for {@link identity}, or an empty one if
 * absent/unreadable. Throws {@link IndexIdentityError} — fail closed — when the
 * file was built against a different target, or predates identity binding, so
 * we never silently skip files or mis-attribute deletes across banks.
 */
export async function loadIndex(path: string, identity: IndexIdentity): Promise<SyncIndex> {
  let raw: string;
  try {
    raw = await readFile(path, "utf8");
  } catch {
    return {}; // first run — no index yet
  }

  let data: Partial<IndexFile>;
  try {
    data = JSON.parse(raw) as Partial<IndexFile>;
  } catch {
    // A corrupt index means a full re-ingest (hash-skips unchanged) rather than
    // a crash; warn because orphan pruning can't run until the index is rebuilt.
    console.warn(`[hindsight] ignoring unreadable sync index at ${path}; starting fresh`);
    return {};
  }

  if (!data.identity) {
    // Legacy (pre-binding) index: adopt explicitly rather than silently trust it.
    throw new IndexIdentityError(
      `sync index at ${path} predates target binding (no identity metadata) and cannot be safely reused.\n` +
        `Delete it to re-sync bank "${identity.bankId}" from scratch, or pass --index <fresh path>.`
    );
  }

  const diffs = identityMismatches(identity, data.identity);
  if (diffs.length > 0) {
    throw new IndexIdentityError(
      `sync index at ${path} is bound to a different target and cannot be reused:\n` +
        diffs.map((d) => `  - ${d}`).join("\n") +
        `\nReusing it would leave the new target incomplete or mis-attribute deletes. ` +
        `Use a distinct --index path for this target, or delete ${path} to re-sync from scratch.`
    );
  }

  return data.syncIndex ?? {};
}

/**
 * A `persist` callback for {@link SyncEngine} that atomically writes the index,
 * stamping the {@link identity} it belongs to so future loads can validate it.
 */
export function makePersist(
  path: string,
  identity: IndexIdentity,
  nowIso: () => string = () => new Date().toISOString()
) {
  return async (index: SyncIndex): Promise<void> => {
    await mkdir(dirname(path), { recursive: true });
    const payload: IndexFile = {
      version: ENVELOPE_VERSION,
      identity,
      syncIndex: index,
      lastSyncAt: nowIso(),
    };
    const tmp = `${path}.tmp`;
    await writeFile(tmp, `${JSON.stringify(payload, null, 2)}\n`);
    await rename(tmp, path); // atomic replace — never leave a half-written index
  };
}

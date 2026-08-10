/**
 * Headless CLI frontend for vault → Hindsight ingestion.
 *
 * It is a thin second frontend over the same {@link SyncEngine} the Obsidian
 * plugin uses, so a server-synced vault produces identical document ids, tags,
 * and prune-ownership as the plugin — the two never fight or duplicate. The only
 * differences are the vault source (filesystem vs Obsidian API), the transport
 * (`fetch` vs `requestUrl`), and where the sync index lives (a JSON file vs the
 * plugin's `data.json`).
 *
 * Everything here is exported and side-effect-free for testing; the executable
 * entrypoint lives in `cli-bin.ts`.
 */

import type { FSWatcher } from "chokidar";
import { basename, relative, resolve, sep } from "node:path";
import { parseArgs } from "node:util";
import { HindsightClient } from "../client";
import { SyncEngine, type ReconcileSummary, type SyncConfig } from "../sync";
import type { Transport } from "../transport";
import { fetchTransport } from "./fetch-transport";
import { FsVault } from "./fs-vault";
import {
  canonicalApiOrigin,
  defaultIndexPath,
  type IndexIdentity,
  loadIndex,
  makePersist,
} from "./json-index";

export interface CliOptions {
  /** Absolute path to the vault root. */
  vault: string;
  bank: string;
  apiUrl: string;
  apiToken?: string;
  include: string[];
  exclude: string[];
  vaultName: string;
  prefixDocId: boolean;
  indexPath: string;
  watch: boolean;
  /** Destination the sync index is bound to (issue #3257). */
  identity: IndexIdentity;
}

const USAGE = `hindsight-obsidian-sync — ingest an Obsidian vault into Hindsight (headless).

Usage:
  hindsight-obsidian-sync reconcile --vault <path> --bank <id> [options]

Options:
  --vault <path>        Vault root directory (required)
  --bank <id>           Target Hindsight bank id (required)
  --api-url <url>       Hindsight API base URL (or env HINDSIGHT_API_URL)
  --api-token <token>   API token (or env HINDSIGHT_API_TOKEN)
  --include <folder>    Only sync this folder (repeatable; default: whole vault)
  --exclude <folder>    Skip this folder (repeatable)
  --vault-name <name>   Vault name for tags/ids (default: the vault dir name)
  --prefix-doc-id       Prefix document ids with the vault name (multi-vault banks)
  --index <file>        Sync-index JSON path (default: a per-target file under
                        ~/.hindsight/obsidian/, scoped to bank + API + vault)
  --watch               Keep running and sync changes as they happen
  --help                Show this help
`;

/** Thrown for invalid invocations; its message is printed to stderr (exit 2). */
export class UsageError extends Error {}

/** Thrown for `--help`; its message is printed to stdout (exit 0). */
export class HelpRequested extends Error {}

/** Parse argv into fully-resolved options (applying env fallbacks + defaults). */
export function parseCliArgs(argv: string[]): CliOptions {
  const { values, positionals } = parseArgs({
    args: argv,
    allowPositionals: true,
    options: {
      vault: { type: "string" },
      bank: { type: "string" },
      "api-url": { type: "string" },
      "api-token": { type: "string" },
      include: { type: "string", multiple: true },
      exclude: { type: "string", multiple: true },
      "vault-name": { type: "string" },
      "prefix-doc-id": { type: "boolean", default: false },
      index: { type: "string" },
      watch: { type: "boolean", default: false },
      help: { type: "boolean", default: false },
    },
  });

  if (values.help) throw new HelpRequested(USAGE);

  // The only accepted subcommand is "reconcile" (also the default).
  const command = positionals[0] ?? "reconcile";
  if (command !== "reconcile") throw new UsageError(`unknown command "${command}"\n\n${USAGE}`);

  const vaultRaw = values.vault;
  if (!vaultRaw) throw new UsageError("--vault is required");
  if (!values.bank) throw new UsageError("--bank is required");

  const vault = resolve(vaultRaw);
  const vaultName = values["vault-name"] || basename(vault);
  const apiUrl = values["api-url"] || process.env.HINDSIGHT_API_URL || "";
  if (!apiUrl) throw new UsageError("--api-url is required (or set HINDSIGHT_API_URL)");

  const identity: IndexIdentity = {
    apiOrigin: canonicalApiOrigin(apiUrl),
    bankId: values.bank,
    vaultPath: vault,
    vaultName,
    prefixDocId: values["prefix-doc-id"] ?? false,
  };

  return {
    vault,
    bank: values.bank,
    apiUrl,
    apiToken: values["api-token"] || process.env.HINDSIGHT_API_TOKEN || undefined,
    include: values.include ?? [],
    exclude: values.exclude ?? [],
    vaultName,
    prefixDocId: values["prefix-doc-id"] ?? false,
    indexPath: values.index || defaultIndexPath(identity),
    watch: values.watch ?? false,
    identity,
  };
}

export function buildConfig(opts: CliOptions): SyncConfig {
  return {
    bankId: opts.bank,
    includeFolders: opts.include,
    excludeFolders: opts.exclude,
    vaultName: opts.vaultName,
    prefixDocId: opts.prefixDocId,
  };
}

/**
 * Assemble the engine from options. `transport` is injectable so tests can drive
 * the full stack (FsVault → SyncEngine → HindsightClient) without real network.
 */
export async function buildEngine(
  opts: CliOptions,
  transport: Transport = fetchTransport
): Promise<{ engine: SyncEngine; vault: FsVault }> {
  const vault = new FsVault(opts.vault);
  const client = new HindsightClient(opts.apiUrl, opts.apiToken, transport);
  const index = await loadIndex(opts.indexPath, opts.identity);
  const engine = new SyncEngine(
    client,
    vault,
    buildConfig(opts),
    index,
    makePersist(opts.indexPath, opts.identity)
  );
  return { engine, vault };
}

/**
 * Map filesystem events to engine operations. A rename arrives from chokidar as
 * unlink(old) + add(new), which the engine handles correctly as delete-then-
 * create — same net result as the plugin's dedicated rename path.
 */
export function createWatchHandlers(engine: SyncEngine, vault: FsVault) {
  return {
    async onUpsert(relPath: string): Promise<void> {
      const file = vault.syncFileFor(relPath);
      if (file) await engine.ingestFile(file, { force: true });
    },
    async onUnlink(relPath: string): Promise<void> {
      await engine.handleDelete(relPath);
    },
  };
}

function formatSummary(label: string, s: ReconcileSummary): string {
  return `${label}: +${s.added} added, ~${s.updated} updated, -${s.deleted} deleted, =${s.unchanged} unchanged`;
}

/** The subset of chokidar options tests tune for deterministic, fast watching. */
export interface WatchOverrides {
  usePolling?: boolean;
  interval?: number;
  awaitWriteFinish?: boolean | { stabilityThreshold?: number; pollInterval?: number };
}

/**
 * Wire a chokidar watcher over the vault to the engine and return it (live).
 * Split out from {@link startWatch} so tests can drive real filesystem events
 * and then close the watcher, instead of blocking forever. `overrides` lets
 * tests tighten the debounce / force polling for deterministic CI behavior.
 */
export async function watchVault(
  opts: CliOptions,
  engine: SyncEngine,
  vault: FsVault,
  overrides: WatchOverrides = {}
): Promise<FSWatcher> {
  const { watch } = await import("chokidar");
  const handlers = createWatchHandlers(engine, vault);
  const toRel = (abs: string) => relative(opts.vault, abs).split(sep).join("/");
  const isMarkdown = (p: string) => p.toLowerCase().endsWith(".md");
  const report = (err: unknown) =>
    console.error(`[hindsight] ${err instanceof Error ? err.message : String(err)}`);

  const watcher = watch(opts.vault, {
    ignoreInitial: true,
    persistent: true,
    // Skip dotfolders (.obsidian/.trash/.git); coalesce partial saves.
    ignored: (p: string) => p.split(sep).some((seg) => seg.startsWith(".")),
    awaitWriteFinish: { stabilityThreshold: 300, pollInterval: 100 },
    ...overrides,
  });
  watcher.on("add", (p: string) => isMarkdown(p) && handlers.onUpsert(toRel(p)).catch(report));
  watcher.on("change", (p: string) => isMarkdown(p) && handlers.onUpsert(toRel(p)).catch(report));
  watcher.on("unlink", (p: string) => isMarkdown(p) && handlers.onUnlink(toRel(p)).catch(report));
  return watcher;
}

/** Start watch mode: an initial reconcile, then live sync until interrupted. */
export async function startWatch(
  opts: CliOptions,
  engine: SyncEngine,
  vault: FsVault,
  log: (msg: string) => void = console.log
): Promise<void> {
  log(formatSummary("initial reconcile", await engine.reconcile()));
  await watchVault(opts, engine, vault); // returns a live watcher; let it run
  log(`watching ${opts.vault} for changes (Ctrl-C to stop)`);
  await new Promise<never>(() => {}); // run until the process is signalled
}

/** Run the CLI. Returns a process exit code; never calls process.exit itself. */
export async function runCli(
  argv: string[],
  log: (msg: string) => void = console.log,
  logErr: (msg: string) => void = console.error
): Promise<number> {
  let opts: CliOptions;
  try {
    opts = parseCliArgs(argv);
  } catch (err) {
    if (err instanceof HelpRequested) {
      log(err.message);
      return 0;
    }
    logErr(err instanceof UsageError ? err.message : `hindsight-obsidian-sync: ${describe(err)}`);
    return 2;
  }
  try {
    const { engine, vault } = await buildEngine(opts);
    if (opts.watch) {
      await startWatch(opts, engine, vault, log);
      return 0;
    }
    log(formatSummary("reconcile", await engine.reconcile()));
    return 0;
  } catch (err) {
    logErr(`hindsight-obsidian-sync: ${describe(err)}`);
    return 1;
  }
}

function describe(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

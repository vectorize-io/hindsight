#!/usr/bin/env node
/**
 * hindsight-fs CLI.
 *
 * Mirrors a Hindsight bank's knowledge base (folders + pages) as a folder of
 * markdown files that stay current via a background refresh loop. Once mounted,
 * ordinary shell tools (ls, cat, grep, find, rg, fzf …) work against real files.
 */

import { promises as fs } from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { resolveConfig, saveConfig, type ConfigOverrides, type MountConfig } from "./config.js";
import { runSync } from "./sync.js";
import { runLoop } from "./loop.js";
import { HindsightFsClient } from "./client.js";
import { startDaemon, stopDaemon, logPath } from "./daemon.js";
import { planMirror } from "./format.js";
import { loadState } from "./state.js";
import { computeHealth } from "./health.js";
import { CONTROL_DIR } from "./paths.js";

// ── Arg parsing ────────────────────────────────────────

interface ParsedArgs {
  command: string;
  positionals: string[];
  flags: Record<string, string | boolean>;
}

const FLAG_ALIASES: Record<string, string> = {
  b: "bank",
  u: "api-url",
  t: "token",
  i: "interval",
  d: "dir",
  h: "help",
  v: "version",
};

const BOOLEAN_FLAGS = new Set([
  "once",
  "detach",
  "help",
  "version",
  "full",
  "quiet",
  "writable",
  "json",
]);

function parseArgs(argv: string[]): ParsedArgs {
  const positionals: string[] = [];
  const flags: Record<string, string | boolean> = {};

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg.startsWith("--")) {
      const eq = arg.indexOf("=");
      let name = eq === -1 ? arg.slice(2) : arg.slice(2, eq);
      name = FLAG_ALIASES[name] ?? name;
      if (eq !== -1) {
        flags[name] = arg.slice(eq + 1);
      } else if (BOOLEAN_FLAGS.has(name)) {
        flags[name] = true;
      } else {
        flags[name] = argv[++i] ?? "";
      }
    } else if (arg.startsWith("-") && arg.length > 1) {
      const name = FLAG_ALIASES[arg.slice(1)] ?? arg.slice(1);
      if (BOOLEAN_FLAGS.has(name)) {
        flags[name] = true;
      } else {
        flags[name] = argv[++i] ?? "";
      }
    } else {
      positionals.push(arg);
    }
  }

  const command = positionals.shift() ?? "help";
  return { command, positionals, flags };
}

function overridesFrom(args: ParsedArgs): ConfigOverrides {
  const o: ConfigOverrides = {};
  // For dir-taking commands, the first positional is the mount dir.
  if (args.positionals[0]) o.dir = args.positionals[0];
  if (typeof args.flags.dir === "string" && args.flags.dir) o.dir = args.flags.dir;
  if (typeof args.flags.bank === "string" && args.flags.bank) o.bankId = args.flags.bank;
  if (typeof args.flags["api-url"] === "string" && args.flags["api-url"])
    o.apiUrl = args.flags["api-url"];
  if (typeof args.flags.token === "string" && args.flags.token) o.apiToken = args.flags.token;
  if (typeof args.flags.interval === "string" && args.flags.interval) {
    o.intervalSeconds = Number(args.flags.interval);
  }
  if (args.flags.writable === true) o.writable = true;
  return o;
}

// ── Output helpers ─────────────────────────────────────

function out(msg: string): void {
  process.stdout.write(msg + "\n");
}
function err(msg: string): void {
  process.stderr.write(msg + "\n");
}
function stamp(): string {
  return new Date().toISOString().replace("T", " ").slice(0, 19);
}

// ── Commands ───────────────────────────────────────────

async function cmdSync(args: ParsedArgs): Promise<void> {
  const config = await resolveConfig(overridesFrom(args), { requireBank: true });
  await saveConfig(config);
  const result = await runSync(config);
  const reverted = result.reverted > 0 ? `, ${result.reverted} reverted` : "";
  out(
    `Synced ${result.total} pages / ${result.folders} folders into ${config.dir} ` +
      `(${result.written} updated, ${result.unchanged} unchanged, ${result.removed} removed${reverted})`
  );
}

async function cmdMount(args: ParsedArgs): Promise<void> {
  const config = await resolveConfig(overridesFrom(args), { requireBank: true });
  await saveConfig(config);

  if (args.flags.once === true) {
    await cmdSync(args);
    return;
  }

  if (args.flags.detach === true) {
    const { pid, alreadyRunning } = await startDaemon(config);
    if (alreadyRunning) {
      out(`Already mounted at ${config.dir} (daemon pid ${pid}).`);
    } else {
      out(`Mounted bank "${config.bankId}" at ${config.dir} in background (pid ${pid}).`);
      out(`Logs: ${logPath(config.dir)} — stop with: hindsight-fs stop ${config.dir}`);
    }
    return;
  }

  // Foreground: run until Ctrl-C.
  const controller = new AbortController();
  const onSignal = () => controller.abort();
  process.on("SIGINT", onSignal);
  process.on("SIGTERM", onSignal);
  await runLoop(config, {
    signal: controller.signal,
    log: (m) => err(`[${stamp()}] ${m}`),
  });
}

async function cmdStart(args: ParsedArgs): Promise<void> {
  await cmdMount({ ...args, flags: { ...args.flags, detach: true } });
}

async function cmdStop(args: ParsedArgs): Promise<void> {
  const config = await resolveConfig(overridesFrom(args));
  const { stopped, pid } = await stopDaemon(config.dir);
  if (stopped) out(`Stopped daemon (pid ${pid}) for ${config.dir}.`);
  else if (pid) out(`Daemon for ${config.dir} was not running (cleaned up stale pid ${pid}).`);
  else out(`No daemon registered for ${config.dir}.`);
}

async function cmdRestart(args: ParsedArgs): Promise<void> {
  await cmdStop(args);
  await cmdStart(args);
}

async function cmdStatus(args: ParsedArgs): Promise<void> {
  const config = await resolveConfig(overridesFrom(args));
  const staleAfterSeconds =
    typeof args.flags["stale-after"] === "string" && args.flags["stale-after"]
      ? Number(args.flags["stale-after"])
      : undefined;
  const report = await computeHealth(config, { staleAfterSeconds });

  if (args.flags.json === true) {
    out(JSON.stringify(report, null, 2));
  } else {
    const d = report.daemon;
    const age =
      report.lastSync.ageSeconds === null ? "never" : `${report.lastSync.ageSeconds}s ago`;
    out(`Mount:    ${report.mount}`);
    out(`Bank:     ${report.bank || "(unset)"}`);
    out(`API:      ${report.apiUrl}`);
    out(
      `Mode:     ${report.mode === "writable" ? "writable (one-way; edits reverted on refresh)" : "read-only (edits blocked)"}`
    );
    out(`Daemon:   ${d.running ? `running (pid ${d.pid})` : "stopped"}`);
    if (d.startedAt) out(`Interval: ${d.intervalSeconds}s (started ${d.startedAt})`);
    out(
      `Last sync: ${report.lastSync.at ?? "never"} (${age})${report.lastSync.ok ? "" : " (FAILED)"}`
    );
    if (report.lastSync.error) out(`Last error: ${report.lastSync.error}`);
    out(`Mirrored: ${report.mirroredFiles} file(s)`);
    out(`Health:   ${report.healthy ? "ok" : report.status.toUpperCase()}`);
  }

  if (!report.healthy) process.exitCode = 1;
}

async function cmdList(args: ParsedArgs): Promise<void> {
  const config = await resolveConfig(overridesFrom(args), { requireBank: true });
  const client = new HindsightFsClient({ apiUrl: config.apiUrl, apiToken: config.apiToken });
  const snapshot = await client.loadKnowledge(config.bankId);
  const plan = planMirror(snapshot);
  if (plan.dirs.length === 0 && plan.files.length === 0) {
    out(`No knowledge base in bank "${config.bankId}".`);
    return;
  }
  for (const dir of plan.dirs) out(`${dir}/`);
  for (const page of plan.files) out(page.relPath);
}

async function cmdUnmount(args: ParsedArgs): Promise<void> {
  const config = await resolveConfig(overridesFrom(args));
  await stopDaemon(config.dir);

  const state = await loadState(config.dir, config.bankId, config.apiUrl);
  let removed = 0;
  for (const entry of Object.values(state.files)) {
    try {
      await fs.unlink(path.join(config.dir, entry.file));
      removed++;
    } catch {
      /* already gone */
    }
  }
  await fs.rm(path.join(config.dir, CONTROL_DIR), { recursive: true, force: true });
  out(`Unmounted ${config.dir} — removed ${removed} mirrored file(s) and control data.`);
}

async function cmdLogs(args: ParsedArgs): Promise<void> {
  const config = await resolveConfig(overridesFrom(args));
  try {
    const log = await fs.readFile(logPath(config.dir), "utf8");
    const lines = log.split("\n");
    const tail = lines.slice(Math.max(0, lines.length - 40)).join("\n");
    process.stdout.write(tail.endsWith("\n") ? tail : tail + "\n");
  } catch {
    out(`No logs for ${config.dir}.`);
  }
}

/** Hidden entrypoint used by the detached daemon process. */
async function cmdRun(args: ParsedArgs): Promise<void> {
  const config = await resolveConfig(overridesFrom(args), { requireBank: true });
  const controller = new AbortController();
  const onSignal = () => controller.abort();
  process.on("SIGTERM", onSignal);
  process.on("SIGINT", onSignal);
  await runLoop(config, {
    signal: controller.signal,
    log: (m) => process.stdout.write(`[${stamp()}] ${m}\n`),
  });
  process.exit(0);
}

async function readVersion(): Promise<string> {
  try {
    const pkgPath = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "package.json");
    const pkg = JSON.parse(await fs.readFile(pkgPath, "utf8")) as { version?: string };
    return pkg.version ?? "0.0.0";
  } catch {
    return "0.0.0";
  }
}

function printHelp(): void {
  out(`hindsight-fs — mount a Hindsight bank's knowledge base as a live local folder

USAGE
  hindsight-fs <command> [dir] [options]

COMMANDS
  mount [dir]      Mirror the bank into <dir> and keep it refreshed (foreground; Ctrl-C to stop)
                   Add --detach to run in the background, or --once for a single pass.
  start [dir]      Mount in the background (alias for: mount --detach)
  stop [dir]       Stop the background daemon for <dir>
  restart [dir]    Restart the background daemon
  sync [dir]       Run a single refresh pass and exit
  status [dir]     Show daemon + last-sync health for <dir>
                   Add --json for a machine-readable report. Exits non-zero when
                   the mount is unhealthy (dead, failed, or stale).
  list             List the bank's knowledge-base folders + pages (no files written)
  logs [dir]       Print the tail of the background daemon log
  unmount [dir]    Stop the daemon and delete mirrored files + control data
  help             Show this help
  version          Show the version

OPTIONS
  -b, --bank <id>        Bank to mirror (env: HINDSIGHT_BANK_ID)
  -u, --api-url <url>    API base URL (env: HINDSIGHT_API_URL, default http://localhost:8000)
  -t, --token <token>    Bearer token (env: HINDSIGHT_API_TOKEN)
  -i, --interval <sec>   Refresh interval in seconds (default 30)
  -d, --dir <path>       Mount directory (overrides positional; env: HINDSIGHT_FS_DIR)
      --writable         Make mirrored files editable (default: read-only)
      --once             For 'mount': run a single pass instead of looping
      --detach           For 'mount': run in the background
      --json             For 'status': print a machine-readable health report
      --stale-after <s>  For 'status': seconds before a sync is "stale"
                         (default: max(interval × 3, 15))

FILES
  Folders become directories and pages become <name>.md (YAML frontmatter +
  markdown body), nested to match the knowledge-base tree. The mirror is one-way
  (API → disk). By default files are read-only (mode 0444) so agents cannot edit
  them; any drift is also reverted on the next refresh. Control data lives in
  <dir>/.hindsight-fs/ (config, state, daemon log, index.md).

EXAMPLES
  hindsight-fs mount ./kb --bank my-agent --interval 15
  hindsight-fs start ./kb --bank my-agent
  ls -R ./kb && cat ./kb/policies/billing-policy.md
  grep -ril "net-30" ./kb
  hindsight-fs stop ./kb`);
}

// ── Dispatch ───────────────────────────────────────────

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));

  if (args.flags.version === true || args.command === "version") {
    out(`hindsight-fs ${await readVersion()}`);
    return;
  }
  if (args.flags.help === true || args.command === "help") {
    printHelp();
    return;
  }

  switch (args.command) {
    case "mount":
      return cmdMount(args);
    case "start":
      return cmdStart(args);
    case "stop":
      return cmdStop(args);
    case "restart":
      return cmdRestart(args);
    case "sync":
      return cmdSync(args);
    case "status":
      return cmdStatus(args);
    case "list":
    case "ls":
      return cmdList(args);
    case "logs":
      return cmdLogs(args);
    case "unmount":
    case "umount":
      return cmdUnmount(args);
    case "__run":
      return cmdRun(args);
    default:
      err(`Unknown command: ${args.command}\n`);
      printHelp();
      process.exitCode = 1;
  }
}

main().catch((e) => {
  err(`Error: ${e instanceof Error ? e.message : String(e)}`);
  process.exitCode = 1;
});

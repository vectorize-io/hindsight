import { execFileSync, spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

/**
 * A compaction counts as a new session start (#6527), driven through the BUILT hook binaries as
 * separate processes, the way each host runs them: real event JSON on stdin, a stub Hindsight
 * server on the other end, the session's state handed from one process to the next in temp files.
 *
 * Each host reports a compaction differently — a SessionStart with `source: "compact"`, Cursor's
 * `preCompact`, Kimi's `PostCompact` — and every one must bring the memory back on the next prompt.
 *
 * Run with `npm run test:compaction-e2e` (it builds first).
 */
const packageRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const dist = join(packageRoot, "dist");
const MEMORY = "MEMORY-6527-retry-only-429-and-408";

interface HostEvents {
  harness: string;
  entry: { sessionStart: string; prompt: string };
  start(session: string, cwd: string): Record<string, unknown>;
  prompt(session: string, cwd: string, text: string): Record<string, unknown>;
  compaction(session: string, cwd: string): Record<string, unknown>;
}

const claudeStyle = (harness: string, prefix: string): HostEvents => ({
  harness,
  entry: { sessionStart: `${prefix}-sessionstart-hook.js`, prompt: `${prefix}-hook.js` },
  start: (session_id, cwd) => ({
    hook_event_name: "SessionStart",
    session_id,
    cwd,
    source: "startup",
  }),
  prompt: (session_id, cwd, prompt) => ({
    hook_event_name: "UserPromptSubmit",
    session_id,
    cwd,
    prompt,
  }),
  compaction: (session_id, cwd) => ({
    hook_event_name: "SessionStart",
    session_id,
    cwd,
    source: "compact",
  }),
});

const HOSTS: HostEvents[] = [
  claudeStyle("claude-code", "claude"),
  claudeStyle("codex", "codex"),
  {
    harness: "cursor-cli",
    entry: { sessionStart: "cursor-sessionstart-hook.js", prompt: "cursor-hook.js" },
    start: (conversation_id, cwd) => ({
      hook_event_name: "sessionStart",
      conversation_id,
      workspace_roots: [cwd],
    }),
    prompt: (conversation_id, cwd, prompt) => ({
      hook_event_name: "beforeSubmitPrompt",
      conversation_id,
      workspace_roots: [cwd],
      prompt,
    }),
    compaction: (conversation_id, cwd) => ({
      hook_event_name: "preCompact",
      conversation_id,
      workspace_roots: [cwd],
      trigger: "auto",
    }),
  },
  {
    harness: "kimi-code",
    entry: { sessionStart: "kimi-sessionstart-hook.js", prompt: "kimi-hook.js" },
    start: (session_id, cwd) => ({
      hook_event_name: "SessionStart",
      session_id,
      cwd,
      source: "startup",
    }),
    prompt: (session_id, cwd, text) => ({
      hook_event_name: "UserPromptSubmit",
      session_id,
      cwd,
      prompt: [{ type: "text", text }],
    }),
    compaction: (session_id, cwd) => ({
      hook_event_name: "PostCompact",
      session_id,
      cwd,
      trigger: "auto",
    }),
  },
];

let server: Server;
let recalls = 0;
let root: string;
let env: NodeJS.ProcessEnv;

beforeAll(async () => {
  // Answers recall with the memory; everything else is an empty bank.
  server = createServer((req, res) => {
    req.resume();
    req.on("end", () => {
      if (req.method === "POST" && req.url?.endsWith("/memories/recall")) {
        recalls++;
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ results: [{ text: MEMORY }] }));
        return;
      }
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(req.url?.includes("/pages") ? { pages: [] } : {}));
    });
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  const port = (server.address() as AddressInfo).port;

  root = mkdtempSync(join(tmpdir(), "hindsight-compaction-e2e-"));
  const home = join(root, "home");
  mkdirSync(join(home, ".hindsight"), { recursive: true });
  writeFileSync(
    join(home, ".hindsight", "coding-agent.json"),
    JSON.stringify({
      serverMode: "self-hosted",
      apiUrl: `http://127.0.0.1:${port}`,
      autoInject: "recall",
      autoSeed: false,
      autoUpdate: false,
      retainSessions: false,
    })
  );
  const tmp = join(root, "tmp");
  mkdirSync(tmp);
  env = {
    ...process.env,
    HOME: home,
    TMPDIR: tmp,
    HINDSIGHT_CONFIG: join(home, ".hindsight", "coding-agent.json"),
    HINDSIGHT_DIAG_FILE: join(root, "diag.jsonl"),
  };
  delete env.HINDSIGHT_DISABLE_HOOKS;
  delete env.HINDSIGHT_API_URL;
  delete env.HINDSIGHT_API_TOKEN;
});

afterAll(() => {
  server?.close();
  if (root) rmSync(root, { recursive: true, force: true });
});

/** Run one hook process. Async on purpose: a blocking spawn would stall this process's event loop,
 *  and with it the stub server the hook is waiting on. */
async function runEntry(
  entry: string,
  event: Record<string, unknown>,
  cwd: string
): Promise<string> {
  const child = spawn(process.execPath, [join(dist, entry)], { cwd, env });
  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (d) => (stdout += d));
  child.stderr.on("data", (d) => (stderr += d));
  child.stdin.end(JSON.stringify(event));
  const code = await new Promise<number | null>((r) => child.on("close", r));
  const diagFile = join(root, "diag.jsonl");
  const diag = existsSync(diagFile) ? readFileSync(diagFile, "utf8") : "";
  expect(code, `${stderr}\n${diag}`).toBe(0);
  return stdout;
}

/** A repo the bank can be named after. */
function makeRepo(name: string): string {
  const dir = join(root, name);
  mkdirSync(dir);
  execFileSync("git", ["init", "-q"], { cwd: dir });
  execFileSync("git", ["remote", "add", "origin", `https://github.com/example/${name}.git`], {
    cwd: dir,
  });
  return dir;
}

describe("a compaction is a new session start (#6527)", () => {
  for (const host of HOSTS) {
    it(`${host.harness}: the first prompt after a compaction gets memory again`, async () => {
      const cwd = makeRepo(`repo-${host.harness}`);
      const session = `session-${host.harness}`;
      await runEntry(host.entry.sessionStart, host.start(session, cwd), cwd);

      const first = await runEntry(
        host.entry.prompt,
        host.prompt(session, cwd, "fix the retry loop"),
        cwd
      );
      expect(first).toContain(MEMORY);
      // Memory is delivered once per session: the second prompt carries none.
      const second = await runEntry(
        host.entry.prompt,
        host.prompt(session, cwd, "and the tests"),
        cwd
      );
      expect(second).not.toContain(MEMORY);

      await runEntry(host.entry.sessionStart, host.compaction(session, cwd), cwd);

      const recallsBefore = recalls;
      const afterCompaction = await runEntry(
        host.entry.prompt,
        host.prompt(session, cwd, "continue with the retry loop"),
        cwd
      );
      expect(afterCompaction).toContain(MEMORY);
      expect(recalls).toBe(recallsBefore + 1);
    });
  }
});

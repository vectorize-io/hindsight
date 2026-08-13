import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { chmodSync, existsSync, mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import { tmpdir } from "node:os";
import { delimiter, join, resolve } from "node:path";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";

let root: string;
let server: Server;
let apiUrl: string;
let requests: string[];

beforeAll(async () => {
  root = mkdtempSync(join(tmpdir(), "hs-stop-hook-config-"));
  requests = [];
  server = createServer((request, response) => {
    const chunks: Buffer[] = [];
    request.on("data", (chunk: Buffer) => chunks.push(chunk));
    request.on("end", () => {
      requests.push(`${request.method} ${request.url} ${Buffer.concat(chunks).toString("utf8")}`);
      const payload = request.url?.includes("/operations/")
        ? { status: "completed" }
        : request.url?.endsWith("/memories")
          ? { operation_id: "test-op" }
          : request.url === "/version"
            ? { api_version: "0.9.0" }
            : {};
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify(payload));
    });
  });
  await new Promise<void>((resolveListen) => server.listen(0, "127.0.0.1", resolveListen));
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("test server did not bind TCP");
  apiUrl = `http://127.0.0.1:${address.port}`;
});

afterAll(async () => {
  await new Promise<void>((resolveClose, reject) =>
    server.close((error) => (error ? reject(error) : resolveClose()))
  );
  rmSync(root, { recursive: true, force: true });
});

beforeEach(() => {
  requests.length = 0;
});

type Harness = "codex" | "cursor-cli";

async function unusedPort(): Promise<number> {
  const probe = createServer();
  await new Promise<void>((resolveListen) => probe.listen(0, "127.0.0.1", resolveListen));
  const address = probe.address();
  if (!address || typeof address === "string") throw new Error("port probe did not bind TCP");
  await new Promise<void>((resolveClose, reject) =>
    probe.close((error) => (error ? reject(error) : resolveClose()))
  );
  return address.port;
}

async function runStopHook(harness: Harness, config: Record<string, unknown>): Promise<void> {
  const sessionId = randomUUID();
  const configPath = join(root, `${harness}-${sessionId}-config.json`);
  const transcriptPath = join(root, `${harness}-${sessionId}-transcript.jsonl`);
  writeFileSync(configPath, JSON.stringify({ serverMode: "self-hosted", apiUrl, ...config }));

  const event =
    harness === "codex"
      ? { session_id: sessionId, transcript_path: transcriptPath, cwd: root }
      : { conversation_id: sessionId, transcript_path: transcriptPath, workspace_root: root };
  const transcript =
    harness === "codex"
      ? {
          type: "response_item",
          payload: {
            type: "message",
            role: "user",
            content: [{ type: "input_text", text: "remember this" }],
          },
        }
      : { role: "user", content: "remember this" };
  writeFileSync(transcriptPath, `${JSON.stringify(transcript)}\n`);

  const entry = resolve("src", harness === "codex" ? "codex-stop-hook.ts" : "cursor-stop-hook.ts");
  const viteNode = resolve("node_modules", "vite-node", "vite-node.mjs");
  const child = spawn(process.execPath, [viteNode, entry], {
    env: {
      ...process.env,
      HINDSIGHT_CONFIG: configPath,
      HINDSIGHT_DIAG_FILE: join(root, `${harness}-diagnostics.jsonl`),
    },
    stdio: ["pipe", "pipe", "pipe"],
  });
  let stderr = "";
  child.stderr.on("data", (chunk: Buffer) => (stderr += chunk.toString()));
  child.stdin.end(JSON.stringify(event));

  await new Promise<void>((resolveExit, reject) => {
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error(`${harness} Stop hook timed out: ${stderr}`));
    }, 10_000);
    child.on("exit", (code, signal) => {
      clearTimeout(timer);
      if (signal) reject(new Error(`${harness} Stop hook exited via ${signal}: ${stderr}`));
      else if (code !== 0) reject(new Error(`${harness} Stop hook exited ${code}: ${stderr}`));
      else resolveExit();
    });
  });
}

describe("Stop-hook retainSessions", () => {
  it.each<Harness>(["codex", "cursor-cli"])(
    "%s makes no API request when globally disabled",
    async (harness) => {
      await runStopHook(harness, { bankId: "test-bank", retainSessions: false });
      expect(requests).toEqual([]);
    }
  );

  it("retains by default", async () => {
    await runStopHook("codex", { bankId: "test-bank" });
    expect(requests.some((request) => request.includes("/memories"))).toBe(true);
  });

  it("returns before daemon startup when disabled", async () => {
    const bin = join(root, "sentinel-bin");
    const sentinel = join(root, "daemon-started");
    mkdirSync(bin, { recursive: true });
    for (const command of ["uvx", "cargo"]) {
      const executable = join(bin, command);
      writeFileSync(executable, "#!/bin/sh\nexit 0\n");
      chmodSync(executable, 0o755);
    }
    const node = join(bin, "node");
    writeFileSync(
      node,
      `#!${process.execPath}\nrequire("node:fs").writeFileSync(process.env.HINDSIGHT_DAEMON_SENTINEL, "started");\n`
    );
    chmodSync(node, 0o755);

    const saved = {
      path: process.env.PATH,
      provider: process.env.HINDSIGHT_API_LLM_PROVIDER,
      sentinel: process.env.HINDSIGHT_DAEMON_SENTINEL,
    };
    process.env.PATH = `${bin}${delimiter}${saved.path ?? ""}`;
    process.env.HINDSIGHT_API_LLM_PROVIDER = "test";
    process.env.HINDSIGHT_DAEMON_SENTINEL = sentinel;
    try {
      await runStopHook("codex", {
        apiPort: await unusedPort(),
        bankId: "test-bank",
        retainSessions: false,
        serverMode: "daemon",
      });
      expect(existsSync(sentinel)).toBe(false);
    } finally {
      if (saved.path === undefined) delete process.env.PATH;
      else process.env.PATH = saved.path;
      if (saved.provider === undefined) delete process.env.HINDSIGHT_API_LLM_PROVIDER;
      else process.env.HINDSIGHT_API_LLM_PROVIDER = saved.provider;
      if (saved.sentinel === undefined) delete process.env.HINDSIGHT_DAEMON_SENTINEL;
      else process.env.HINDSIGHT_DAEMON_SENTINEL = saved.sentinel;
    }
  });

  it("honors a harness-specific opt-out", async () => {
    await runStopHook("codex", {
      bankId: "test-bank",
      harnesses: { codex: { retainSessions: false } },
    });
    expect(requests).toEqual([]);
  });

  it("honors a bank-specific opt-out", async () => {
    await runStopHook("codex", {
      bankId: "test-bank",
      banks: { "test-bank": { retainSessions: false } },
    });
    expect(requests).toEqual([]);
  });
});

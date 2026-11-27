/**
 * #5089 end to end: the BUILT Claude Code prompt hook, run as its own process the way Claude Code
 * runs it, reads `HINDSIGHT_INJECT_QUERY_MAX_CHARS` from its environment and sends a recall
 * question no longer than that to a real HTTP server. Run with `npm run build` first.
 */
import { execFileSync, spawn } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

const hook = join(dirname(fileURLToPath(import.meta.url)), "..", "dist", "claude-hook.js");

describe.runIf(existsSync(hook))("injectQueryMaxChars through the built Claude Code hook", () => {
  let server: Server;
  let url: string;
  const queries: string[] = [];

  beforeAll(async () => {
    // Answers recall with one observation and everything else with 404, as a server with no pages.
    server = createServer((req, res) => {
      let body = "";
      req.on("data", (c) => (body += c));
      req.on("end", () => {
        if (req.method === "POST" && req.url?.endsWith("/memories/recall")) {
          queries.push(JSON.parse(body).query);
          res.writeHead(200, { "content-type": "application/json" });
          res.end(JSON.stringify({ results: [{ text: "Uploads retry 3 times." }] }));
          return;
        }
        res.writeHead(404).end();
      });
    });
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  });

  afterAll(() => server.close());

  const runHook = async (prompt: string, maxChars?: string) => {
    const home = mkdtempSync(join(tmpdir(), "hs-inject-query-"));
    const repo = join(home, "repo");
    execFileSync("git", ["init", "-q", repo]);
    try {
      const child = spawn(process.execPath, [hook], {
        env: {
          ...process.env,
          HOME: home,
          HINDSIGHT_CONFIG: join(home, "none.json"),
          HINDSIGHT_SERVER_MODE: "self-hosted",
          HINDSIGHT_API_URL: url,
          HINDSIGHT_API_TOKEN: "test-key",
          HINDSIGHT_AUTO_INJECT: "recall",
          HINDSIGHT_AUTO_SEED: "false",
          HINDSIGHT_AUTO_UPDATE: "false",
          HINDSIGHT_INJECT_QUERY_MAX_CHARS: maxChars ?? "",
        },
      });
      let out = "";
      child.stdout.on("data", (c) => (out += c));
      child.stdin.end(
        JSON.stringify({
          hook_event_name: "UserPromptSubmit",
          session_id: `s-${Date.now()}-${Math.random()}`,
          cwd: repo,
          prompt,
        })
      );
      await new Promise((r) => child.on("close", r));
      return out;
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  };

  it("cuts the recall question to the configured characters", async () => {
    const prompt = "Уточни, как ретраятся загрузки. ".repeat(40);
    const out = await runHook(prompt, "40");
    expect(queries.at(-1)).toBe(Array.from(prompt).slice(0, 40).join(""));
    // The recalled memory still reaches the model.
    expect(out).toContain("Uploads retry 3 times.");
  });

  it("leaves the question as it was when unset", async () => {
    const prompt = "how do uploads retry? ".repeat(100);
    await runHook(prompt);
    // Today's bound: ~500 tokens, costed at half a token per ASCII character.
    expect(queries.at(-1)).toBe(prompt.slice(0, 1000));
  });
});

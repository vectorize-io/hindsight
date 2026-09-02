/**
 * End-to-end test against a REAL HTTP server. Runs the CLI's `runCli` over a
 * temp vault pointed at a throwaway node:http server, so the whole stack —
 * FsVault → SyncEngine → HindsightClient → fetch → sockets → server — runs
 * for real, with nothing mocked. This is the closest thing to hitting a live
 * Hindsight server that can run in CI.
 */

import { rm, mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { runCli } from "../../src/node/cli";

interface Captured {
  method: string;
  url: string;
  auth: string | undefined;
  body: string;
}

let root: string;
let server: Server;
let received: Captured[];
let baseUrl: string;

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), "hs-e2e-"));
  received = [];
  server = createServer((req, res) => {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
    });
    req.on("end", () => {
      received.push({
        method: req.method ?? "",
        url: req.url ?? "",
        auth: req.headers.authorization,
        body,
      });
      res.writeHead(200, { "content-type": "application/json" });
      res.end("{}");
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

afterEach(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
  await rm(root, { recursive: true, force: true });
});

describe("runCli end-to-end over real HTTP", () => {
  it("POSTs a retain with the bearer token and correct document id, then DELETEs on prune", async () => {
    await mkdir(join(root, "Work"), { recursive: true });
    await writeFile(join(root, "Work", "note.md"), "# Note\nships in Q3");
    const indexPath = join(root, "idx.json");
    const args = [
      "--vault",
      root,
      "--bank",
      "team",
      "--api-url",
      baseUrl,
      "--api-token",
      "hsk_secret",
      "--vault-name",
      "TeamVault",
      "--index",
      indexPath,
    ];

    // First run: the note is ingested.
    expect(
      await runCli(
        args,
        () => {},
        () => {}
      )
    ).toBe(0);

    const retain = received.find((r) => r.url.endsWith("/memories"));
    expect(retain).toBeDefined();
    expect(retain!.method).toBe("POST");
    expect(retain!.url).toBe("/v1/default/banks/team/memories");
    expect(retain!.auth).toBe("Bearer hsk_secret");
    const item = JSON.parse(retain!.body).items[0];
    expect(item.document_id).toBe("Work/note.md");
    expect(item.content).toContain("ships in Q3");
    expect(item.tags).toEqual(expect.arrayContaining(["vault:TeamVault", "folder:Work"]));

    // Delete the note and re-run: the document is pruned via a real DELETE.
    received.length = 0;
    await rm(join(root, "Work", "note.md"));
    expect(
      await runCli(
        args,
        () => {},
        () => {}
      )
    ).toBe(0);

    const del = received.find((r) => r.method === "DELETE");
    expect(del).toBeDefined();
    expect(del!.url).toBe("/v1/default/banks/team/documents/Work/note.md");
    expect(del!.auth).toBe("Bearer hsk_secret");
  }, 20000);

  it("returns exit code 1 and reports when the server errors", async () => {
    await writeFile(join(root, "x.md"), "# X\nbody");
    // Point at a closed port so fetch fails at the socket level.
    const errs: string[] = [];
    const code = await runCli(
      [
        "--vault",
        root,
        "--bank",
        "b",
        "--api-url",
        "http://127.0.0.1:1",
        "--index",
        join(root, "i.json"),
      ],
      () => {},
      (m) => errs.push(m)
    );
    expect(code).toBe(1);
    expect(errs.join("\n")).toMatch(/hindsight-obsidian-sync:/);
  }, 20000);
});

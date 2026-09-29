import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, readFileSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { main } from "../src/cli.js";
import { BEGIN_MARKER } from "../src/rulesFile.js";
import { SERVER_NAME } from "../src/zedSettings.js";

function tmp() {
  return mkdtempSync(join(tmpdir(), "hz-cli-"));
}

/** Run main() while capturing stdout. */
function run(argv) {
  const lines = [];
  const orig = console.log;
  console.log = (...args) => lines.push(args.join(" "));
  let code;
  try {
    code = main(argv);
  } finally {
    console.log = orig;
  }
  return { code, out: lines.join("\n") };
}

function paths(dir) {
  return [
    "--settings-path",
    join(dir, "settings.json"),
    "--rules-path",
    join(dir, "AGENTS.md"),
    "--config-path",
    join(dir, "zed.json"),
  ];
}

test("init writes settings, rule, and scaffolds config", () => {
  const dir = tmp();
  try {
    const { code } = run(["init", "--api-token", "tok", "--bank-id", "proj", ...paths(dir)]);
    assert.equal(code, 0);

    const settings = JSON.parse(readFileSync(join(dir, "settings.json"), "utf-8"));
    const server = settings.context_servers[SERVER_NAME];
    assert.equal(server.command, "npx");
    assert.ok(server.args.includes("mcp-remote"));
    assert.ok(server.args.some((a) => a.includes("/mcp/proj/")));
    assert.ok(server.args.includes("Authorization: Bearer tok"));

    assert.ok(readFileSync(join(dir, "AGENTS.md"), "utf-8").includes(BEGIN_MARKER));

    const cfg = JSON.parse(readFileSync(join(dir, "zed.json"), "utf-8"));
    assert.equal(cfg.bankId, "proj");
    assert.equal(cfg.hindsightApiToken, "tok");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("init --print-only writes nothing", () => {
  const dir = tmp();
  try {
    const { code, out } = run(["init", "--print-only", "--bank-id", "zed", ...paths(dir)]);
    assert.equal(code, 0);
    assert.ok(out.includes("mcp-remote"));
    assert.ok(out.includes("recall"));
    assert.equal(existsSync(join(dir, "settings.json")), false);
    assert.equal(existsSync(join(dir, "AGENTS.md")), false);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("status reflects installed state", () => {
  const dir = tmp();
  try {
    let res = run(["status", ...paths(dir)]);
    assert.ok(res.out.includes("not installed"));
    run(["init", "--api-token", "tok", ...paths(dir)]);
    res = run(["status", ...paths(dir)]);
    assert.ok(res.out.includes("MCP server"));
    assert.ok(!res.out.includes("not installed"));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("uninstall removes the server and the rule", () => {
  const dir = tmp();
  try {
    run(["init", "--api-token", "tok", ...paths(dir)]);
    const { code } = run(["uninstall", ...paths(dir)]);
    assert.equal(code, 0);
    const settings = JSON.parse(readFileSync(join(dir, "settings.json"), "utf-8"));
    assert.ok(!("context_servers" in settings));
    assert.equal(existsSync(join(dir, "AGENTS.md")), false);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("init on a JSONC settings file prints the manual snippet, leaves it untouched", () => {
  const dir = tmp();
  try {
    const settingsPath = join(dir, "settings.json");
    writeFileSync(settingsPath, '{\n  // keep me\n  "theme": "one"\n}\n');
    const { out } = run(["init", "--api-token", "tok", ...paths(dir)]);
    assert.ok(out.includes("has comments"));
    assert.ok(out.includes("mcp-remote"));
    assert.ok(readFileSync(settingsPath, "utf-8").includes("// keep me"));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("no command prints help and returns non-zero", () => {
  const { code, out } = run([]);
  assert.equal(code, 1);
  assert.ok(out.includes("Usage"));
});

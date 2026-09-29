import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, readFileSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  BEGIN_MARKER,
  END_MARKER,
  RULE_TEXT,
  writeRule,
  clearRule,
  isInstalled,
} from "../src/rulesFile.js";

function tmpPath(name = "AGENTS.md") {
  return join(mkdtempSync(join(tmpdir(), "hz-rules-")), name);
}

test("write creates the file with the fenced block", () => {
  const p = tmpPath();
  try {
    writeRule(p);
    const text = readFileSync(p, "utf-8");
    assert.ok(text.includes(BEGIN_MARKER));
    assert.ok(text.includes(END_MARKER));
    assert.ok(text.includes("recall"));
    assert.ok(isInstalled(p));
  } finally {
    rmSync(join(p, ".."), { recursive: true, force: true });
  }
});

test("write preserves user content and puts our block on top", () => {
  const p = tmpPath();
  try {
    writeFileSync(p, "# My rules\n\nBe concise.\n");
    writeRule(p);
    const text = readFileSync(p, "utf-8");
    assert.ok(text.startsWith(BEGIN_MARKER));
    assert.ok(text.includes("# My rules"));
    assert.ok(text.includes("Be concise."));
    // Exactly one block after a repeated write.
    writeRule(p);
    const text2 = readFileSync(p, "utf-8");
    assert.equal(text2.split(BEGIN_MARKER).length - 1, 1);
    assert.ok(text2.includes("# My rules"));
  } finally {
    rmSync(join(p, ".."), { recursive: true, force: true });
  }
});

test("clear removes our block but keeps user content", () => {
  const p = tmpPath();
  try {
    writeFileSync(p, "# My rules\n\nBe concise.\n");
    writeRule(p);
    clearRule(p);
    const text = readFileSync(p, "utf-8");
    assert.ok(!text.includes(BEGIN_MARKER));
    assert.ok(text.includes("# My rules"));
    assert.ok(!isInstalled(p));
  } finally {
    rmSync(join(p, ".."), { recursive: true, force: true });
  }
});

test("clear deletes the file if it held only our block", () => {
  const p = tmpPath();
  try {
    writeRule(p);
    clearRule(p);
    assert.equal(existsSync(p), false);
  } finally {
    rmSync(join(p, ".."), { recursive: true, force: true });
  }
});

test("custom rule text round-trips through the block", () => {
  const p = tmpPath();
  try {
    writeRule(p, "CUSTOM RULE");
    const text = readFileSync(p, "utf-8");
    assert.ok(text.includes("CUSTOM RULE"));
    assert.ok(!text.includes(RULE_TEXT));
  } finally {
    rmSync(join(p, ".."), { recursive: true, force: true });
  }
});

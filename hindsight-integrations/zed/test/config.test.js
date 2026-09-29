import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadConfig, DEFAULT_HINDSIGHT_API_URL, DEFAULT_BANK_ID } from "../src/config.js";

function tmp() {
  return mkdtempSync(join(tmpdir(), "hz-config-"));
}

test("defaults when no file and empty env", () => {
  const dir = tmp();
  try {
    const cfg = loadConfig({ configFile: join(dir, "missing.json"), env: {} });
    assert.equal(cfg.hindsightApiUrl, DEFAULT_HINDSIGHT_API_URL);
    assert.equal(cfg.hindsightApiToken, null);
    assert.equal(cfg.bankId, DEFAULT_BANK_ID);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("reads values from the config file", () => {
  const dir = tmp();
  try {
    const file = join(dir, "zed.json");
    writeFileSync(
      file,
      JSON.stringify({
        hindsightApiUrl: "http://localhost:8888",
        hindsightApiToken: "tok",
        bankId: "proj",
      })
    );
    const cfg = loadConfig({ configFile: file, env: {} });
    assert.equal(cfg.hindsightApiUrl, "http://localhost:8888");
    assert.equal(cfg.hindsightApiToken, "tok");
    assert.equal(cfg.bankId, "proj");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("environment variables win over the file", () => {
  const dir = tmp();
  try {
    const file = join(dir, "zed.json");
    writeFileSync(file, JSON.stringify({ hindsightApiUrl: "http://file", bankId: "fromfile" }));
    const cfg = loadConfig({
      configFile: file,
      env: {
        HINDSIGHT_API_URL: "http://env",
        HINDSIGHT_API_TOKEN: "envtok",
        HINDSIGHT_ZED_BANK_ID: "fromenv",
      },
    });
    assert.equal(cfg.hindsightApiUrl, "http://env");
    assert.equal(cfg.hindsightApiToken, "envtok");
    assert.equal(cfg.bankId, "fromenv");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("malformed config file falls back to defaults", () => {
  const dir = tmp();
  try {
    const file = join(dir, "zed.json");
    writeFileSync(file, "{ not json");
    const cfg = loadConfig({ configFile: file, env: {} });
    assert.equal(cfg.hindsightApiUrl, DEFAULT_HINDSIGHT_API_URL);
    assert.equal(cfg.bankId, DEFAULT_BANK_ID);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

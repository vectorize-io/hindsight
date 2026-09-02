import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { loadConfig, applyBankConfig, readEnvConfig, resolveConfig } from "./config";

let root: string;
let globalCfg: string;

function writeJson(path: string, value: unknown): void {
  mkdirSync(join(path, ".."), { recursive: true });
  writeFileSync(path, JSON.stringify(value));
}

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hs-cfg-"));
  globalCfg = join(root, "global.json");
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

describe("loadConfig layering", () => {
  it("missing files yield defaults", () => {
    const cfg = loadConfig({ path: join(root, "nope.json") });
    expect(cfg.apiUrl).toBe("https://api.hindsight.vectorize.io");
    expect(cfg.bankId).toBeUndefined();
    expect(cfg.disabled).toBe(false);
  });

  it("malformed global file falls back to defaults with a warning", () => {
    writeFileSync(globalCfg, "{not json");
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    const cfg = loadConfig({ path: globalCfg });
    expect(cfg.apiUrl).toBe("https://api.hindsight.vectorize.io");
    expect(err).toHaveBeenCalledOnce();
    err.mockRestore();
  });

  it("applies the requesting harness's section over the base", () => {
    writeJson(globalCfg, {
      apiUrl: "http://x:1",
      bankId: "shared",
      harnesses: {
        "claude-code": { bankId: "claude-bank" },
        opencode: { disabled: true },
      },
    });
    expect(loadConfig({ path: globalCfg, harness: "claude-code" }).bankId).toBe("claude-bank");
    expect(loadConfig({ path: globalCfg, harness: "claude-code" }).apiUrl).toBe("http://x:1");
    expect(loadConfig({ path: globalCfg, harness: "opencode" }).disabled).toBe(true);
    expect(loadConfig({ path: globalCfg, harness: "opencode" }).bankId).toBe("shared");
    expect(loadConfig({ path: globalCfg }).bankId).toBe("shared"); // no harness: base only
  });

  it("resolves harness to the ASKING harness when the file sets none — not the opencode default (#3247)", () => {
    writeJson(globalCfg, { bankId: "shared" }); // no explicit `harness:` field
    expect(loadConfig({ path: globalCfg, harness: "claude-code" }).harness).toBe("claude-code");
    expect(loadConfig({ path: globalCfg, harness: "kilo" }).harness).toBe("kilo");
  });

  it("an explicit harness field in the config file still wins over the asking harness", () => {
    writeJson(globalCfg, { harness: "opencode" });
    expect(loadConfig({ path: globalCfg, harness: "claude-code" }).harness).toBe("opencode");
  });

  it("legacy string signature still works as the global path", () => {
    writeJson(globalCfg, { bankId: "legacy" });
    expect(loadConfig(globalCfg).bankId).toBe("legacy");
  });

  it("pageRefreshEveryTurns defaults to 10", () => {
    expect(loadConfig({ harness: "claude-code" }).pageRefreshEveryTurns).toBe(10);
  });

  it("pageRefreshEveryTurns override wins over the default", () => {
    writeJson(globalCfg, { pageRefreshEveryTurns: 25 });
    expect(loadConfig({ path: globalCfg }).pageRefreshEveryTurns).toBe(25);
  });
});

describe("maxParallelRetains", () => {
  it("defaults to 10 when unset", () => {
    expect(loadConfig({ harness: "claude-code" }).maxParallelRetains).toBe(10);
  });

  it("config file value wins over the default", () => {
    writeJson(globalCfg, { maxParallelRetains: 3 });
    expect(loadConfig({ path: globalCfg }).maxParallelRetains).toBe(3);
  });

  const ENV = { ...process.env };
  afterEach(() => {
    process.env = { ...ENV };
  });

  it("reads HINDSIGHT_MAX_PARALLEL_RETAINS as a number", () => {
    writeJson(globalCfg, {});
    process.env.HINDSIGHT_MAX_PARALLEL_RETAINS = "6";
    expect(loadConfig({ path: globalCfg }).maxParallelRetains).toBe(6);
  });

  it("ignores a malformed env value and falls back to the default", () => {
    writeJson(globalCfg, {});
    vi.spyOn(console, "error").mockImplementation(() => {});
    process.env.HINDSIGHT_MAX_PARALLEL_RETAINS = "lots";
    expect(loadConfig({ path: globalCfg }).maxParallelRetains).toBe(10);
  });
});

/**
 * #3590: the hindsight_reflect tool aborted at a hardcoded 120s. The tool's window is now its own
 * knob, defaulting ABOVE the server's 300s reflect wall timeout — and it inherits an explicitly
 * raised reflectTimeoutMs, because that is the field users reaching for a longer reflect set.
 */
describe("reflectToolTimeoutMs / reflectBudget", () => {
  it("defaults above the server's reflect wall timeout, leaving the hook window untouched", () => {
    const cfg = resolveConfig({});
    expect(cfg.reflectToolTimeoutMs).toBe(330000);
    expect(cfg.reflectTimeoutMs).toBe(120000);
    expect(cfg.reflectBudget).toBe("high");
  });

  it("inherits an explicitly raised reflectTimeoutMs", () => {
    expect(resolveConfig({ reflectTimeoutMs: 660000 }).reflectToolTimeoutMs).toBe(660000);
  });

  it("is never LOWERED by a short reflectTimeoutMs — that bounds the hook, not the tool", () => {
    const cfg = resolveConfig({ reflectTimeoutMs: 5000 });
    expect(cfg.reflectTimeoutMs).toBe(5000);
    expect(cfg.reflectToolTimeoutMs).toBe(330000);
  });

  it("an explicit reflectToolTimeoutMs wins over both", () => {
    expect(
      resolveConfig({ reflectTimeoutMs: 660000, reflectToolTimeoutMs: 90000 }).reflectToolTimeoutMs
    ).toBe(90000);
  });

  const ENV = { ...process.env };
  afterEach(() => {
    process.env = { ...ENV };
  });

  it("reads the env fallbacks", () => {
    writeJson(globalCfg, {});
    process.env.HINDSIGHT_REFLECT_TOOL_TIMEOUT_MS = "600000";
    process.env.HINDSIGHT_REFLECT_BUDGET = "mid";
    const cfg = loadConfig({ path: globalCfg });
    expect(cfg.reflectToolTimeoutMs).toBe(600000);
    expect(cfg.reflectBudget).toBe("mid");
  });

  it("falls back to high on an unknown budget rather than sending it to the API", () => {
    // The API rejects an unknown budget outright, so a typo here would fail every reflect call.
    expect(resolveConfig({ reflectBudget: "highest" as never }).reflectBudget).toBe("high");
  });
});

// A project-local .hindsight/coding-agent.json comes from the (untrusted) opened repo. It must not be
// able to redirect the API endpoint/token or the global bank map — otherwise a malicious repo could
// exfiltrate the user's token + prompts to its own server just by being opened.
describe("loadConfig — untrusted project-local layer is sanitized (security)", () => {
  it("the user-global config CAN still set apiUrl/apiToken (only the project layer is restricted)", () => {
    writeJson(globalCfg, { apiUrl: "https://real.example", apiToken: "REAL-TOKEN" });
    const cfg = loadConfig({ path: globalCfg });
    expect(cfg.apiUrl).toBe("https://real.example");
    expect(cfg.apiToken).toBe("REAL-TOKEN");
  });
});

describe("manageBankConfig (#3927)", () => {
  it("defaults to true — a bank the plugin creates still gets the shape its ingestion needs", () => {
    expect(resolveConfig({}).manageBankConfig).toBe(true);
  });

  it("is settable per bank, which is where a shared global bank needs it", () => {
    // The #3927 setup: ONE bank for coding and personal-assistant work alike. The plugin keeps
    // managing every other bank; this one is the user's to shape.
    const cfg = resolveConfig({
      banks: { "my-global-bank": { manageBankConfig: false } },
    });
    expect(applyBankConfig(cfg, "my-global-bank").cfg.manageBankConfig).toBe(false);
    expect(applyBankConfig(cfg, "coding-agent::repo").cfg.manageBankConfig).toBe(true);
  });
});

describe("banks.<bankId> overrides (per-repo opt-in/out, applied AFTER bank resolution)", () => {
  it("overrides behavioral fields for the matching bank only; others untouched", () => {
    const cfg = resolveConfig({
      gitIngest: "message",
      banks: {
        "coding-agent::secret": { disabled: true },
        "coding-agent::mono": { gitIngest: "full", retainSessions: false },
      },
    });
    expect(applyBankConfig(cfg, "coding-agent::secret").cfg.disabled).toBe(true);
    const mono = applyBankConfig(cfg, "coding-agent::mono").cfg;
    expect(mono.gitIngest).toBe("full");
    expect(mono.retainSessions).toBe(false);
    expect(mono.disabled).toBe(false);
    const other = applyBankConfig(cfg, "coding-agent::other").cfg;
    expect(other.disabled).toBe(false);
    expect(other.gitIngest).toBe("message");
  });

  it("ignores bank-resolution fields inside a bank section (cannot re-route memory)", () => {
    const cfg = resolveConfig({
      banks: {
        b1: { bankId: "evil", mapPathToBank: { "/x": "evil" }, disabled: true } as never,
      },
    });
    const out = applyBankConfig(cfg, "b1").cfg;
    expect(out.disabled).toBe(true);
    expect(out.bankId).toBe(cfg.bankId); // untouched
    expect(out.mapPathToBank).toBe(cfg.mapPathToBank);
  });

  it("an override only changes the fields it names (defaults don't reset the rest)", () => {
    const cfg = resolveConfig({ retainSessions: false, banks: { b: { gitIngest: "none" } } });
    const out = applyBankConfig(cfg, "b").cfg;
    expect(out.gitIngest).toBe("none");
    expect(out.retainSessions).toBe(false); // kept from the base, not reset to default true
  });
});

describe("banks.<id>.bank — rename inside the banks tree", () => {
  it("renames the destination bank; other fields still apply; unmatched ids unchanged", () => {
    const cfg = resolveConfig({
      banks: { "coding-agent::old": { bank: "team::shared", gitIngest: "full" } },
    });
    const r = applyBankConfig(cfg, "coding-agent::old");
    expect(r.bankId).toBe("team::shared");
    expect(r.cfg.gitIngest).toBe("full");
    expect(applyBankConfig(cfg, "coding-agent::other").bankId).toBe("coding-agent::other");
  });

  it("single hop: the rename target's own section is NOT consulted (no chaining)", () => {
    const cfg = resolveConfig({
      banks: {
        a: { bank: "b" },
        b: { bank: "c", disabled: true }, // must not apply to the a->b hop
      },
    });
    const r = applyBankConfig(cfg, "a");
    expect(r.bankId).toBe("b"); // not "c"
    expect(r.cfg.disabled).toBe(false);
  });
});

/**
 * Env vars are a FALLBACK beneath the config file — for containers, CI and secret managers that
 * inject a token rather than writing a credential to disk. The file must keep winning wherever it
 * sets a value, or an existing setup would change behaviour just by having env present.
 */
describe("environment fallback", () => {
  const ENV = { ...process.env };
  afterEach(() => {
    process.env = { ...ENV };
  });

  it("supplies apiUrl and apiToken when the file omits them", () => {
    writeJson(globalCfg, { bankId: "b" }); // no apiUrl/apiToken
    process.env.HINDSIGHT_API_URL = "http://localhost:8888";
    process.env.HINDSIGHT_API_TOKEN = "tok-from-env";
    const cfg = loadConfig({ path: globalCfg });
    expect(cfg.apiUrl).toBe("http://localhost:8888");
    expect(cfg.apiToken).toBe("tok-from-env");
  });

  it("the FILE wins over env — env is a fallback, not an override", () => {
    writeJson(globalCfg, { apiUrl: "https://from-file", apiToken: "tok-from-file" });
    process.env.HINDSIGHT_API_URL = "https://from-env";
    process.env.HINDSIGHT_API_TOKEN = "tok-from-env";
    const cfg = loadConfig({ path: globalCfg });
    expect(cfg.apiUrl).toBe("https://from-file");
    expect(cfg.apiToken).toBe("tok-from-file");
  });

  it("parses booleans and numbers rather than passing strings through", () => {
    writeJson(globalCfg, {});
    process.env.HINDSIGHT_AUTO_REFLECT = "false";
    process.env.HINDSIGHT_MANAGE_BANK_CONFIG = "false";
    process.env.HINDSIGHT_DISABLED = "1";
    process.env.HINDSIGHT_SEED_LIMIT = "5";
    const cfg = loadConfig({ path: globalCfg });
    expect(cfg.autoReflect).toBe(false);
    expect(cfg.manageBankConfig).toBe(false);
    expect(cfg.disabled).toBe(true);
    expect(cfg.seedLimit).toBe(5);
  });

  it("ignores a malformed number instead of resolving it to NaN", () => {
    writeJson(globalCfg, {});
    vi.spyOn(console, "error").mockImplementation(() => {});
    process.env.HINDSIGHT_REFLECT_TIMEOUT_MS = "soon";
    // NaN here would silently break the reflect timeout in a way that is very hard to trace.
    expect(loadConfig({ path: globalCfg }).reflectTimeoutMs).toBe(120000);
  });

  it("an empty env var does not mask the file or the default", () => {
    writeJson(globalCfg, { apiUrl: "https://from-file" });
    process.env.HINDSIGHT_API_URL = "";
    process.env.HINDSIGHT_SURVEY_MODEL = "   ";
    const cfg = loadConfig({ path: globalCfg });
    expect(cfg.apiUrl).toBe("https://from-file");
    expect(cfg.surveyModel).toBe("haiku");
  });
});

describe("retainTags / retainMetadata", () => {
  it("default to empty, so a retain is unchanged unless configured", () => {
    const cfg = resolveConfig({});
    expect(cfg.retainTags).toEqual([]);
    expect(cfg.retainMetadata).toEqual({});
  });

  it("carries templates through verbatim — resolution happens per retain", () => {
    const cfg = resolveConfig({
      retainTags: ["project:{gitProject}"],
      retainMetadata: { repo: "{gitProject}" },
    });
    expect(cfg.retainTags).toEqual(["project:{gitProject}"]);
    expect(cfg.retainMetadata).toEqual({ repo: "{gitProject}" });
  });

  it("ignores non-string entries rather than failing the whole retain", () => {
    // A config typo (a number, a nested object) would otherwise reach the API as a tag.
    const cfg = resolveConfig({
      retainTags: ["ok", 42, null, "  "] as unknown as string[],
      retainMetadata: { good: "x", bad: { nested: true } } as unknown as Record<string, string>,
    });
    expect(cfg.retainTags).toEqual(["ok"]);
    expect(cfg.retainMetadata).toEqual({ good: "x" });
  });
});

describe("HINDSIGHT_RETAIN_TAGS", () => {
  it("reads a comma-separated list — the env form of retainTags (#2896)", () => {
    expect(
      readEnvConfig({ HINDSIGHT_RETAIN_TAGS: "project:{gitProject},env:work" }).retainTags
    ).toEqual(["project:{gitProject}", "env:work"]);
  });

  it("trims entries and drops empties, so a trailing comma is not an empty tag", () => {
    expect(readEnvConfig({ HINDSIGHT_RETAIN_TAGS: " a , ,b, " }).retainTags).toEqual(["a", "b"]);
  });

  it("is absent when unset or empty, leaving the file value alone", () => {
    expect(readEnvConfig({}).retainTags).toBeUndefined();
    expect(readEnvConfig({ HINDSIGHT_RETAIN_TAGS: "" }).retainTags).toBeUndefined();
    expect(readEnvConfig({ HINDSIGHT_RETAIN_TAGS: " , " }).retainTags).toBeUndefined();
  });

  it("has no retainMetadata counterpart — map-valued settings stay file-only", () => {
    expect(readEnvConfig({ HINDSIGHT_RETAIN_METADATA: "repo=x" }).retainMetadata).toBeUndefined();
  });
});

describe("observationScopes", () => {
  it("defaults to one global scope per bank, so two agents on one repo share its beliefs (#3564)", () => {
    expect(loadConfig({ path: join(root, "nope.json") }).observationScopes).toBe("shared");
  });

  it("takes any of the server's scalar modes verbatim", () => {
    for (const mode of ["shared", "combined", "per_tag", "all_combinations"] as const) {
      writeJson(globalCfg, { observationScopes: mode });
      expect(loadConfig({ path: globalCfg }).observationScopes).toBe(mode);
    }
  });

  it("takes per_source, the one scoping an explicit list cannot express", () => {
    writeJson(globalCfg, { observationScopes: "per_source" });
    expect(loadConfig({ path: globalCfg }).observationScopes).toBe("per_source");
  });

  it("takes an explicit scope list, dropping non-string entries", () => {
    expect(
      resolveConfig({ observationScopes: [["project:demo"], ["team:eng", "x"]] }).observationScopes
    ).toEqual([["project:demo"], ["team:eng", "x"]]);
    expect(
      resolveConfig({ observationScopes: [["a", 7, ""], "nope"] as never }).observationScopes
    ).toEqual([["a"]]);
  });

  it("falls back to the default on an unusable value rather than sending it to the API", () => {
    // `[]` in particular: the API reads zero scopes as no spec and silently applies `combined`,
    // which is the opposite of what writing the field was meant to say.
    expect(resolveConfig({ observationScopes: [] }).observationScopes).toBe("shared");
    expect(resolveConfig({ observationScopes: "per-tag" as never }).observationScopes).toBe(
      "shared"
    );
    expect(resolveConfig({ observationScopes: 3 as never }).observationScopes).toBe("shared");
  });

  it("is overridable per bank, since whether agents should share beliefs is a per-repo call", () => {
    const cfg = resolveConfig({
      banks: { "coding-agent::mono": { observationScopes: "combined" } },
    });
    expect(applyBankConfig(cfg, "coding-agent::mono").cfg.observationScopes).toBe("combined");
    expect(applyBankConfig(cfg, "coding-agent::other").cfg.observationScopes).toBe("shared");
  });

  it("reads HINDSIGHT_OBSERVATION_SCOPES for the scalar modes; a scope LIST stays file-only", () => {
    expect(readEnvConfig({ HINDSIGHT_OBSERVATION_SCOPES: "per_tag" }).observationScopes).toBe(
      "per_tag"
    );
    expect(readEnvConfig({}).observationScopes).toBeUndefined();
  });
});

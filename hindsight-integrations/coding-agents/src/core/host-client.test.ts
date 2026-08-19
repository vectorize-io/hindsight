import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

let root: string;
let cfgPath: string;

/** `CONFIG_PATH` is resolved from the environment when core/config is first imported, so each case
 *  points HINDSIGHT_CONFIG at its own file and re-imports the module graph. */
async function loadFactory() {
  vi.resetModules();
  process.env.HINDSIGHT_CONFIG = cfgPath;
  return import("./host-client");
}

function writeConfig(value: unknown): void {
  mkdirSync(join(cfgPath, ".."), { recursive: true });
  writeFileSync(cfgPath, JSON.stringify(value));
}

const ENV = { ...process.env };

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hs-host-"));
  cfgPath = join(root, "coding-agent.json");
});

afterEach(() => {
  process.env = { ...ENV };
  rmSync(root, { recursive: true, force: true });
});

describe("resolveHostMemory", () => {
  it("forwards every client setting the hosts used to pass by hand", async () => {
    // dsh and Prime Agent each hand-built their ClientOpts and both omitted maxParallelRetains, so
    // those two hosts silently ignored the setting. One builder is what stops that recurring.
    writeConfig({ apiUrl: "http://server", apiToken: "k", maxParallelRetains: 3 });
    const { resolveHostMemory } = await loadFactory();

    const { client } = resolveHostMemory("dsh", root);
    expect(client.apiUrl).toBe("http://server");
    expect(client.apiToken).toBe("k");
    expect(client.maxParallelRetains).toBe(3);
  });

  it("enforces optInOnly for every host, not just the ones that remembered to pass a directory", async () => {
    // dsh called applyBankConfig WITHOUT the directory, so `optInOnly` was never enforced there:
    // an unapproved repo still got a bank and still had memories written for it.
    writeConfig({ optInOnly: true, optInPaths: ["/somewhere/else"] });
    const { resolveHostMemory } = await loadFactory();

    expect(resolveHostMemory("dsh", root).cfg.disabled).toBe(true);
  });

  it("does not derive a bank when memory is disabled — that path is a zero-overhead baseline", async () => {
    // Bank derivation shells out to git. `disabled` promises the same agent with NO memory work,
    // which is what makes it usable as an A/B baseline, so it must stop before that.
    writeConfig({ disabled: true });
    const { resolveHostMemory } = await loadFactory();

    const { cfg, bankId } = resolveHostMemory("dsh", root);
    expect(cfg.disabled).toBe(true);
    expect(bankId).toBe("");
  });

  it("re-resolves the token from the live config, so a rotation does not need a restart", async () => {
    writeConfig({ apiUrl: "http://server", apiToken: "old-key" });
    const { resolveHostMemory } = await loadFactory();
    const { client } = resolveHostMemory("dsh", root);
    expect(client.apiToken).toBe("old-key");

    // The operator rotates the credential while the host keeps running; the next 401 picks it up.
    writeConfig({ apiUrl: "http://server", apiToken: "new-key" });
    const calls: (string | null)[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, init: RequestInit) => {
        const auth = new Headers(init.headers).get("Authorization");
        calls.push(auth);
        return new Response(JSON.stringify(auth === "Bearer new-key" ? { ok: true } : {}), {
          status: auth === "Bearer new-key" ? 200 : 401,
          headers: { "Content-Type": "application/json" },
        });
      })
    );

    await client.req("GET", "http://server/thing");
    expect(calls).toEqual(["Bearer old-key", "Bearer new-key"]);
    vi.unstubAllGlobals();
  });

  it("honours a per-bank apiToken on re-resolution, not just on the first read", async () => {
    // `banks.<id>.apiToken` is a legitimate override (it is not stripped by BANK_OVERRIDE_EXCLUDED),
    // so a provider that re-read only the top level would hand back the wrong credential.
    const { resolveHostMemory: probe } = await loadFactory();
    writeConfig({ apiUrl: "http://server", apiToken: "global" });
    const bankId = probe("dsh", root).bankId;

    writeConfig({
      apiUrl: "http://server",
      apiToken: "global",
      banks: { [bankId]: { apiToken: "per-bank" } },
    });
    const { resolveHostMemory } = await loadFactory();
    expect(resolveHostMemory("dsh", root).client.apiToken).toBe("per-bank");
  });
});

/**
 * The #3600 shape: a capability wired per-host is a capability the next host forgets. Three
 * settings had already gone missing that way (maxParallelRetains twice, optInOnly once, the live
 * credential everywhere), so the rule is structural — a long-lived host does not build its own
 * client, and a module that does has to say why.
 */
describe("long-lived hosts build their client through the shared factory", () => {
  const SRC = fileURLToPath(new URL("..", import.meta.url));

  /** Modules that construct a client directly, and why that is correct for them. */
  const DIRECT: Record<string, string> = {
    "core/host-client.ts": "the shared factory itself",
    "core/hook.ts": "one-shot hook process — re-reads config on every invocation",
    "core/retain-hook.ts": "one-shot hook process, same",
    "core/session-start.ts": "one-shot hook process, same",
    "status.ts": "one-shot CLI resolving config from --config/--harness, not from a workspace",
    "deepen.ts": "one-shot CLI, same",
  };

  function sourceFiles(dir: string, prefix = ""): string[] {
    return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
      const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
      if (entry.isDirectory())
        return entry.name === "e2e" ? [] : sourceFiles(join(dir, entry.name), rel);
      return entry.name.endsWith(".ts") && !entry.name.includes(".test.") ? [rel] : [];
    });
  }

  it("has no module constructing a client outside the factory without a stated reason", () => {
    const unexplained = sourceFiles(SRC).filter(
      (rel) =>
        !(rel in DIRECT) && readFileSync(join(SRC, rel), "utf8").includes("new HindsightClient(")
    );
    expect(unexplained).toEqual([]);
  });

  it("keeps no entry for a module that stopped constructing one", () => {
    const stale = Object.keys(DIRECT).filter(
      (rel) => !readFileSync(join(SRC, rel), "utf8").includes("new HindsightClient(")
    );
    expect(stale).toEqual([]);
  });
});

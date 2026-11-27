import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { getHarness, HARNESS_NAMES, PLUGIN_ENTRYPOINTS } from "./registry";
import { HOOK_HARNESSES, type HookHarnessName } from "./hook-lifecycle";
import { ALL_HARNESS_SETUPS } from "../e2e/harnesses";

describe("HARNESS_NAMES", () => {
  it("lists all registered harnesses", () => {
    expect(HARNESS_NAMES).toEqual(
      expect.arrayContaining([
        "opencode",
        "opencode2",
        "kilo",
        "cline-cli",
        "pi",
        "prime-agent",
        "dsh",
        "claude-code",
        "cursor-cli",
        "codex",
        "dcode",
        "antigravity-cli",
        "devin-cli",
        "copilot-cli",
        "grok-build",
        "qwen-code",
        "factory-droid",
        "zcode",
        "traecode",
        "kimi-code",
        "workbuddy",
        "codebuddy",
      ])
    );
    expect(HARNESS_NAMES).toHaveLength(22);
  });

  // Family guard: a new harness that lands without a Docker E2E setup is never driven through its
  // real CLI, and no per-harness test can notice the absence. Kimi's prompt-parsing bug (a block
  // array read as a string, so recall never ran) passed every unit test and only the real CLI
  // showed it.
  it("gives every installable harness a Docker E2E setup and image", () => {
    // Antigravity's harness E2E is covered separately (see e2e/harnesses.ts).
    // TraeCode is the agent inside the TRAE desktop IDE, with no documented headless mode a
    // container could drive; its hook wire is covered by journal-harnesses.test.ts instead.
    // WorkBuddy ships inside Tencent's desktop app — same story as TraeCode: no documented
    // headless mode a container could drive. CodeBuddy Code has a CLI, but driving it needs a
    // Tencent account. Both are covered by installer.test.ts and their transcript-reader tests.
    const exempt = new Set(["antigravity-cli", "traecode", "workbuddy", "codebuddy"]);
    const root = fileURLToPath(new URL("../..", import.meta.url));
    const setups = new Set(ALL_HARNESS_SETUPS.map((setup) => setup.name));
    for (const name of HARNESS_NAMES.filter((n) => !exempt.has(n))) {
      expect(setups.has(name), `${name} has no entry in ALL_HARNESS_SETUPS`).toBe(true);
      expect(existsSync(join(root, "e2e", `Dockerfile.${name}`)), `e2e/Dockerfile.${name}`).toBe(
        true
      );
    }
  });
});

describe("getHarness", () => {
  it("resolves hook harnesses without touching the opencode adapter", async () => {
    for (const name of [
      "claude-code",
      "cursor-cli",
      "codex",
      "dcode",
      "antigravity-cli",
      "devin-cli",
    ]) {
      const adapter = await getHarness(name);
      expect(adapter.name).toBe(name);
      // Lightweight hook adapters have no persistent runtime — createRuntime always throws before
      // touching its argument, so a stand-in value is fine here.
      expect(() => adapter.createRuntime({} as never)).toThrow();
    }
  });

  it("resolves the opencode adapter by name", async () => {
    const adapter = await getHarness("opencode");
    expect(adapter.name).toBe("opencode");
    // Deliberate invariant: opencode via the registry is a no-runtime adapter (same shape as the
    // hook harnesses) — the real opencode runtime is built by src/index.ts importing opencodeAdapter
    // directly, bypassing this registry. Lock it so a future change can't silently make this look
    // functional.
    expect(() => adapter.createRuntime({} as never)).toThrow();
  });

  it("resolves opencode2 as its own plugin harness, separate from v1", async () => {
    // Same no-runtime shape as opencode above, but a DIFFERENT entrypoint: v2's plugin contract
    // shares nothing with v1's, so handing either host the other's export loads a plugin that
    // registers nothing and reports no error.
    const adapter = await getHarness("opencode2");
    expect(adapter.name).toBe("opencode2");
    expect(() => adapter.createRuntime({} as never)).toThrow(/src\/opencode2\.ts/);
  });

  it("resolves Cline as a native-plugin harness rather than a hook binary", async () => {
    const adapter = await getHarness("cline-cli");
    expect(adapter.name).toBe("cline-cli");
    expect(() => adapter.createRuntime({} as never)).toThrow(/src\/cline\.ts/);
  });

  it("resolves pi and Prime Agent to their own runtimes", async () => {
    // pi is an extension (src/pi.ts); Prime Agent 0.10's runtime runs inside the MCP server
    // (src/prime-agent.ts). Swapping them would retain one host's sessions under the other's bank.
    for (const [name, entry] of [
      ["pi", /src\/pi\.ts/],
      ["prime-agent", /src\/prime-agent\.ts/],
    ] as const) {
      const adapter = await getHarness(name);
      expect(adapter.name).toBe(name);
      expect(() => adapter.createRuntime({} as never)).toThrow(entry);
    }
  });

  it("resolves DeepSeek Harness as a native Cordis-plugin harness", async () => {
    const adapter = await getHarness("dsh");
    expect(adapter.name).toBe("dsh");
    expect(() => adapter.createRuntime({} as never)).toThrow(/src\/dsh\.ts/);
  });

  it("rejects unknown harness names", async () => {
    await expect(getHarness("nope")).rejects.toThrow(/unknown harness/);
  });
});

/**
 * The registry and the installer are separate lists that must agree: `deepen` resolves a harness
 * through the registry, so an installer the registry doesn't know throws "unknown harness" and the
 * background git-diff enrichment dies. That is exactly how grok-build and copilot-cli shipped
 * broken in 0.0.1 — installable, but unusable by deepen.
 */
describe("registry covers every installable harness", () => {
  it("has no harness the installer wires but the registry rejects", async () => {
    const { INSTALLERS } = await import("../installer");
    const missing = INSTALLERS.map((i) => i.name).filter((n) => !HARNESS_NAMES.includes(n));
    expect(missing).toEqual([]);
  });

  // The check above is DIRECTIONAL, and the other direction is just as broken: a harness in the
  // registry with no installer is advertised by the README and rejected by `install <name>` with
  // "unknown harness". Every hook harness must be installable — the plugin harnesses are wired by
  // their own entrypoints and are exempt only because they still carry an INSTALLERS entry.
  it("has no hook harness the registry names but the installer cannot wire", async () => {
    const { INSTALLERS } = await import("../installer");
    const { HOOK_HARNESSES } = await import("./hook-lifecycle");
    const installable = new Set(INSTALLERS.map((i) => i.name));
    const missing = Object.keys(HOOK_HARNESSES).filter((n) => !installable.has(n));
    expect(missing).toEqual([]);
  });
});

/**
 * The harness name an entrypoint reports is a bare string literal that nothing else checks: it
 * selects the `harnesses.<name>` config section, feeds `{harness}` bank templating, and is stamped
 * on every document that host retains. A typo there ships green — the registry, the installer and
 * the control plane's logo map are three SEPARATE hand-maintained lists, so none of them notices
 * that the running plugin calls itself something else.
 *
 * So assert it over the whole family, enumerated from the registry rather than a fourth list: every
 * entrypoint the registry names must report the harness the registry maps it to.
 */
describe("every plugin entrypoint reports the harness the registry maps it to", () => {
  const PKG = fileURLToPath(new URL("../..", import.meta.url));

  /** The two idioms an entrypoint uses to name itself: the argument it hands its shared factory
   *  (opencode/Kilo via createPluginEntry, opencode 2 via createOpencode2PluginEntry, pi/Prime Agent
   *  via createPiExtension), or its own module constant (Cline and dsh, which build their runtime
   *  themselves). Matching the factory NAMES rather than a generic call keeps the guard honest: an
   *  entrypoint that names itself some new way fails here until this list is widened, which is
   *  exactly the review that a new host needs.  */
  const DECLARES_HARNESS =
    /(?:createPluginEntry|createOpencode2PluginEntry|createPiExtension)\("([^"]+)"\)|const HARNESS = "([^"]+)"/;

  it.each(Object.entries(PLUGIN_ENTRYPOINTS))("%s (%s)", (harness, entry) => {
    const declaration = readFileSync(join(PKG, entry), "utf8").match(DECLARES_HARNESS);
    expect(declaration, `${entry} names no harness in a form this guard recognises`).not.toBeNull();
    expect(declaration![1] ?? declaration![2]).toBe(harness);
  });
});

/**
 * A compaction counts as a new session start (#6527). Every harness has to make that decision, so
 * this list names each one: how its compaction reaches us, or why it cannot. A new harness fails
 * here until someone checks its host for a compaction signal.
 */
const COMPACTION_SIGNAL: Record<string, string> = {
  "claude-code": "SessionStart source compact",
  codex: "SessionStart source compact",
  "qwen-code": "SessionStart source compact",
  "factory-droid": "SessionStart source compact",
  "cursor-cli": "hook:preCompact",
  "kimi-code": "hook:PostCompact",
  opencode: "event session.compacted",
  kilo: "event session.compacted (opencode fork, same plugin entry)",
  opencode2: "event session.compaction.ended",
  pi: "event session_compact",
  // No documented compaction signal on these hosts. A Claude-style host among them that does send
  // SessionStart `source: "compact"` is still handled: runSessionStartHook checks it for every host.
  "copilot-cli": "none: sessionStart sources are startup/resume/new",
  "antigravity-cli": "none: no SessionStart or compaction hook",
  "devin-cli": "none documented",
  "grok-build": "none documented",
  zcode: "none documented",
  traecode: "none documented",
  dcode: "none documented",
  workbuddy: "none documented",
  codebuddy: "none documented",
  "cline-cli": "none: the plugin API has beforeModel/afterRun only",
  dsh: "none: no compaction event among the Cordis agent events",
  "prime-agent": "none: no hooks or extensions",
};

describe("every harness decides how a compaction reaches it (#6527)", () => {
  it("names every harness", () => {
    expect(Object.keys(COMPACTION_SIGNAL).sort()).toEqual([...HARNESS_NAMES].sort());
  });

  it.each(Object.entries(COMPACTION_SIGNAL).filter(([, how]) => how.startsWith("hook:")))(
    "%s registers its compaction hook on the SessionStart entry",
    (harness, how) => {
      const spec = HOOK_HARNESSES[harness as HookHarnessName];
      expect(spec.additionalHooks).toContainEqual(
        expect.objectContaining({
          event: how.slice("hook:".length),
          entry: spec.install.sessionStart.entry,
        })
      );
    }
  );
});

/**
 * Which persistent-plugin hosts replay a session whose write-back failed before the host exited
 * (#6559): only the ones whose host can read back a session by id after the process that ran it is
 * gone. A new host lands here, and has to say which it is.
 */
describe("persistent-plugin hosts that replay a failed session after a restart", () => {
  const PKG = fileURLToPath(new URL("../..", import.meta.url));

  /** harness -> the file that wires its RuntimeCore, and whether it sets a past-session source. */
  const PAST_SESSION_SOURCE: Record<string, { file: string; replays: boolean }> = {
    // The host's server keeps every session: GET /session/{id}/message reads any of them.
    opencode: { file: "src/harness/plugin-entry.ts", replays: true },
    kilo: { file: "src/harness/plugin-entry.ts", replays: true },
    // ctx.sessionPersistence.open(id, "read") reads a session no agent holds.
    dsh: { file: "src/dsh.ts", replays: true },
    // `session.context` reads the live session's context; not known to reach a session this
    // process never opened, so it is not trusted to.
    opencode2: { file: "src/harness/opencode2.ts", replays: false },
    // The host hands over the transcript in its own events and offers no read by session id.
    "cline-cli": { file: "src/cline.ts", replays: false },
    pi: { file: "src/harness/pi-extension.ts", replays: false },
    // Its transcript source returns the turns of the run it was given, whatever id it is asked
    // for: as a past-session source it would write that run into another session's document.
    "prime-agent": { file: "src/prime-agent.ts", replays: false },
  };

  it("covers every persistent-plugin host", () => {
    expect(Object.keys(PAST_SESSION_SOURCE).sort()).toEqual(Object.keys(PLUGIN_ENTRYPOINTS).sort());
  });

  it.each(Object.entries(PAST_SESSION_SOURCE))("%s", (_harness, { file, replays }) => {
    expect(readFileSync(join(PKG, file), "utf8").includes("setPastSessionSource")).toBe(replays);
  });
});

import { beforeEach, describe, expect, it, vi } from "vitest";
import { resolveConfig } from "./config";
import type { HindsightClient } from "./hindsight";
import { RuntimeCore } from "./runtime";

// The real ensureDaemon shells out (uvx/cargo probes) and spawns a detached process; only WHERE it
// is called from is under test here — the deciding itself is covered in daemon.test.ts.
const daemonSpy = vi.hoisted(() => vi.fn(async () => {}));
vi.mock("./daemon", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./daemon")>()),
  ensureDaemon: daemonSpy,
}));

describe("RuntimeCore", () => {
  it("uses the shared prompt lifecycle and consumes the new-bank reflect deferral once", async () => {
    const client = {
      listDocumentIds: vi.fn(async () => new Set(["git:existing"])),
      listPages: vi.fn(async () => ({ items: [] })),
      reflect: vi.fn(async () => "shared reflect"),
    } as unknown as HindsightClient;
    const runtime = new RuntimeCore(client, "bank-1", resolveConfig({}));

    await runtime.seedIfCold("/definitely-not-a-git-repository");
    await runtime.onPrompt("runtime-shared-lifecycle", "first prompt");
    expect(client.reflect).not.toHaveBeenCalled();

    await runtime.onPrompt("runtime-shared-lifecycle", "second prompt");
    expect(client.reflect).toHaveBeenCalledTimes(1);
    expect(runtime.getInjection("runtime-shared-lifecycle")).toContain("shared reflect");
  });
});

/**
 * `messages.transform` fires while the host BUILDS a request, so the transcript it passes never
 * contains the reply that request produces. session.idle is the only moment a completed exchange is
 * readable — without it write-back lagged a turn and a session's final exchange was lost entirely.
 */
describe("RuntimeCore session-idle write-back", () => {
  const turn = (role: string, content: string) => ({ role, content }) as never;

  /** A client that records what retain() was given. */
  function makeClient() {
    const retained: { turns: unknown; docId: string }[] = [];
    const client = {
      retain: vi.fn(async (text: string, _ctx: string, docId: string) => {
        retained.push({ turns: text, docId });
      }),
    } as unknown as HindsightClient;
    return { client, retained };
  }

  it("retains the completed exchange on idle — the reply the turn-driven path never sees", async () => {
    const { client, retained } = makeClient();
    const runtime = new RuntimeCore(client, "bank-1", resolveConfig({}));

    // The turn-driven path sees only what existed BEFORE the reply.
    await runtime.onTranscript("s1", [turn("user", "how do we round?")]);
    await new Promise((r) => setTimeout(r, 0)); // retain is fire-and-forget
    expect(retained).toHaveLength(1);
    expect(String(retained[0].turns)).not.toContain("we round half up");

    // Idle refetches, so the assistant's answer finally lands.
    runtime.setTranscriptSource(async () => [
      turn("user", "how do we round?"),
      turn("assistant", "we round half up"),
    ]);
    await runtime.onSessionIdle("s1");
    await new Promise((r) => setTimeout(r, 0)); // retain is fire-and-forget

    expect(retained).toHaveLength(2);
    expect(String(retained[1].turns)).toContain("we round half up");
    expect(retained[1].docId).toBe("conversation:s1");
  });

  it("writes back every turn — no client-side cadence to strand a short session", async () => {
    // There used to be a retainEveryTurns threshold here, which suppressed a session shorter than
    // it entirely. Batching now happens server-side (engine.retain.fold coalesces queued retains
    // for one document), so every turn is submitted immediately and nothing is held in a process
    // the host can close without warning.
    const { client, retained } = makeClient();
    const runtime = new RuntimeCore(client, "bank-1", resolveConfig({}));

    await runtime.onTranscript("s2", [turn("user", "only turn")]);
    await new Promise((r) => setTimeout(r, 0));
    expect(retained).toHaveLength(1);

    runtime.setTranscriptSource(async () => [turn("user", "only turn"), turn("assistant", "done")]);
    await runtime.onSessionIdle("s2");
    await new Promise((r) => setTimeout(r, 0));
    expect(retained).toHaveLength(2);
  });

  it("does not re-retain when idle fires again with no new turns", async () => {
    const { client, retained } = makeClient();
    const runtime = new RuntimeCore(client, "bank-1", resolveConfig({}));
    runtime.setTranscriptSource(async () => [turn("user", "hi"), turn("assistant", "hello")]);

    await runtime.onSessionIdle("s3");
    await runtime.onSessionIdle("s3");
    await new Promise((r) => setTimeout(r, 0));
    expect(retained).toHaveLength(1);
  });

  it("no-ops when the host cannot refetch, rather than retaining a stale transcript", async () => {
    const { client, retained } = makeClient();
    const runtime = new RuntimeCore(client, "bank-1", resolveConfig({}));
    await runtime.onSessionIdle("s4"); // no transcript source wired
    await new Promise((r) => setTimeout(r, 0));
    expect(retained).toHaveLength(0);
  });

  it("survives a failing refetch without throwing into the host", async () => {
    const { client, retained } = makeClient();
    const runtime = new RuntimeCore(client, "bank-1", resolveConfig({}));
    runtime.setTranscriptSource(async () => {
      throw new Error("session gone");
    });
    await expect(runtime.onSessionIdle("s5")).resolves.toBeUndefined();
    expect(retained).toHaveLength(0);
  });
});

/**
 * Daemon parity with the hook harnesses (#3524).
 *
 * A persistent-plugin host runs no hook binaries, so `runSessionStartHook`/`runRetainHook` — where
 * the hook harnesses ensure the daemon — never execute for it. dsh in daemon mode therefore never
 * started one: every `hindsight_*` call failed with ECONNREFUSED until the user ran daemon-start.js
 * by hand, and again after each idle-out. RuntimeCore is the one place all five plugin hosts
 * (opencode, Kilo, Cline, Prime Agent, dsh) share, so the two ensure points live here.
 */
describe("RuntimeCore daemon lifecycle", () => {
  const daemonCfg = resolveConfig({ serverMode: "daemon" });
  const client = {
    listDocumentIds: vi.fn(async () => new Set<string>()),
    listPages: vi.fn(async () => ({ items: [] })),
    retain: vi.fn(async () => {}),
  } as unknown as HindsightClient;

  beforeEach(() => daemonSpy.mockClear());

  it("starts the daemon at seedIfCold — the plugin hosts' SessionStart", async () => {
    const core = new RuntimeCore(client, "bank-1", daemonCfg, "dsh", "/repos/one");
    await core.seedIfCold("/repos/one");
    expect(daemonSpy).toHaveBeenCalledWith(daemonCfg, "dsh", { waitMs: 12_000 });
  });

  // A daemon that is gone by write-back time — crashed, or stopped by the user between turns —
  // would lose the whole exchange with no second chance to start one.
  it("starts the daemon again on write-back — the Stop hook these hosts don't have", async () => {
    const core = new RuntimeCore(client, "bank-1", daemonCfg, "dsh", "/repos/one");
    core.setTranscriptSource(async () => [{ role: "user", content: "hi" }] as never);
    await core.onSessionIdle("s1");
    await new Promise((r) => setTimeout(r, 0)); // retain is fire-and-forget
    expect(daemonSpy).toHaveBeenCalledWith(daemonCfg, "dsh", { waitMs: 40_000 });
    expect(client.retain).toHaveBeenCalled();
  });

  // Waiting inside the fire-and-forget chain is the point: the host's stop handler awaits
  // onSessionIdle, and a 40s block there would freeze the UI on a cold daemon.
  it("never makes the host wait for the daemon on the write path", async () => {
    let released = () => {};
    daemonSpy.mockImplementationOnce(() => new Promise<void>((r) => (released = r)));
    const core = new RuntimeCore(client, "bank-1", daemonCfg, "dsh", "/repos/one");
    core.setTranscriptSource(async () => [{ role: "user", content: "hi" }] as never);
    await expect(core.onSessionIdle("s2")).resolves.toBeUndefined();
    released();
  });
});

/**
 * The plugin harnesses (opencode, kilo, cline, dsh, ...) register the tools through toolSpecs()
 * rather than the MCP server, so they hit the same #3590 bug: without these forwarded, every
 * hindsight_reflect on those hosts aborts at the client's hardcoded 120s.
 */
describe("RuntimeCore reflect tool settings", () => {
  it("forwards the resolved reflect timeout and budget into hindsight_reflect", async () => {
    const reflect = vi.fn(async () => "answer");
    const client = { reflect } as unknown as HindsightClient;
    const core = new RuntimeCore(
      client,
      "bank-1",
      resolveConfig({ reflectToolTimeoutMs: 660_000, reflectBudget: "mid" })
    );

    const tool = core.toolSpecs().find((spec) => spec.name === "hindsight_reflect")!;
    await tool.handler({ query: "why?" });
    expect(reflect).toHaveBeenCalledWith("why?", { budget: "mid", timeoutMs: 660_000 });
  });
});

/**
 * dsh serves several repositories from ONE process, launched in a directory that is routinely not
 * the session's — so it constructs a core per workspace and passes that root. Every tool that
 * answers "what does this repo's memory look like" has to be bound to it: otherwise
 * `hindsight_sync_status` runs its git checks in the launch directory and looks for THAT repo's
 * `gitlog:<name>` document in the SESSION's bank, reporting a seeded repo as unsynced.
 */
describe("RuntimeCore tool workspace binding", () => {
  const client = {
    listDocumentIds: vi.fn(async () => new Set<string>()),
    listPages: vi.fn(async () => ({ items: [] })),
    activeOperations: vi.fn(async () => 0),
  } as unknown as HindsightClient;

  const workspaceOf = async (core: RuntimeCore): Promise<string> => {
    const diagnose = core.toolSpecs().find((spec) => spec.name === "hindsight_diagnose")!;
    return (JSON.parse((await diagnose.handler({})).content[0].text) as { workspace: string })
      .workspace;
  };

  it("binds the tools to the workspace the host opened, not the process cwd", async () => {
    const core = new RuntimeCore(client, "bank-1", resolveConfig({}), "dsh", "/repos/session-one");
    expect(await workspaceOf(core)).toBe("/repos/session-one");
    expect(await workspaceOf(core)).not.toBe(process.cwd());
  });

  it("falls back to the process cwd for hosts whose process IS the project", async () => {
    const core = new RuntimeCore(client, "bank-1", resolveConfig({}));
    expect(await workspaceOf(core)).toBe(process.cwd());
  });
});

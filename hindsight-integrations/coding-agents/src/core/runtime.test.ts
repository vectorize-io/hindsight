import { describe, expect, it, vi } from "vitest";
import { resolveConfig } from "./config";
import type { HindsightClient } from "./hindsight";
import { RuntimeCore } from "./runtime";

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

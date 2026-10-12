import { describe, expect, it, vi } from "vitest";
import { createMemoryProviderRuntime, type MemoryProviderDeps } from "./memory-provider.js";
import { agentOnlyBankId } from "./index.js";

const operator = { kind: "operator" as const, scopes: ["operator.read"] };
const ownerSession = {
  kind: "session" as const,
  sessionKey: "agent:main:main",
  sandboxed: false,
  audience: { kind: "owner-private" as const, agentId: "main" },
};
const groupSession = {
  kind: "session" as const,
  sessionKey: "agent:main:telegram:group:1",
  sandboxed: false,
  audience: {
    kind: "conversation" as const,
    agentId: "main",
    sessionKey: "agent:main:telegram:group:1",
    sessionId: "s1",
  },
};

function deps(overrides: Partial<MemoryProviderDeps> = {}): MemoryProviderDeps {
  return {
    providerId: "hindsight-openclaw",
    agentBank: (agentId) => `bank-${agentId}`,
    recall: vi.fn().mockResolvedValue([
      { id: "m1", text: "Prefers persimmons", score: 0.9 },
      { id: "m2", text: "Lives in Rome", score: 0.2 },
      { id: "m3", text: "Unscored", score: null },
    ]),
    getMemory: vi.fn().mockResolvedValue("Prefers persimmons"),
    healthy: vi.fn().mockResolvedValue(true),
    ...overrides,
  };
}

async function open(d: MemoryProviderDeps, authority: object, agentId = "main") {
  const { provider } = await createMemoryProviderRuntime(d).open({
    agentId,
    context: { authority: authority as never, assertCurrent: () => {} },
  });
  return provider!;
}

describe("native memory provider (#4916)", () => {
  it("searches the agent's bank for an operator, honouring minScore and maxResults", async () => {
    const d = deps();
    const provider = await open(d, operator);
    const page = await provider.search({ query: "fruit", minScore: 0.5 });
    expect(d.recall).toHaveBeenCalledWith("bank-main", "fruit", undefined);
    expect(page.hits).toEqual([
      {
        reference: { providerId: "hindsight-openclaw", id: "m1" },
        excerpt: "Prefers persimmons",
        score: 0.9,
        source: "memory",
      },
    ]);
    expect((await provider.search({ query: "fruit", maxResults: 1 })).hits).toHaveLength(1);
  });

  it("lets the agent owner's private session read, and refuses a group conversation", async () => {
    expect((await (await open(deps(), ownerSession)).search({ query: "x" })).hits).toHaveLength(3);
    const group = await open(deps(), groupSession);
    await expect(group.search({ query: "x" })).rejects.toThrow(/not readable/);
    await expect(
      group.get({ reference: { providerId: "hindsight-openclaw", id: "m1" } })
    ).rejects.toThrow(/not readable/);
  });

  it("refuses search when the bank is keyed by user or channel", async () => {
    const provider = await open(deps({ agentBank: () => undefined }), operator);
    await expect(provider.search({ query: "x" })).rejects.toThrow(/user or channel/);
  });

  it("reports health to every caller, even ones that can't read", async () => {
    expect(await (await open(deps(), groupSession)).health()).toEqual({ status: "ready" });
    const down = await open(deps({ healthy: vi.fn().mockResolvedValue(false) }), {
      kind: "host",
      operation: "status",
    });
    expect((await down.health()).status).toBe("unavailable");
  });

  it("gets a memory by reference and rejects another provider's reference", async () => {
    const d = deps();
    const provider = await open(d, operator);
    expect(
      await provider.get({ reference: { providerId: "hindsight-openclaw", id: "m1" } })
    ).toEqual({
      status: "ok",
      reference: { providerId: "hindsight-openclaw", id: "m1" },
      text: "Prefers persimmons",
    });
    expect(await provider.get({ reference: { providerId: "memory-core", id: "m1" } })).toEqual({
      status: "not_found",
    });
    expect(d.getMemory).toHaveBeenCalledTimes(1);
  });
});

describe("agentOnlyBankId", () => {
  it("resolves banks that depend on the agent alone", () => {
    expect(agentOnlyBankId("main", { dynamicBankGranularity: ["agent"] })).toBe("main");
    expect(agentOnlyBankId("main", { dynamicBankGranularity: ["agent"], bankIdPrefix: "p" })).toBe(
      "p-main"
    );
    expect(agentOnlyBankId("main", { dynamicBankId: false, bankId: "shared" })).toBe("shared");
    expect(
      agentOnlyBankId("ops", {
        dynamicBankGranularity: ["agent", "user"],
        agentBankMap: { ops: "ops-bank" },
      })
    ).toBe("ops-bank");
  });

  it("returns nothing when the bank is keyed by user, channel or provider", () => {
    expect(agentOnlyBankId("main", {})).toBeUndefined(); // default: agent, channel, user
    expect(agentOnlyBankId("main", { dynamicBankGranularity: ["agent", "user"] })).toBeUndefined();
    expect(
      agentOnlyBankId("main", { dynamicBankGranularity: ["agent", "provider"] })
    ).toBeUndefined();
  });
});

import { describe, it, expect } from "vitest";
import { deriveBankId } from "./index.js";
import type { PluginHookAgentContext, PluginConfig } from "./types.js";

describe("deriveBankId", () => {
  const ctx: PluginHookAgentContext = {
    agentId: "agent-123",
    channelId: "channel-456",
    senderId: "user-789",
    messageProvider: "slack",
  };

  const baseConfig: PluginConfig = {
    dynamicBankId: true,
  };

  it("should use default isolation fields when not specified", () => {
    const bankId = deriveBankId(ctx, baseConfig);
    expect(bankId).toBe("agent-123::channel-456::user-789");
  });

  it("should default to dynamic bank ID when dynamicBankId is not specified", () => {
    const config: PluginConfig = {};
    const bankId = deriveBankId(ctx, config);
    expect(bankId).toBe("agent-123::channel-456::user-789");
  });

  it('should support ["agent", "user"] isolation', () => {
    const config: PluginConfig = { ...baseConfig, dynamicBankGranularity: ["agent", "user"] };
    const bankId = deriveBankId(ctx, config);
    expect(bankId).toBe("agent-123::user-789");
  });

  it('should support ["user"] isolation', () => {
    const config: PluginConfig = { ...baseConfig, dynamicBankGranularity: ["user"] };
    const bankId = deriveBankId(ctx, config);
    expect(bankId).toBe("user-789");
  });

  it('should support ["agent"] isolation', () => {
    const config: PluginConfig = { ...baseConfig, dynamicBankGranularity: ["agent"] };
    const bankId = deriveBankId(ctx, config);
    expect(bankId).toBe("agent-123");
  });

  it('should support ["channel"] isolation', () => {
    const config: PluginConfig = { ...baseConfig, dynamicBankGranularity: ["channel"] };
    const bankId = deriveBankId(ctx, config);
    expect(bankId).toBe("channel-456");
  });

  it('should support ["provider"] isolation', () => {
    const config: PluginConfig = { ...baseConfig, dynamicBankGranularity: ["provider"] };
    const bankId = deriveBankId(ctx, config);
    expect(bankId).toBe("slack");
  });

  it("should support mixed fields including provider", () => {
    const config: PluginConfig = { ...baseConfig, dynamicBankGranularity: ["provider", "user"] };
    const bankId = deriveBankId(ctx, config);
    expect(bankId).toBe("slack::user-789");
  });

  it("should prepend bankIdPrefix if set", () => {
    const config: PluginConfig = { ...baseConfig, bankIdPrefix: "prod" };
    const bankId = deriveBankId(ctx, config);
    expect(bankId).toBe("prod-agent-123::channel-456::user-789");
  });

  it("should use fallback values for missing context fields", () => {
    const partialCtx: PluginHookAgentContext = {
      agentId: "agent-123",
    };
    const bankId = deriveBankId(partialCtx, baseConfig);
    expect(bankId).toBe("agent-123::unknown::anonymous");
  });

  it("should parse sessionKey as fallback for missing channel and provider", () => {
    const ctxWithSession: PluginHookAgentContext = {
      agentId: "my-agent",
      sessionKey: "agent:my-agent:telegram:group:-100123456:topic:7",
    };
    const config: PluginConfig = {
      ...baseConfig,
      dynamicBankGranularity: ["agent", "channel", "provider"],
    };
    const bankId = deriveBankId(ctxWithSession, config);
    expect(bankId).toBe("my-agent::group%3A-100123456%3Atopic%3A7::telegram");
  });

  it('should return "openclaw" if dynamicBankId is false', () => {
    const config: PluginConfig = { dynamicBankId: false };
    const bankId = deriveBankId(ctx, config);
    expect(bankId).toBe("openclaw");
  });

  it("should return configured bankId when dynamicBankId is false", () => {
    const config: PluginConfig = {
      dynamicBankId: false,
      bankId: "shared-bank",
      bankIdPrefix: "prod",
      dynamicBankGranularity: ["provider", "user"],
    };
    const bankId = deriveBankId(ctx, config);
    expect(bankId).toBe("prod-shared-bank");
  });

  it("should ignore ctx.channelId when it is a provider name and fall back to sessionKey (issue #854)", () => {
    const ctxDiscord: PluginHookAgentContext = {
      agentId: "main",
      channelId: "discord",
      sessionKey: "agent:main:discord:channel:1472750640760623226",
    };
    const config: PluginConfig = { ...baseConfig, dynamicBankGranularity: ["agent", "channel"] };
    const bankId = deriveBankId(ctxDiscord, config);
    expect(bankId).toBe("main::channel%3A1472750640760623226");
  });

  it("should encode segments to prevent separator collisions", () => {
    const ctxWithSeparator: PluginHookAgentContext = {
      agentId: "a::b",
      channelId: "c",
      senderId: "user-1",
    };
    const ctxWithoutSeparator: PluginHookAgentContext = {
      agentId: "a",
      channelId: "b::c",
      senderId: "user-1",
    };
    const bankId1 = deriveBankId(ctxWithSeparator, baseConfig);
    const bankId2 = deriveBankId(ctxWithoutSeparator, baseConfig);
    // These must NOT collide
    expect(bankId1).not.toBe(bankId2);
    // Segment delimiters are encoded, preserving unique values.
    expect(bankId1).toBe("a%3A%3Ab::c::user-1");
    expect(bankId2).toBe("a::b%3A%3Ac::user-1");
  });
});

// Mixed topology in one gateway: some agents share a named bank, the rest keep
// their derived ones. Before this, isolation was binary — one static bank for
// everyone, or a derived bank per agent with no way to group. (#3890)
describe("deriveBankId with agentBankMap", () => {
  const ctx: PluginHookAgentContext = {
    agentId: "inbound",
    channelId: "channel-456",
    senderId: "user-789",
    messageProvider: "slack",
  };

  it("routes a mapped agent to its bank, ahead of dynamic derivation", () => {
    const config: PluginConfig = {
      dynamicBankId: true,
      agentBankMap: { inbound: "ps-technology", limpieza: "ps-limpieza" },
    };
    expect(deriveBankId(ctx, config)).toBe("ps-technology");
  });

  it("leaves an unmapped agent on its derived bank", () => {
    const config: PluginConfig = {
      dynamicBankId: true,
      agentBankMap: { limpieza: "ps-limpieza" },
    };
    expect(deriveBankId(ctx, config)).toBe("inbound::channel-456::user-789");
  });

  it("wins over the static bank, so both topologies can coexist", () => {
    const config: PluginConfig = {
      dynamicBankId: false,
      bankId: "shared-bank",
      agentBankMap: { inbound: "ps-technology" },
    };
    expect(deriveBankId(ctx, config)).toBe("ps-technology");
  });

  it("leaves an unmapped agent on the static bank", () => {
    const config: PluginConfig = {
      dynamicBankId: false,
      bankId: "shared-bank",
      agentBankMap: { limpieza: "ps-limpieza" },
    };
    expect(deriveBankId(ctx, config)).toBe("shared-bank");
  });

  it("uses the mapped name exactly, without bankIdPrefix", () => {
    // The operator named this bank; prefixing it would point at a different one.
    const config: PluginConfig = {
      dynamicBankId: true,
      bankIdPrefix: "prod",
      agentBankMap: { inbound: "ps-technology" },
    };
    expect(deriveBankId(ctx, config)).toBe("ps-technology");
  });

  it("resolves the agent id from the session key when the context carries none", () => {
    const config: PluginConfig = {
      dynamicBankId: true,
      agentBankMap: { "my-agent": "ps-shared" },
    };
    const ctxWithSession: PluginHookAgentContext = {
      sessionKey: "agent:my-agent:telegram:group:-100123456",
    };
    expect(deriveBankId(ctxWithSession, config)).toBe("ps-shared");
  });

  it("falls back to derivation when there is no context to map", () => {
    const config: PluginConfig = {
      dynamicBankId: true,
      agentBankMap: { inbound: "ps-technology" },
    };
    expect(deriveBankId(undefined, config)).toBe("openclaw");
  });

  it("does not inherit bank ids from the prototype chain", () => {
    // An agent literally called "toString" must not pick up Object.prototype's
    // method, which is truthy and is not a bank id.
    const config: PluginConfig = { dynamicBankId: true, agentBankMap: {} };
    const ctxNamedLikeAPrototypeKey: PluginHookAgentContext = {
      agentId: "toString",
      channelId: "channel-456",
      senderId: "user-789",
    };
    expect(deriveBankId(ctxNamedLikeAPrototypeKey, config)).toBe("toString::channel-456::user-789");
  });

  it("ignores a mapped value that is not a usable string", () => {
    // The backfill CLI builds its config straight from openclaw.json, without
    // normalizeAgentBankMap, so the lookup itself has to reject this.
    const config = {
      dynamicBankId: true,
      agentBankMap: { inbound: 42 },
    } as unknown as PluginConfig;
    expect(deriveBankId(ctx, config)).toBe("inbound::channel-456::user-789");
  });
});

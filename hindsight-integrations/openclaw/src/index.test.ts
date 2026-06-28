import { describe, it, expect } from "vitest";
import { createRequire } from "module";
import {
  stripMemoryTags,
  extractRecallQuery,
  formatCurrentTimeForRecall,
  formatMemories,
  prepareRetentionTranscript,
  countUserTurns,
  getRetentionTurnIndex,
  sliceLastTurnsByUserBoundary,
  composeRecallQuery,
  truncateRecallQuery,
  buildRetainRequest,
  meetsMinimumVersion,
  parseSessionKey,
  extractTelegramDirectSenderId,
  resolveSessionIdentity,
  resolveAndCacheIdentity,
  getIdentitySkipReason,
  isEphemeralOperationalText,
  deriveBankId,
  resolveBankIdForKnowledgeTools,
  normalizeRetainTags,
  extractInlineRetainTags,
  stripInlineRetainTags,
  stripInlineTimestampPrefix,
  stripRuntimeEnvelope,
  getPluginConfig,
  formatHookPerf,
  DEFAULT_RETAIN_CONTEXT,
} from "./index.js";
import type { PluginConfig, MemoryResult, MoltbotPluginAPI } from "./types.js";

const require = createRequire(import.meta.url);
const openclawManifest = require("../openclaw.plugin.json") as {
  configSchema?: {
    properties?: {
      retainContext?: {
        default?: string;
      };
    };
  };
};

// ---------------------------------------------------------------------------
// stripMemoryTags
// ---------------------------------------------------------------------------

describe("stripMemoryTags", () => {
  it("strips simple hindsight_memories tags", () => {
    const input =
      "User: Hello\n<hindsight_memories>\nRelevant memories here...\n</hindsight_memories>\nAssistant: How can I help?";
    expect(stripMemoryTags(input)).toBe("User: Hello\n\nAssistant: How can I help?");
  });

  it("strips relevant_memories tags", () => {
    const input = "Before\n<relevant_memories>\nSome data\n</relevant_memories>\nAfter";
    expect(stripMemoryTags(input)).toBe("Before\n\nAfter");
  });

  it("strips multiple hindsight_memories blocks", () => {
    const input =
      "Start\n<hindsight_memories>\nBlock 1\n</hindsight_memories>\nMiddle\n<hindsight_memories>\nBlock 2\n</hindsight_memories>\nEnd";
    expect(stripMemoryTags(input)).toBe("Start\n\nMiddle\n\nEnd");
  });

  it("handles multiline memory blocks with JSON", () => {
    const input =
      'User: What is the weather?\n<hindsight_memories>\n[\n  {"memory": "User likes sunny weather"}\n]\n</hindsight_memories>\nAssistant: Let me check';
    const result = stripMemoryTags(input);
    expect(result).toBe("User: What is the weather?\n\nAssistant: Let me check");
  });

  it("preserves content without memory tags", () => {
    const input = "User: Hello\nAssistant: Hi there!";
    expect(stripMemoryTags(input)).toBe(input);
  });

  it("strips both tag types in same content", () => {
    const input =
      "A\n<hindsight_memories>\nH mem\n</hindsight_memories>\nB\n<relevant_memories>\nR mem\n</relevant_memories>\nC";
    expect(stripMemoryTags(input)).toBe("A\n\nB\n\nC");
  });

  it("strips tags from a real-world agent conversation with injected memories", () => {
    const input =
      '[role: system]\n<hindsight_memories>\nRelevant memories:\n[{"text": "User prefers dark mode"}]\nUser message: How do I enable dark mode?\n</hindsight_memories>\n[system:end]\n\n[role: user]\nHow do I enable dark mode?\n[user:end]\n\n[role: assistant]\nLet me help you enable dark mode.\n[assistant:end]';

    const result = stripMemoryTags(input);

    expect(result).not.toContain("<hindsight_memories>");
    expect(result).not.toContain("</hindsight_memories>");
    expect(result).not.toContain("User prefers dark mode");
    expect(result).toContain("[role: user]");
    expect(result).toContain("How do I enable dark mode?");
    expect(result).toContain("[role: assistant]");
  });
});

// ---------------------------------------------------------------------------
// stripRuntimeEnvelope
// ---------------------------------------------------------------------------

describe("stripRuntimeEnvelope", () => {
  it("strips leading message id and opaque sender prefix while preserving text", () => {
    const input =
      "[message_id: om_x100b6d3512c5ccb0c084ad240a38842]\n" +
      "ou_cb923a19782fe748cd9fff99454eee31: 我是只retain context的那次改动";

    expect(stripRuntimeEnvelope(input)).toBe("我是只retain context的那次改动");
  });

  it("removes standalone opaque runtime ids", () => {
    const input = "om_x100b6d3512c5ccb0c084ad240a38842\n真实内容\noc_abcdef123456";

    expect(stripRuntimeEnvelope(input)).toBe("真实内容");
  });

  it("does not strip ordinary user text with a colon", () => {
    const input = "计划: 今天修 retain 污染";

    expect(stripRuntimeEnvelope(input)).toBe(input);
  });
});

// ---------------------------------------------------------------------------
// extractRecallQuery
// ---------------------------------------------------------------------------

describe("extractRecallQuery", () => {
  it("returns rawMessage when it is long enough", () => {
    expect(extractRecallQuery("What is my favorite food?", undefined)).toBe(
      "What is my favorite food?"
    );
  });

  it("returns null when rawMessage is too short and prompt is absent", () => {
    expect(extractRecallQuery("Hi", undefined)).toBeNull();
    expect(extractRecallQuery("", "")).toBeNull();
    expect(extractRecallQuery(undefined, undefined)).toBeNull();
  });

  it("returns null when both rawMessage and prompt are too short", () => {
    expect(extractRecallQuery("Hey", "Hey")).toBeNull();
  });

  it("falls back to prompt when rawMessage is absent", () => {
    const result = extractRecallQuery(undefined, "What programming language do I prefer?");
    expect(result).toBe("What programming language do I prefer?");
  });

  it("returns null when rawMessage is absent and prompt is bare metadata", () => {
    const metadataPrompt =
      'Conversation info (untrusted metadata):\n```json\n{"message_id": "abc123"}\n```';
    expect(extractRecallQuery(undefined, metadataPrompt)).toBeNull();
  });

  it("falls back to prompt when rawMessage is metadata but prompt has real content", () => {
    const result = extractRecallQuery(
      "Conversation info (untrusted metadata):",
      "System: You are c0der.\n\nhow many cats do i have?"
    );
    expect(result).toBe("how many cats do i have?");
  });

  it("strips leading System: lines from prompt", () => {
    const prompt = "System: You are an agent.\nSystem: Use tools wisely.\n\nWhat is my name?";
    const result = extractRecallQuery(undefined, prompt);
    expect(result).not.toContain("System:");
    expect(result).toContain("What is my name?");
  });

  it("strips [Channel] envelope header and returns inner message", () => {
    const prompt = "[Telegram Chat]\nWhat is my favorite hobby?";
    const result = extractRecallQuery(undefined, prompt);
    expect(result).toBe("What is my favorite hobby?");
  });

  it("strips [from: SenderName] footer from group chat prompts", () => {
    const prompt = "[Slack Channel #general]\nWhat should I eat for lunch?\n[from: Alice]";
    const result = extractRecallQuery(undefined, prompt);
    expect(result).not.toContain("[from: Alice]");
    expect(result).toContain("What should I eat for lunch?");
  });

  it("strips Feishu runtime ids from recall query text", () => {
    const rawMessage =
      "[message_id: om_x100b6d3512c5ccb0c084ad240a38842]\n" +
      "ou_cb923a19782fe748cd9fff99454eee31: 这个修复是否进入生产？";

    const result = extractRecallQuery(rawMessage, undefined);

    expect(result).toBe("这个修复是否进入生产？");
  });

  it("handles full envelope with System lines, channel header, and from footer", () => {
    const prompt =
      "System: You are a helpful agent.\n\n[Discord Server]\nRemind me what I said about Python?\n[from: Bob]";
    const result = extractRecallQuery(undefined, prompt);
    expect(result).not.toContain("System:");
    expect(result).not.toContain("[Discord");
    expect(result).not.toContain("[from: Bob]");
    expect(result).toContain("Remind me what I said about Python?");
  });

  it("strips session abort hint from prompt", () => {
    const prompt =
      "Note: The previous agent run was aborted by the user\n\n[Telegram]\nWhat is my cat's name?";
    const result = extractRecallQuery(undefined, prompt);
    expect(result).not.toContain("Note: The previous agent run was aborted");
    expect(result).toContain("What is my cat's name?");
  });

  it("returns null when prompt reduces to < 5 chars after stripping", () => {
    // Envelope with almost-empty inner message
    const prompt = "[Telegram Chat]\nHi";
    const result = extractRecallQuery(undefined, prompt);
    expect(result).toBeNull();
  });

  it("prefers rawMessage over prompt even when prompt is longer", () => {
    const rawMessage = "What do I like to eat?";
    const prompt = "[Telegram]\nWhat do I like to eat?\n[from: Alice]";
    const result = extractRecallQuery(rawMessage, prompt);
    // Should return the clean rawMessage verbatim
    expect(result).toBe(rawMessage);
    expect(result).not.toContain("[from: Alice]");
  });

  it("trims whitespace from result", () => {
    const result = extractRecallQuery("   What is my job?   ", undefined);
    expect(result).toBe("What is my job?");
  });

  it("rejects OpenClaw untrusted metadata messages as rawMessage", () => {
    const result = extractRecallQuery("Conversation info (untrusted metadata):", undefined);
    expect(result).toBeNull();
  });

  it("rejects untrusted metadata even when prompt is also metadata", () => {
    const result = extractRecallQuery(
      "Conversation info (untrusted metadata):",
      "Conversation info (untrusted metadata): some details"
    );
    expect(result).toBeNull();
  });

  it("falls back to prompt when rawMessage is metadata", () => {
    const result = extractRecallQuery(
      "Conversation info (untrusted metadata):",
      "How many cats do I have?"
    );
    expect(result).toBe("How many cats do I have?");
  });
});

// ---------------------------------------------------------------------------
// formatMemories
// ---------------------------------------------------------------------------

describe("formatMemories", () => {
  const makeMemoryResult = (overrides: Partial<MemoryResult>): MemoryResult => ({
    id: "mem-1",
    text: "default text",
    type: "world",
    entities: [],
    context: "",
    occurred_start: null,
    occurred_end: null,
    mentioned_at: null,
    document_id: null,
    metadata: null,
    chunk_id: null,
    tags: [],
    ...overrides,
  });

  it("formats memories as a bulleted list", () => {
    const memories: MemoryResult[] = [
      makeMemoryResult({
        id: "1",
        text: "User prefers dark mode",
        type: "world",
        mentioned_at: "2023-01-01T12:00:00Z",
      }),
      makeMemoryResult({
        id: "2",
        text: "User is learning Rust",
        type: "experience",
        mentioned_at: null,
      }),
    ];
    const output = formatMemories(memories);
    expect(output).toBe(
      "- User prefers dark mode [world] (2023-01-01T12:00:00Z)\n\n- User is learning Rust [experience]"
    );
  });

  it("returns empty string for empty memories", () => {
    expect(formatMemories([])).toBe("");
  });
});

// ---------------------------------------------------------------------------
// formatCurrentTimeForRecall — UTC label (#1789, mirrors #1568)
// ---------------------------------------------------------------------------

describe("formatCurrentTimeForRecall", () => {
  it("renders the UTC suffix so the LLM doesn't misread it as local time", () => {
    const fixed = new Date(Date.UTC(2026, 4, 27, 16, 25, 0)); // 2026-05-27 16:25 UTC
    expect(formatCurrentTimeForRecall(fixed)).toBe("2026-05-27 16:25 UTC");
  });

  it("zero-pads month / day / hours / minutes", () => {
    const fixed = new Date(Date.UTC(2026, 0, 3, 4, 5, 0));
    expect(formatCurrentTimeForRecall(fixed)).toBe("2026-01-03 04:05 UTC");
  });
});

// ---------------------------------------------------------------------------
// retention helpers
// ---------------------------------------------------------------------------

describe("countUserTurns", () => {
  it("counts user messages across a resumed conversation history", () => {
    expect(
      countUserTurns([
        { role: "user", content: "turn 1" },
        { role: "assistant", content: "reply 1" },
        { role: "system", content: "meta" },
        { role: "user", content: "turn 2" },
        { role: "assistant", content: "reply 2" },
        { role: "user", content: "turn 3" },
      ])
    ).toBe(3);
  });
});

describe("getRetentionTurnIndex", () => {
  it("uses the full conversation turn count for per-turn retention", () => {
    expect(getRetentionTurnIndex(7, 1)).toBe(7);
  });

  it("derives a stable window sequence for chunked retention", () => {
    expect(getRetentionTurnIndex(6, 3)).toBe(2);
  });

  it("returns null when a chunk boundary has not been reached", () => {
    expect(getRetentionTurnIndex(5, 3)).toBeNull();
  });
});

describe("normalizeRetainTags", () => {
  it("trims, deduplicates, and preserves order for string arrays", () => {
    expect(
      normalizeRetainTags([" source_system:openclaw ", "agent:main", "agent:main", ""])
    ).toEqual(["source_system:openclaw", "agent:main"]);
  });

  it("drops non-string values instead of stringifying them", () => {
    expect(
      normalizeRetainTags([
        "agent:main",
        { a: 1 } as unknown as string,
        42 as unknown as string,
        null as unknown as string,
      ])
    ).toEqual(["agent:main"]);
  });

  it("accepts comma-separated strings", () => {
    expect(normalizeRetainTags(" source_system:openclaw, agent:main , agent:main ")).toEqual([
      "source_system:openclaw",
      "agent:main",
    ]);
  });
});

describe("inline retain tag helpers", () => {
  it("extracts retain tags from inline directives", () => {
    expect(
      extractInlineRetainTags(
        "hello <retain_tags> client:acme, type:decision, client:acme </retain_tags> world"
      )
    ).toEqual(["client:acme", "type:decision"]);
  });

  it("supports hindsight_retain_tags alias and strips directives from content", () => {
    const input =
      "Keep this.\n<hindsight_retain_tags>scope:user</hindsight_retain_tags>\nNot the directive.";
    expect(extractInlineRetainTags(input)).toEqual(["scope:user"]);
    expect(stripInlineRetainTags(input)).toBe("Keep this.\n\nNot the directive.");
  });
});

describe("buildRetainRequest", () => {
  it("uses session-scoped doc id + update_mode=append when API supports it", () => {
    const request = buildRetainRequest(
      "hello world",
      2,
      {
        agentId: "main",
        sessionKey: "agent:main:main",
        messageProvider: "discord",
        channelId: "channel:123",
        senderId: "user:456",
      },
      {
        retainSource: "openclaw",
        retainTags: ["source_system:openclaw", "agent:agentname"],
      },
      1700000000000,
      { appendSupported: true }
    );

    expect(request).toEqual({
      content: "hello world",
      documentId: "openclaw:agent:main:main",
      context: DEFAULT_RETAIN_CONTEXT,
      metadata: {
        retained_at: expect.any(String),
        message_count: "2",
        source: "openclaw",
        retention_scope: "turn",
        turn_index: "1",
        session_key: "agent:main:main",
        agent_id: "main",
        provider: "discord",
        channel_type: "discord",
        channel_id: "channel:123",
        thread_id: undefined,
        sender_id: "user:456",
      },
      tags: ["source_system:openclaw", "agent:agentname"],
      updateMode: "append",
    });
  });

  it("includes the default retain context guidance", () => {
    const request = buildRetainRequest("hello world", 1, {}, {}, 1700000000000, { turnIndex: 1 });

    expect(request.context).toBe(DEFAULT_RETAIN_CONTEXT);
  });

  it("keeps sender/channel/provider in retain metadata", () => {
    const request = buildRetainRequest(
      "hello world",
      1,
      {
        sessionKey: "agent:main:feishu:oc_abcdef123456",
        messageProvider: "feishu",
        channelId: "oc_abcdef123456",
        senderId: "ou_cb923a19782fe748cd9fff99454eee31",
      },
      {},
      1700000000000,
      { turnIndex: 1 }
    );

    expect(request.metadata).toMatchObject({
      provider: "feishu",
      channel_id: "oc_abcdef123456",
      sender_id: "ou_cb923a19782fe748cd9fff99454eee31",
    });
  });

  it("describes routing metadata and assistant/user roles in the default retain context", () => {
    expect(DEFAULT_RETAIN_CONTEXT).toContain("routing identifiers");
    expect(DEFAULT_RETAIN_CONTEXT).toContain("operational routing identifiers");
    expect(DEFAULT_RETAIN_CONTEXT).toContain("AI assistant");
  });

  it("uses a configured retain context when provided", () => {
    const request = buildRetainRequest(
      "hello world",
      1,
      {},
      { retainContext: "Custom extraction guidance." },
      1700000000000,
      { turnIndex: 1 }
    );

    expect(request.context).toBe("Custom extraction guidance.");
  });

  it("trims configured retain context before sending it", () => {
    const request = buildRetainRequest(
      "hello world",
      1,
      {},
      { retainContext: "  Custom extraction guidance. \n" },
      1700000000000,
      { turnIndex: 1 }
    );

    expect(request.context).toBe("Custom extraction guidance.");
  });

  it("falls back to per-turn doc id when appendSupported is false (older API)", () => {
    const request = buildRetainRequest(
      "hello world",
      2,
      {
        agentId: "main",
        sessionKey: "agent:main:main",
        messageProvider: "discord",
      },
      { retainSource: "openclaw" },
      1700000000000,
      { turnIndex: 4, appendSupported: false }
    );
    expect(request.documentId).toBe("openclaw:agent:main:main:turn:000004");
    expect(request.updateMode).toBeUndefined();
  });

  it("defaults to per-turn fallback when appendSupported flag is omitted (conservative)", () => {
    const request = buildRetainRequest(
      "hello world",
      2,
      {
        agentId: "main",
        sessionKey: "agent:main:main",
        messageProvider: "discord",
      },
      { retainSource: "openclaw" },
      1700000000000,
      { turnIndex: 6 }
    );
    expect(request.documentId).toBe("openclaw:agent:main:main:turn:000006");
    expect(request.updateMode).toBeUndefined();
  });

  it("uses window ids and metadata for chunked retention", () => {
    const request = buildRetainRequest(
      "hello world",
      4,
      {
        agentId: "agentname",
        sessionKey: "agent:agentname:discord:group:123:topic:456",
        messageProvider: "discord",
        senderId: "user:456",
      },
      {
        retainSource: "openclaw",
      },
      1700000000000,
      {
        retentionScope: "window",
        turnIndex: 2,
        windowTurns: 2,
      }
    );

    expect(request.documentId).toBe(
      "openclaw:agent:agentname:discord:group:123:topic:456:window:000002"
    );
    expect(request.metadata).toMatchObject({
      source: "openclaw",
      retention_scope: "window",
      turn_index: "2",
      agent_id: "agentname",
      provider: "discord",
      channel_type: "discord",
      channel_id: "group:123:topic:456",
      thread_id: "456",
      sender_id: "user:456",
      window_turns: "2",
    });
  });

  it("merges configured retain tags with inline per-message tags", () => {
    const request = buildRetainRequest(
      "hello world",
      1,
      {},
      {
        retainTags: ["source_system:openclaw", "agent:main"],
      },
      1700000000000,
      {
        turnIndex: 1,
        tags: ["client:acme", "agent:main"],
      }
    );

    expect(request.tags).toEqual(["source_system:openclaw", "agent:main", "client:acme"]);
  });

  it("defaults source metadata to openclaw when unset", () => {
    const request = buildRetainRequest("hello world", 1, {}, {}, 1700000000000, { turnIndex: 1 });
    expect(request.metadata?.source).toBe("openclaw");
    expect(request.tags).toBeUndefined();
  });

  it("preserves provider fallback without backfilling channel_type from the session key", () => {
    const request = buildRetainRequest(
      "hello world",
      1,
      {
        sessionKey: "agent:main:telegram:direct:12345",
      },
      {},
      1700000000000,
      { turnIndex: 1 }
    );

    expect(request.metadata).toMatchObject({
      provider: "telegram",
      channel_type: undefined,
      channel_id: "direct:12345",
      sender_id: "12345",
    });
  });
});

describe("stripInlineTimestampPrefix", () => {
  it("strips weekday/date/time/GMT offset prefixes", () => {
    expect(stripInlineTimestampPrefix("[Wed 2026-04-15 10:44 GMT+2] hello")).toBe("hello");
    expect(stripInlineTimestampPrefix("[Mon 2026-01-05 9:07 GMT-5] x")).toBe("x");
    expect(stripInlineTimestampPrefix("[Sun 2025-12-07 23:59:30 UTC] y")).toBe("y");
  });

  it("leaves unrelated content untouched", () => {
    expect(stripInlineTimestampPrefix("just text")).toBe("just text");
    expect(stripInlineTimestampPrefix("[Random] not a timestamp")).toBe("[Random] not a timestamp");
  });
});

// ---------------------------------------------------------------------------
// prepareRetentionTranscript
// ---------------------------------------------------------------------------

describe("prepareRetentionTranscript", () => {
  const baseConfig: PluginConfig = {
    dynamicBankId: true,
    retainRoles: ["user", "assistant"],
  };

  it("lifts message timestamps into a structured field and strips inline prefix (json+toolcalls)", () => {
    const messages = [
      {
        role: "user",
        timestamp: 1776246240000,
        content: [{ type: "text", text: "[Wed 2026-04-15 10:44 GMT+2] just pick some news" }],
      },
      {
        role: "assistant",
        timestamp: 1776246243000,
        content: [{ type: "text", text: "Got it." }],
      },
    ];
    const result = prepareRetentionTranscript(messages, baseConfig);
    expect(result).not.toBeNull();
    const parsed = JSON.parse(result!.transcript);
    expect(parsed).toEqual([
      {
        role: "user",
        content: [{ type: "text", text: "just pick some news" }],
        timestamp: "2026-04-15T09:44:00.000Z",
      },
      {
        role: "assistant",
        content: [{ type: "text", text: "Got it." }],
        timestamp: "2026-04-15T09:44:03.000Z",
      },
    ]);
  });

  it("lifts message timestamps into a structured field (json without toolcalls)", () => {
    const config: PluginConfig = { ...baseConfig, retainToolCalls: false };
    const messages = [
      {
        role: "user",
        timestamp: 1776246240000,
        content: "[Wed 2026-04-15 10:44 GMT+2] hi there",
      },
    ];
    const result = prepareRetentionTranscript(messages, config);
    expect(result).not.toBeNull();
    const parsed = JSON.parse(result!.transcript);
    expect(parsed).toEqual([
      { role: "user", content: "hi there", timestamp: "2026-04-15T09:44:00.000Z" },
    ]);
  });

  it("returns null if no user message found (turn boundary)", () => {
    const messages = [
      { role: "assistant", content: "Hello" },
      { role: "system", content: "Context" },
    ];
    const result = prepareRetentionTranscript(messages, baseConfig);
    expect(result).toBeNull();
  });

  it("retains from last user message onwards", () => {
    const messages = [
      { role: "user", content: "Old user" },
      { role: "assistant", content: "Old assistant" },
      { role: "user", content: "New user" },
      { role: "assistant", content: "New assistant" },
    ];
    const result = prepareRetentionTranscript(messages, baseConfig);
    expect(result).not.toBeNull();
    expect(result?.transcript).toContain("New user");
    expect(result?.transcript).toContain("New assistant");
    expect(result?.transcript).not.toContain("Old user");
  });

  it("filters out excluded roles", () => {
    const config: PluginConfig = { ...baseConfig, retainRoles: ["user"] };
    const messages = [
      { role: "user", content: "User msg" },
      { role: "assistant", content: "Assistant msg" },
    ];
    const result = prepareRetentionTranscript(messages, config);
    expect(result).not.toBeNull();
    expect(result?.transcript).toContain("User msg");
    expect(result?.transcript).not.toContain("Assistant msg");
  });

  it("handles array content", () => {
    const messages = [{ role: "user", content: [{ type: "text", text: "Hello array" }] }];
    const result = prepareRetentionTranscript(messages, baseConfig);
    expect(result?.transcript).toContain("Hello array");
  });

  it("strips memory tags from retained content (feedback loop prevention)", () => {
    const messages = [
      { role: "user", content: "What is dark mode?" },
      {
        role: "assistant",
        content:
          "<hindsight_memories>\nUser prefers dark mode\n</hindsight_memories>\nHere is how to enable dark mode.",
      },
    ];
    const result = prepareRetentionTranscript(messages, baseConfig);
    expect(result).not.toBeNull();
    expect(result?.transcript).not.toContain("<hindsight_memories>");
    expect(result?.transcript).not.toContain("User prefers dark mode");
    expect(result?.transcript).toContain("Here is how to enable dark mode.");
  });

  it("strips inline retain-tag directives from retained content", () => {
    const messages = [
      {
        role: "user",
        content:
          "Remember this.\n<retain_tags>client:acme, type:decision</retain_tags>\nActual content.",
      },
      { role: "assistant", content: "Got it." },
    ];
    const result = prepareRetentionTranscript(messages, baseConfig);
    expect(result).not.toBeNull();
    expect(result?.transcript).toContain("Remember this.");
    expect(result?.transcript).toContain("Actual content.");
    expect(result?.transcript).not.toContain("<retain_tags>");
    expect(result?.transcript).not.toContain("client:acme");
  });

  it("strips OpenClaw metadata and Feishu runtime headers from retained structured content", () => {
    const messages = [
      {
        role: "user",
        content:
          'Conversation info (untrusted metadata):\n```json\n{"message_id":"om_x100b6d3512c5ccb0c084ad240a38842","sender_id":"ou_cb923a19782fe748cd9fff99454eee31"}\n```\n' +
          "[message_id: om_x100b6d3512c5ccb0c084ad240a38842]\n" +
          "ou_cb923a19782fe748cd9fff99454eee31: 我是只retain context的那次改动",
      },
    ];

    const result = prepareRetentionTranscript(messages, baseConfig);

    expect(result).not.toBeNull();
    expect(JSON.parse(result!.transcript)).toEqual([
      {
        role: "user",
        content: [{ type: "text", text: "我是只retain context的那次改动" }],
      },
    ]);
    expect(result!.transcript).not.toContain("Conversation info");
    expect(result!.transcript).not.toContain("message_id");
    expect(result!.transcript).not.toContain("om_x100b6d3512c5ccb0c084ad240a38842");
    expect(result!.transcript).not.toContain("ou_cb923a19782fe748cd9fff99454eee31");
  });

  it("strips Feishu runtime headers from retained text-only content", () => {
    const config: PluginConfig = { ...baseConfig, retainToolCalls: false };
    const messages = [
      {
        role: "user",
        content:
          "[message_id: om_x100b6d3512c5ccb0c084ad240a38842]\n" +
          "ou_cb923a19782fe748cd9fff99454eee31: 我是只retain context的那次改动",
      },
    ];

    const result = prepareRetentionTranscript(messages, config);

    expect(result).not.toBeNull();
    expect(JSON.parse(result!.transcript)).toEqual([
      { role: "user", content: "我是只retain context的那次改动" },
    ]);
  });

  it("strips runtime headers that appear after an inline timestamp", () => {
    const messages = [
      {
        role: "user",
        content:
          "[Wed 2026-06-03 19:54 GMT+8]\n" +
          "[message_id: om_x100b6d3512c5ccb0c084ad240a38842]\n" +
          "ou_cb923a19782fe748cd9fff99454eee31: 我是只retain context的那次改动",
      },
    ];

    const result = prepareRetentionTranscript(messages, baseConfig);

    expect(result).not.toBeNull();
    expect(JSON.parse(result!.transcript)).toEqual([
      {
        role: "user",
        content: [{ type: "text", text: "我是只retain context的那次改动" }],
      },
    ]);
  });

  it("strips memory tags from user message when prependContext is prepended to it", () => {
    // Simulates the host prepending prependContext to the user message content
    const userContent = `<hindsight_memories>\nRelevant memories:\n- User prefers dark mode [world]\n\nUser message: What is dark mode?\n</hindsight_memories>\nWhat is dark mode?`;
    const messages = [
      { role: "user", content: userContent },
      { role: "assistant", content: "Dark mode is a display setting." },
    ];
    const result = prepareRetentionTranscript(messages, baseConfig);
    expect(result).not.toBeNull();
    expect(result?.transcript).not.toContain("<hindsight_memories>");
    expect(result?.transcript).not.toContain("User prefers dark mode");
    expect(result?.transcript).toContain("What is dark mode?");
    expect(result?.transcript).toContain("Dark mode is a display setting.");
  });

  it("emits Anthropic-shaped typed blocks by default (retainToolCalls=true)", () => {
    const messages = [
      { role: "user", content: "Hello there" },
      { role: "assistant", content: "Hi back" },
    ];
    const result = prepareRetentionTranscript(messages, baseConfig);
    expect(result).not.toBeNull();
    const parsed = JSON.parse(result!.transcript);
    expect(parsed).toEqual([
      { role: "user", content: [{ type: "text", text: "Hello there" }] },
      { role: "assistant", content: [{ type: "text", text: "Hi back" }] },
    ]);
    expect(result!.transcript).not.toContain("[role:");
  });

  it("flattens content to a string when retainToolCalls is false", () => {
    const config: PluginConfig = { ...baseConfig, retainToolCalls: false };
    const messages = [
      { role: "user", content: "Hello there" },
      { role: "assistant", content: "Hi back" },
    ];
    const result = prepareRetentionTranscript(messages, config);
    expect(JSON.parse(result!.transcript)).toEqual([
      { role: "user", content: "Hello there" },
      { role: "assistant", content: "Hi back" },
    ]);
  });

  it("retains assistant tool_use blocks and folds toolResult into a user tool_result block", () => {
    const messages = [
      { role: "user", content: [{ type: "text", text: "What is the weather?" }] },
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "deliberation — should be stripped" },
          { type: "text", text: "Let me check." },
          { type: "toolCall", id: "call_abc", name: "get_weather", arguments: { city: "SF" } },
        ],
      },
      {
        role: "toolResult",
        toolCallId: "call_abc",
        toolName: "get_weather",
        content: [{ type: "text", text: "sunny, 62F" }],
      },
      { role: "assistant", content: [{ type: "text", text: "It's sunny, 62F." }] },
    ];
    const result = prepareRetentionTranscript(messages, baseConfig);
    expect(result).not.toBeNull();
    const parsed = JSON.parse(result!.transcript);
    expect(parsed).toEqual([
      { role: "user", content: [{ type: "text", text: "What is the weather?" }] },
      {
        role: "assistant",
        content: [
          { type: "text", text: "Let me check." },
          { type: "tool_use", name: "get_weather", input: { city: "SF" }, id: "call_abc" },
        ],
      },
      {
        role: "user",
        content: [{ type: "tool_result", content: "sunny, 62F", tool_use_id: "call_abc" }],
      },
      { role: "assistant", content: [{ type: "text", text: "It's sunny, 62F." }] },
    ]);
  });

  it("filters operational MCP tool calls to avoid feedback loops", () => {
    const messages = [
      { role: "user", content: "recall stuff" },
      {
        role: "assistant",
        content: [
          { type: "toolCall", id: "c1", name: "mcp__hindsight__recall", arguments: { query: "x" } },
          {
            type: "toolCall",
            id: "c2",
            name: "mcp__other__send_message",
            arguments: { text: "hi" },
          },
          { type: "text", text: "Done." },
        ],
      },
    ];
    const result = prepareRetentionTranscript(messages, baseConfig);
    const parsed = JSON.parse(result!.transcript);
    const assistantBlocks = parsed[1].content;
    expect(
      assistantBlocks.some((b: any) => b.type === "tool_use" && b.name === "mcp__hindsight__recall")
    ).toBe(false);
    expect(
      assistantBlocks.some(
        (b: any) => b.type === "tool_use" && b.name === "mcp__other__send_message"
      )
    ).toBe(true);
  });

  it("truncates tool_result content at 2000 chars", () => {
    const big = "x".repeat(3000);
    const messages = [
      { role: "user", content: "run tool" },
      { role: "assistant", content: [{ type: "toolCall", id: "c1", name: "noop", arguments: {} }] },
      { role: "toolResult", toolCallId: "c1", content: [{ type: "text", text: big }] },
    ];
    const result = prepareRetentionTranscript(messages, baseConfig);
    const parsed = JSON.parse(result!.transcript);
    const toolResult = parsed.find((m: any) => m.content.some((b: any) => b.type === "tool_result"))
      .content[0];
    expect(toolResult.content.endsWith("... (truncated)")).toBe(true);
    expect(toolResult.content.length).toBe(2000 + "... (truncated)".length);
  });

  it('emits legacy text markers when retainFormat is "text"', () => {
    const config: PluginConfig = { ...baseConfig, retainFormat: "text" };
    const messages = [
      { role: "user", content: "Hello there" },
      { role: "assistant", content: "Hi back" },
    ];
    const result = prepareRetentionTranscript(messages, config);
    expect(result).not.toBeNull();
    expect(result!.transcript).toContain("[role: user]\nHello there\n[user:end]");
    expect(result!.transcript).toContain("[role: assistant]\nHi back\n[assistant:end]");
  });

  it("reports accurate messageCount excluding empty messages", () => {
    const messages = [
      { role: "user", content: "Real message" },
      { role: "assistant", content: "<hindsight_memories>\nonly tags\n</hindsight_memories>" },
      { role: "assistant", content: "Actual response" },
    ];
    const result = prepareRetentionTranscript(messages, baseConfig);
    expect(result).not.toBeNull();
    // The middle message becomes empty after tag stripping, so messageCount should be 2
    expect(result?.messageCount).toBe(2);
  });

  it("does not wrap retained JSON content in a context header", () => {
    const config: PluginConfig = { ...baseConfig, retainToolCalls: false };
    const messages = [{ role: "user", content: "What's MIN-123 status?" }];
    const result = prepareRetentionTranscript(messages, config);
    expect(result).not.toBeNull();
    const parsed = JSON.parse(result!.transcript);
    expect(parsed).toEqual([{ role: "user", content: "What's MIN-123 status?" }]);
    expect(result!.transcript).not.toContain("[context]");
    expect(result!.transcript).not.toContain("sender:");
    expect(result!.transcript).not.toContain("channel:");
    expect(result!.transcript).not.toContain("provider:");
    expect(result?.messageCount).toBe(1);
  });

  it("does not wrap retained text content in a context header", () => {
    const config: PluginConfig = { ...baseConfig, retainFormat: "text" };
    const messages = [{ role: "user", content: "ping" }];
    const result = prepareRetentionTranscript(messages, config);
    expect(result).not.toBeNull();
    expect(result!.transcript).not.toContain("[context]");
    expect(result!.transcript).not.toContain("sender:");
    expect(result!.transcript).not.toContain("channel:");
    expect(result!.transcript).not.toContain("provider:");
    expect(result!.transcript).toContain("[role: user]\nping\n[user:end]");
  });
});

// ---------------------------------------------------------------------------
// sliceLastTurnsByUserBoundary
// ---------------------------------------------------------------------------

describe("sliceLastTurnsByUserBoundary", () => {
  it("returns the whole message list when requested turns exceed available user turns", () => {
    const messages = [
      { role: "system", content: "System preface" },
      { role: "user", content: "Turn 1 user" },
      { role: "assistant", content: "Turn 1 assistant" },
      { role: "user", content: "Turn 2 user" },
      { role: "assistant", content: "Turn 2 assistant" },
    ];

    const result = sliceLastTurnsByUserBoundary(messages, 3);
    expect(result).toEqual(messages);
  });

  it("slices by real user-turn boundaries with system/tool messages present", () => {
    const messages = [
      { role: "system", content: "System preface" },
      { role: "user", content: "Turn 1 user" },
      { role: "assistant", content: "Turn 1 assistant" },
      { role: "tool", content: "Tool output in turn 1" },
      { role: "user", content: "Turn 2 user" },
      { role: "assistant", content: "Turn 2 assistant" },
      { role: "system", content: "System note in turn 2" },
      { role: "user", content: "Turn 3 user" },
      { role: "assistant", content: "Turn 3 assistant" },
    ];

    const result = sliceLastTurnsByUserBoundary(messages, 2);
    expect(result).toEqual(messages.slice(4));
  });

  it("returns empty list for invalid turn counts", () => {
    const messages = [{ role: "user", content: "Hello" }];
    expect(sliceLastTurnsByUserBoundary(messages, 0)).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// composeRecallQuery + truncateRecallQuery
// ---------------------------------------------------------------------------

describe("composeRecallQuery", () => {
  it("returns latest query unchanged when recallContextTurns is 1", () => {
    const query = composeRecallQuery(
      "What is my preference?",
      [{ role: "user", content: "Old message" }],
      1
    );
    expect(query).toBe("What is my preference?");
  });

  it("includes prior user/assistant context when recallContextTurns > 1", () => {
    const messages = [
      { role: "user", content: "I like dark mode." },
      { role: "assistant", content: "Got it, dark mode noted." },
      { role: "user", content: "What theme do I prefer?" },
    ];

    const query = composeRecallQuery("What theme do I prefer?", messages, 2);
    expect(query).toContain("What theme do I prefer?");
    expect(query).toContain("user: I like dark mode.");
    expect(query).toContain("assistant: Got it, dark mode noted.");
    // latest message should appear after prior context
    expect(query.indexOf("Prior context:")).toBeLessThan(query.indexOf("What theme do I prefer?"));
  });

  it("strips Feishu runtime ids from prior recall context", () => {
    const messages = [
      {
        role: "user",
        content:
          "[message_id: om_x100b6d3512c5ccb0c084ad240a38842]\n" +
          "ou_cb923a19782fe748cd9fff99454eee31: 我在修 retain 污染",
      },
      { role: "assistant", content: "收到。" },
      { role: "user", content: "现在状态呢？" },
    ];

    const query = composeRecallQuery("现在状态呢？", messages, 2);

    expect(query).toContain("user: 我在修 retain 污染");
    expect(query).not.toContain("message_id");
    expect(query).not.toContain("om_x100b6d3512c5ccb0c084ad240a38842");
    expect(query).not.toContain("ou_cb923a19782fe748cd9fff99454eee31");
  });

  it("respects recallRoles when building prior context", () => {
    const messages = [
      { role: "system", content: "System context" },
      { role: "assistant", content: "Assistant context" },
      { role: "user", content: "What theme do I prefer?" },
    ];

    const query = composeRecallQuery("What theme do I prefer?", messages, 2, ["user"]);
    expect(query).toBe("What theme do I prefer?");
  });

  it("falls back to latest query when context has no usable text", () => {
    const messages = [{ role: "tool", content: "binary blob" }];
    const query = composeRecallQuery("Summarize my preference", messages, 3);
    expect(query).toBe("Summarize my preference");
  });
});

describe("truncateRecallQuery", () => {
  it("keeps query unchanged when under max", () => {
    const query = "short query";
    expect(truncateRecallQuery(query, query, 100)).toBe(query);
  });

  it("falls back to latest query when non-context query is over max", () => {
    const latest = "What foods do I like?";
    const long = `${latest} ${"x".repeat(300)}`;
    expect(truncateRecallQuery(long, latest, 20)).toBe(latest.slice(0, 20));
  });

  it("trims prior context first and preserves latest section", () => {
    const latest = "What foods do I like?";
    const composed = [
      "Prior context:",
      "user: I like sushi.",
      "assistant: You like sushi and ramen.",
      "user: Also pizza.",
      latest,
    ].join("\n\n");

    const truncated = truncateRecallQuery(composed, latest, 180);
    expect(truncated).toContain(latest);
    expect(truncated.length).toBeLessThanOrEqual(180);
  });
});

// ---------------------------------------------------------------------------
// session identity + operational guardrails
// ---------------------------------------------------------------------------

describe("session identity helpers", () => {
  const baseConfig: PluginConfig = {
    dynamicBankId: true,
    dynamicBankGranularity: ["agent", "channel", "user"],
  };

  it("parses main sessions", () => {
    expect(parseSessionKey("agent:main:main")).toEqual({
      agentId: "main",
      provider: "main",
      channel: "main",
    });
  });

  it("parses operational cron-like sessions", () => {
    expect(parseSessionKey("agent:worker:cron:nightly:cleanup")).toEqual({
      agentId: "worker",
      provider: "cron",
      channel: "nightly:cleanup",
    });
  });

  it("extracts telegram direct sender ids from channel ids", () => {
    expect(extractTelegramDirectSenderId("direct:12345")).toBe("12345");
    expect(extractTelegramDirectSenderId("group:12345")).toBeUndefined();
  });

  it("resolves telegram direct identity from session key when senderId is missing", () => {
    const resolved = resolveSessionIdentity({
      agentId: "main",
      sessionKey: "agent:main:telegram:direct:12345",
    });

    expect(resolved).toMatchObject({
      agentId: "main",
      messageProvider: "telegram",
      channelId: "direct:12345",
      senderId: "12345",
    });
  });

  it("resolves msteams direct identity from session key when senderId is missing", () => {
    const resolved = resolveSessionIdentity({
      agentId: "nemoclaw",
      sessionKey: "agent:nemoclaw:msteams:direct:user-teams-42",
    });

    expect(resolved).toMatchObject({
      agentId: "nemoclaw",
      messageProvider: "msteams",
      channelId: "direct:user-teams-42",
      senderId: "user-teams-42",
    });
  });

  it("derives bank ids from resolved telegram direct identity", () => {
    const bankId = deriveBankId(
      {
        agentId: "main",
        sessionKey: "agent:main:telegram:direct:12345",
      },
      baseConfig
    );

    expect(bankId).toBe("main::direct%3A12345::12345");
  });

  it("allows agent:*:main sessions by default (default granularity includes 'agent')", () => {
    const result = getIdentitySkipReason({ sessionKey: "agent:main:main" });
    expect(result.reason).toBeUndefined();
    expect(result.resolvedCtx?.senderId).toBe("agent-user:main");
  });

  it.each([
    "agent:worker:cron:nightly:cleanup",
    "agent:worker:heartbeat:node-1",
    "agent:worker:subagent:abc123",
  ])("marks operational sessions as final skips: %s", (sessionKey) => {
    const result = getIdentitySkipReason({ sessionKey });
    expect(result.reason).toEqual({
      kind: "final",
      detail: `operational session ${sessionKey}`,
    });
  });

  it("marks temp sessions as final skips", () => {
    const result = getIdentitySkipReason({ sessionKey: "temp:compose:123" });
    expect(result.reason).toEqual({
      kind: "final",
      detail: "ephemeral temp session temp:compose:123",
    });
  });

  it("marks missing provider as retryable", () => {
    const result = getIdentitySkipReason({ senderId: "12345" });
    expect(result.reason).toEqual({
      kind: "retryable",
      detail: "missing stable message provider",
    });
  });

  it("marks missing sender as retryable", () => {
    const result = getIdentitySkipReason({ messageProvider: "telegram", channelId: "group:12345" });
    expect(result.reason).toEqual({
      kind: "retryable",
      detail: "missing stable sender identity",
    });
  });

  it("marks telegram direct sender mismatches as final skips", () => {
    const result = getIdentitySkipReason({
      sessionKey: "agent:main:telegram:direct:12345",
      messageProvider: "telegram",
      channelId: "direct:12345",
      senderId: "99999",
    });

    expect(result.reason).toEqual({
      kind: "final",
      detail: "telegram direct identity mismatch (direct:12345 vs 99999)",
    });
  });

  it("allows agent:*:main sessions through when agent banking is enabled", () => {
    const result = getIdentitySkipReason(
      { sessionKey: "agent:project-alpha:main" },
      { dynamicBankGranularity: ["agent"] }
    );
    expect(result.reason).toBeUndefined();
    expect(result.resolvedCtx?.agentId).toBe("project-alpha");
    expect(result.resolvedCtx?.senderId).toBe("agent-user:project-alpha");
  });

  it("allows provider main when agent banking is enabled", () => {
    const result = getIdentitySkipReason(
      { sessionKey: "agent:main:main" },
      { dynamicBankGranularity: ["agent"] }
    );
    expect(result.reason).toBeUndefined();
  });

  it("still skips cron/heartbeat/subagent providers when agent banking is enabled", () => {
    const result = getIdentitySkipReason(
      { sessionKey: "agent:main:cron:nightly:cleanup" },
      { dynamicBankGranularity: ["agent"] }
    );
    expect(result.reason).toEqual({
      kind: "final",
      detail: "operational session agent:main:cron:nightly:cleanup",
    });
  });

  it("synthesizes sender identity for anonymous CLI sessions when agent banking is enabled", () => {
    const result = getIdentitySkipReason(
      { agentId: "project-beta", messageProvider: "cli", senderId: "anonymous" },
      { dynamicBankGranularity: ["agent"] }
    );
    expect(result.reason).toBeUndefined();
    expect(result.resolvedCtx?.senderId).toBe("agent-user:project-beta");
  });

  it("allows agent:*:main sessions through when a static bankId is configured", () => {
    const result = getIdentitySkipReason(
      { sessionKey: "agent:main:main" },
      { dynamicBankId: false, bankId: "shared-bank" }
    );
    expect(result.reason).toBeUndefined();
    expect(result.resolvedCtx?.senderId).toBe("agent-user:main");
  });

  it("allows agent:*:main when dynamicBankId is false but bankId is missing (default granularity includes 'agent')", () => {
    const result = getIdentitySkipReason(
      { sessionKey: "agent:main:main" },
      { dynamicBankId: false }
    );
    // Default agentBanking is true (default granularity includes 'agent'),
    // so the session is allowed even without an explicit bankId.
    expect(result.reason).toBeUndefined();
  });

  it("does not broaden the carve-out when granularity excludes 'agent' and bankId is missing", () => {
    const result = getIdentitySkipReason(
      { sessionKey: "agent:main:main" },
      { dynamicBankId: false, dynamicBankGranularity: ["channel", "user"] }
    );
    expect(result.reason).toEqual({
      kind: "final",
      detail: "internal main session agent:main:main",
    });
  });

  it("allows agent:*:main sessions with empty config (default granularity includes 'agent')", () => {
    const result = getIdentitySkipReason({ sessionKey: "agent:main:main" }, {});
    expect(result.reason).toBeUndefined();
    expect(result.resolvedCtx?.senderId).toBe("agent-user:main");
  });

  it("skips agent:*:main sessions when granularity explicitly excludes 'agent'", () => {
    const result = getIdentitySkipReason(
      { sessionKey: "agent:main:main" },
      { dynamicBankGranularity: ["channel", "user"] }
    );
    expect(result.reason).toEqual({
      kind: "final",
      detail: "internal main session agent:main:main",
    });
  });

  it("detects ephemeral operational text with or without transcript wrappers", () => {
    expect(isEphemeralOperationalText("A new session was started via /reset.")).toBe(true);
    expect(
      isEphemeralOperationalText("[role: user]\nA new session was started via /new.\n[user:end]")
    ).toBe(true);
    expect(isEphemeralOperationalText("Tell me what I said about dark mode.")).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// resolveAndCacheIdentity — dispatch-surface gate (#1541)
// ---------------------------------------------------------------------------

describe("resolveAndCacheIdentity dispatch-surface gate (#1541)", () => {
  it("does not skip when session provider is synthetic 'main' (telegram dispatch)", () => {
    const { resolvedCtx, skipReason } = resolveAndCacheIdentity({
      sessionKey: "agent:bug1541-tg:main",
      ctx: { sessionKey: "agent:bug1541-tg:main" },
      dispatchChannel: "telegram",
    });

    expect(skipReason).toBeUndefined();
    expect(resolvedCtx?.messageProvider).toBe("telegram");
  });

  it("does not skip when session provider is synthetic 'main' (webchat dispatch)", () => {
    const { skipReason } = resolveAndCacheIdentity({
      sessionKey: "agent:bug1541-wc:main",
      ctx: { sessionKey: "agent:bug1541-wc:main" },
      dispatchChannel: "webchat",
    });

    expect(skipReason).toBeUndefined();
  });

  it("does not skip when a static bankId is configured (feishu session via webchat)", () => {
    const { skipReason } = resolveAndCacheIdentity({
      sessionKey: "agent:main:feishu:direct:user-1541-static",
      ctx: {
        sessionKey: "agent:main:feishu:direct:user-1541-static",
        senderId: "user-1541-static",
      },
      dispatchChannel: "webchat",
      pluginConfig: { dynamicBankId: false, bankId: "shared-bank" },
    });

    expect(skipReason).toBeUndefined();
  });

  it("does not skip when granularity excludes channel and provider (qqbot via webchat)", () => {
    const { skipReason } = resolveAndCacheIdentity({
      sessionKey: "agent:main:qqbot:direct:user-1541-agentonly",
      ctx: {
        sessionKey: "agent:main:qqbot:direct:user-1541-agentonly",
        senderId: "user-1541-agentonly",
      },
      dispatchChannel: "webchat",
      pluginConfig: { dynamicBankGranularity: ["agent", "user"] },
    });

    expect(skipReason).toBeUndefined();
  });

  it("still skips real-provider mismatch with default granularity (qqbot via webchat)", () => {
    const { skipReason } = resolveAndCacheIdentity({
      sessionKey: "agent:main:qqbot:direct:user-1541-default",
      ctx: {
        sessionKey: "agent:main:qqbot:direct:user-1541-default",
        senderId: "user-1541-default",
      },
      dispatchChannel: "webchat",
    });

    expect(skipReason).toEqual({
      kind: "final",
      detail: "dispatch surface webchat does not match session provider qqbot",
    });
  });

  it("does not skip when dispatch surface matches session provider (qqbot via qqbot)", () => {
    const { skipReason } = resolveAndCacheIdentity({
      sessionKey: "agent:main:qqbot:direct:user-1541-match",
      ctx: {
        sessionKey: "agent:main:qqbot:direct:user-1541-match",
        senderId: "user-1541-match",
      },
      dispatchChannel: "qqbot",
    });

    expect(skipReason).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// waitForReady — CLI mode no-op (initPromise is null before service.start())
// ---------------------------------------------------------------------------

describe("waitForReady (CLI mode)", () => {
  it("returns without error when initPromise is null (service.start not called)", async () => {
    // The module sets up global.__hindsightClient on import.
    // In test context, service.start() is never called so initPromise remains null.
    const hindsight = (global as any).__hindsightClient;
    expect(hindsight).toBeDefined();
    // Should resolve without throwing
    await expect(hindsight.waitForReady()).resolves.toBeUndefined();
  });

  it("getClient returns null when service.start not called", () => {
    const hindsight = (global as any).__hindsightClient;
    expect(hindsight.getClient()).toBeNull();
  });
});

describe("meetsMinimumVersion", () => {
  it("treats equal versions as supported", () => {
    expect(meetsMinimumVersion("0.5.0", "0.5.0")).toBe(true);
  });

  it("returns true for newer major/minor/patch", () => {
    expect(meetsMinimumVersion("0.5.1", "0.5.0")).toBe(true);
    expect(meetsMinimumVersion("0.6.0", "0.5.0")).toBe(true);
    expect(meetsMinimumVersion("1.0.0", "0.5.0")).toBe(true);
  });

  it("returns false for older versions", () => {
    expect(meetsMinimumVersion("0.4.22", "0.5.0")).toBe(false);
    expect(meetsMinimumVersion("0.4.0", "0.5.0")).toBe(false);
    expect(meetsMinimumVersion("0.0.1", "0.5.0")).toBe(false);
  });

  it("ignores pre-release suffixes (treats them as the bare version)", () => {
    expect(meetsMinimumVersion("0.5.0-beta.1", "0.5.0")).toBe(true);
    expect(meetsMinimumVersion("0.4.99-rc.1", "0.5.0")).toBe(false);
  });

  it("treats missing patch / minor as zero", () => {
    expect(meetsMinimumVersion("0.5", "0.5.0")).toBe(true);
    expect(meetsMinimumVersion("1", "0.5.0")).toBe(true);
    expect(meetsMinimumVersion("0.4", "0.5.0")).toBe(false);
  });

  it("returns false for malformed versions instead of throwing", () => {
    expect(meetsMinimumVersion("garbage", "0.5.0")).toBe(false);
    expect(meetsMinimumVersion("", "0.5.0")).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// getPluginConfig — whitelist normalisation
// ---------------------------------------------------------------------------

function makeApi(rawConfig: Record<string, unknown>): MoltbotPluginAPI {
  return {
    config: { plugins: { entries: { "hindsight-openclaw": { config: rawConfig } } } },
    registerService: () => undefined,
    on: () => undefined,
    logger: { info: () => undefined, warn: () => undefined, error: () => undefined },
  } as unknown as MoltbotPluginAPI;
}

describe("getPluginConfig — retainQueue whitelist (#1443)", () => {
  it("passes retainQueuePath through when set to a non-empty string", () => {
    const cfg = getPluginConfig(makeApi({ retainQueuePath: "/custom/path/retain.jsonl" }));
    expect(cfg.retainQueuePath).toBe("/custom/path/retain.jsonl");
  });

  it("drops retainQueuePath when blank or non-string", () => {
    expect(getPluginConfig(makeApi({ retainQueuePath: "   " })).retainQueuePath).toBeUndefined();
    expect(getPluginConfig(makeApi({ retainQueuePath: 42 })).retainQueuePath).toBeUndefined();
    expect(getPluginConfig(makeApi({})).retainQueuePath).toBeUndefined();
  });

  it("passes retainQueueMaxAgeMs through (including the sentinel -1)", () => {
    expect(getPluginConfig(makeApi({ retainQueueMaxAgeMs: 86_400_000 })).retainQueueMaxAgeMs).toBe(
      86_400_000
    );
    expect(getPluginConfig(makeApi({ retainQueueMaxAgeMs: -1 })).retainQueueMaxAgeMs).toBe(-1);
  });

  it("drops retainQueueMaxAgeMs when not a number", () => {
    expect(
      getPluginConfig(makeApi({ retainQueueMaxAgeMs: "86400000" })).retainQueueMaxAgeMs
    ).toBeUndefined();
  });

  it("passes retainQueueFlushIntervalMs through when positive", () => {
    expect(
      getPluginConfig(makeApi({ retainQueueFlushIntervalMs: 30_000 })).retainQueueFlushIntervalMs
    ).toBe(30_000);
  });

  it("drops retainQueueFlushIntervalMs when zero, negative, or non-number", () => {
    expect(
      getPluginConfig(makeApi({ retainQueueFlushIntervalMs: 0 })).retainQueueFlushIntervalMs
    ).toBeUndefined();
    expect(
      getPluginConfig(makeApi({ retainQueueFlushIntervalMs: -5 })).retainQueueFlushIntervalMs
    ).toBeUndefined();
  });
});

describe("getPluginConfig — enableKnowledgeTools whitelist", () => {
  // Regression: the field was declared on PluginConfig and read at the
  // tool-registration site, but never copied through getPluginConfig, so the
  // runtime value was always undefined and the agent_knowledge_* tools never
  // registered (live since the feature was first added on Apr 29 2026).
  it("passes enableKnowledgeTools=true through when set", () => {
    const cfg = getPluginConfig(makeApi({ enableKnowledgeTools: true }));
    expect(cfg.enableKnowledgeTools).toBe(true);
  });

  it("defaults to false when not set or set to a non-boolean truthy value", () => {
    expect(getPluginConfig(makeApi({})).enableKnowledgeTools).toBe(false);
    expect(getPluginConfig(makeApi({ enableKnowledgeTools: false })).enableKnowledgeTools).toBe(
      false
    );
    expect(getPluginConfig(makeApi({ enableKnowledgeTools: "true" })).enableKnowledgeTools).toBe(
      false
    );
    expect(getPluginConfig(makeApi({ enableKnowledgeTools: 1 })).enableKnowledgeTools).toBe(false);
  });
});

describe("formatHookPerf (#1406)", () => {
  it("emits the hook name, total ms, and field key=value pairs", () => {
    const line = formatHookPerf("before_prompt_build", 4200, {
      recall_main: "3800ms",
      source: "fresh",
      results: 3,
    });
    expect(line).toBe(
      "perf: before_prompt_build hook_total=4200ms recall_main=3800ms source=fresh results=3"
    );
  });

  it("renders agent_end fields including string outcome and numeric counts", () => {
    const line = formatHookPerf("agent_end", 1200, {
      retain: "1100ms",
      outcome: "ok",
      bank: "main",
      messages: 4,
    });
    expect(line).toBe(
      "perf: agent_end hook_total=1200ms retain=1100ms outcome=ok bank=main messages=4"
    );
  });

  it("skips fields whose value is undefined", () => {
    const line = formatHookPerf("before_prompt_build", 50, {
      recall_main: undefined,
      source: "skipped",
      results: 0,
    });
    expect(line).toBe("perf: before_prompt_build hook_total=50ms source=skipped results=0");
  });
});

describe("getPluginConfig — debugPerfTiming flag (#1406)", () => {
  it("defaults to false when unset", () => {
    expect(getPluginConfig(makeApi({})).debugPerfTiming).toBe(false);
  });

  it("only accepts strict true (not truthy)", () => {
    expect(getPluginConfig(makeApi({ debugPerfTiming: true })).debugPerfTiming).toBe(true);
    expect(getPluginConfig(makeApi({ debugPerfTiming: false })).debugPerfTiming).toBe(false);
    expect(getPluginConfig(makeApi({ debugPerfTiming: "yes" })).debugPerfTiming).toBe(false);
    expect(getPluginConfig(makeApi({ debugPerfTiming: 1 })).debugPerfTiming).toBe(false);
  });
});

describe("getPluginConfig — mission semantics (#1270, #1353)", () => {
  it("does not substitute a default mission when bankMission is unset", () => {
    const cfg = getPluginConfig(makeApi({}));
    expect(cfg.bankMission).toBeUndefined();
  });

  it("treats empty-string bankMission as opt-out (no default fallback)", () => {
    const cfg = getPluginConfig(makeApi({ bankMission: "" }));
    expect(cfg.bankMission).toBeUndefined();
  });

  it("passes through an explicit bankMission verbatim", () => {
    const cfg = getPluginConfig(makeApi({ bankMission: "You are Cooper, the orchestrator." }));
    expect(cfg.bankMission).toBe("You are Cooper, the orchestrator.");
  });

  it("exposes retainMission and observationsMission when set", () => {
    const cfg = getPluginConfig(
      makeApi({
        retainMission: "Extract architectural decisions only.",
        observationsMission: "Synthesise stable preferences.",
      })
    );
    expect(cfg.retainMission).toBe("Extract architectural decisions only.");
    expect(cfg.observationsMission).toBe("Synthesise stable preferences.");
  });

  it("treats empty-string retainMission and observationsMission as unset", () => {
    const cfg = getPluginConfig(makeApi({ retainMission: "", observationsMission: "" }));
    expect(cfg.retainMission).toBeUndefined();
    expect(cfg.observationsMission).toBeUndefined();
  });
});

describe("getPluginConfig — dynamic bank defaults", () => {
  it("leaves bank default fields unset when not configured", () => {
    const cfg = getPluginConfig(makeApi({}));
    expect(cfg.retainExtractionMode).toBeUndefined();
    expect(cfg.enableObservations).toBeUndefined();
    expect(cfg.enableAutoConsolidation).toBeUndefined();
    expect(cfg.dispositionSkepticism).toBeUndefined();
    expect(cfg.entityLabels).toBeUndefined();
  });

  it("normalizes configured bank default fields", () => {
    const labels = [{ name: "person", description: "Human" }];
    const cfg = getPluginConfig(
      makeApi({
        retainExtractionMode: "VERBOSE",
        enableObservations: true,
        enableAutoConsolidation: false,
        dispositionSkepticism: 4,
        dispositionLiteralism: 2,
        dispositionEmpathy: 5,
        entityLabels: labels,
      })
    );
    expect(cfg.retainExtractionMode).toBe("verbose");
    expect(cfg.enableObservations).toBe(true);
    expect(cfg.enableAutoConsolidation).toBe(false);
    expect(cfg.dispositionSkepticism).toBe(4);
    expect(cfg.entityLabels).toEqual(labels);
  });

  it("rejects invalid retainExtractionMode and disposition values", () => {
    const cfg = getPluginConfig(
      makeApi({
        retainExtractionMode: "not-a-mode",
        dispositionSkepticism: 9,
      })
    );
    expect(cfg.retainExtractionMode).toBeUndefined();
    expect(cfg.dispositionSkepticism).toBeUndefined();
  });
});

describe("getPluginConfig — retainContext", () => {
  it("defaults retainContext to the built-in OpenClaw transcript guidance", () => {
    const cfg = getPluginConfig(makeApi({}));
    expect(cfg.retainContext).toBe(DEFAULT_RETAIN_CONTEXT);
  });

  it("passes through an explicit non-empty retainContext", () => {
    const cfg = getPluginConfig(makeApi({ retainContext: "Treat IDs as routing metadata." }));
    expect(cfg.retainContext).toBe("Treat IDs as routing metadata.");
  });

  it("trims an explicit retainContext", () => {
    const cfg = getPluginConfig(makeApi({ retainContext: " Treat IDs as routing metadata. \n" }));
    expect(cfg.retainContext).toBe("Treat IDs as routing metadata.");
  });

  it("falls back to the default when retainContext is blank or non-string", () => {
    expect(getPluginConfig(makeApi({ retainContext: "" })).retainContext).toBe(
      DEFAULT_RETAIN_CONTEXT
    );
    expect(getPluginConfig(makeApi({ retainContext: 42 })).retainContext).toBe(
      DEFAULT_RETAIN_CONTEXT
    );
  });

  it("keeps the plugin manifest default in sync with the code default", () => {
    expect(openclawManifest.configSchema?.properties?.retainContext?.default).toBe(
      DEFAULT_RETAIN_CONTEXT
    );
  });
});

// ---------------------------------------------------------------------------
// resolveBankIdForKnowledgeTools — dynamic user-scoped banking (#2441)
// ---------------------------------------------------------------------------

describe("resolveBankIdForKnowledgeTools", () => {
  const userScopedConfig: PluginConfig = {
    dynamicBankGranularity: ["user"],
    bankIdPrefix: "nemoclaw_intel",
  };

  it("routes msteams direct sessions to the per-user bank, not shared defaults", () => {
    const userId = "a1b2c3d4-e5f6-7890-abcd-ef1234567890";
    const resolution = resolveBankIdForKnowledgeTools(
      {
        agentId: "nemoclaw",
        sessionKey: `agent:nemoclaw:msteams:direct:${userId}`,
      },
      userScopedConfig
    );

    expect(resolution.identityError).toBeUndefined();
    expect(resolution.bankId).toBe(`nemoclaw_intel-${userId}`);
    expect(resolution.bankId).not.toBe("openclaw");
    expect(resolution.bankId).not.toBe("nemoclaw_intel-openclaw");
    expect(resolution.bankId).not.toBe("nemoclaw_intel-anonymous");
  });

  it("does not silently fall back to shared/default bank when user identity is missing", () => {
    const resolution = resolveBankIdForKnowledgeTools(
      {
        agentId: "nemoclaw",
        sessionKey: "agent:nemoclaw:msteams:group:19:general@thread.tacv2",
      },
      userScopedConfig
    );

    expect(resolution.identityError).toMatch(/missing stable sender identity/);
    expect(resolution.bankId).not.toBe("openclaw");
    expect(resolution.bankId).not.toBe("nemoclaw_intel-openclaw");
    expect(resolution.bankId).toBe("nemoclaw_intel-anonymous");
  });

  it("uses static bankId when dynamicBankId is false", () => {
    const resolution = resolveBankIdForKnowledgeTools(
      { sessionKey: "agent:nemoclaw:main" },
      { dynamicBankId: false, bankId: "shared-team-memory" }
    );

    expect(resolution.identityError).toBeUndefined();
    expect(resolution.bankId).toBe("shared-team-memory");
  });
});

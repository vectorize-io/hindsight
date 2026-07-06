import { describe, expect, it } from "vitest";
import {
  HINDSIGHT_CLOUD_API_URL,
  DEFAULT_RECALL_QUERY,
  buildRetainContent,
  recordAssistantMessage,
  recordUserMessage,
  resolveAutoMemory,
  takeTurn,
  type TurnBuffer,
} from "./config";
import { buildRecallMarkdown } from "./client";

const EMPTY_ENV = {} as NodeJS.ProcessEnv;

describe("resolveAutoMemory", () => {
  it("defaults to Hindsight Cloud + the broad recall query", () => {
    const r = resolveAutoMemory({ apiKey: "hsk_k" }, EMPTY_ENV);
    expect(r.apiUrl).toBe(HINDSIGHT_CLOUD_API_URL);
    expect(r.bankId).toBe("default");
    expect(r.recallQuery).toBe(DEFAULT_RECALL_QUERY);
    expect(r.budget).toBe("mid");
  });

  it("reads url/key/bank from the environment", () => {
    const r = resolveAutoMemory({}, {
      HINDSIGHT_API_URL: "http://localhost:8000",
      HINDSIGHT_API_KEY: "env_key",
      HINDSIGHT_BANK_ID: "project-x",
    } as NodeJS.ProcessEnv);
    expect(r.apiUrl).toBe("http://localhost:8000");
    expect(r.apiKey).toBe("env_key");
    expect(r.bankId).toBe("project-x");
  });

  it("prefers explicit options over the environment", () => {
    const r = resolveAutoMemory({ apiUrl: "http://opt", apiKey: "opt_key", bankId: "opt_bank" }, {
      HINDSIGHT_API_URL: "http://env",
      HINDSIGHT_API_KEY: "env_key",
      HINDSIGHT_BANK_ID: "env_bank",
    } as NodeJS.ProcessEnv);
    expect(r.apiUrl).toBe("http://opt");
    expect(r.apiKey).toBe("opt_key");
    expect(r.bankId).toBe("opt_bank");
  });

  it("treats apiKey: null as a no-auth opt-out", () => {
    const r = resolveAutoMemory({ apiUrl: "http://localhost:8000", apiKey: null }, EMPTY_ENV);
    expect(r.apiKey).toBeNull();
  });

  it("throws when targeting Hindsight Cloud without a key", () => {
    expect(() => resolveAutoMemory({}, EMPTY_ENV)).toThrow(/API key/);
  });

  it("allows a self-hosted url with no auth", () => {
    const r = resolveAutoMemory({ apiUrl: "http://localhost:8000", apiKey: null }, EMPTY_ENV);
    expect(r.apiUrl).toBe("http://localhost:8000");
  });
});

describe("turn pairing buffer", () => {
  it("stores both the user message and assistant reply by default", () => {
    const buf: TurnBuffer = new Map();
    recordUserMessage(buf, "t1", "I prefer tabs");
    recordAssistantMessage(buf, "t1", "Noted.");
    const pair = takeTurn(buf, "t1");
    expect(buf.has("t1")).toBe(false); // taken
    expect(buildRetainContent(pair)).toBe("User: I prefer tabs\n\nAssistant: Noted.");
  });

  it("drops the assistant reply when includeAssistant is false", () => {
    const buf: TurnBuffer = new Map();
    recordUserMessage(buf, "t1", "I prefer tabs");
    recordAssistantMessage(buf, "t1", "Noted.");
    expect(buildRetainContent(takeTurn(buf, "t1"), false)).toBe("User: I prefer tabs");
  });

  it("skips a turn with no user text", () => {
    const buf: TurnBuffer = new Map();
    recordAssistantMessage(buf, "t1", "hello");
    expect(buildRetainContent(takeTurn(buf, "t1"))).toBeNull();
    expect(buildRetainContent(undefined)).toBeNull();
  });

  it("keeps just the user text when there is no assistant answer", () => {
    const buf: TurnBuffer = new Map();
    recordUserMessage(buf, "t1", "remember this");
    expect(buildRetainContent(takeTurn(buf, "t1"))).toBe("User: remember this");
  });

  it("strips injected recalled-context from the assistant half when included", () => {
    const buf: TurnBuffer = new Map();
    const recalled = buildRecallMarkdown([{ id: "1", text: "user is vegan" }]);
    recordUserMessage(buf, "t1", "what's for dinner?");
    recordAssistantMessage(buf, "t1", `${recalled}\nHow about pasta?`);
    const content = buildRetainContent(takeTurn(buf, "t1"), true);
    expect(content).toContain("what's for dinner?");
    expect(content).toContain("How about pasta?");
    expect(content).not.toContain("user is vegan");
  });
});

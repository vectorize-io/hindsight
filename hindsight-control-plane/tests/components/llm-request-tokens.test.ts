import { describe, expect, it } from "vitest";
import { totalTokens } from "@/components/llm-requests-view";
import type { LLMRequestEntry } from "@/lib/api";

/**
 * The API's `total_tokens` is input + visible output only — reasoning is billed
 * but reported separately, so the UI adds it back. Rows written before reasoning
 * usage existed carry null, and must keep rendering as "no total" rather than 0.
 */
const entry = (tokens: Partial<LLMRequestEntry>): LLMRequestEntry =>
  ({
    total_tokens: null,
    thoughts_tokens: null,
    ...tokens,
  }) as LLMRequestEntry;

describe("totalTokens", () => {
  it("adds reported reasoning to the API total", () => {
    expect(totalTokens(entry({ total_tokens: 120, thoughts_tokens: 60 }))).toBe(180);
  });

  it("keeps the API total when no reasoning was reported", () => {
    expect(totalTokens(entry({ total_tokens: 120 }))).toBe(120);
  });

  it("reports reasoning alone when the API total is missing", () => {
    expect(totalTokens(entry({ thoughts_tokens: 60 }))).toBe(60);
  });

  it("stays null when nothing was reported", () => {
    expect(totalTokens(entry({}))).toBeNull();
    expect(totalTokens(entry({ thoughts_tokens: 0 }))).toBeNull();
  });
});

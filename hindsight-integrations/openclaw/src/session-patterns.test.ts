import { describe, it, expect } from "vitest";
import {
  compileSessionPattern,
  compileSessionPatterns,
  matchesSessionPattern,
} from "./session-patterns.js";

// ---------------------------------------------------------------------------
// compileSessionPattern
// ---------------------------------------------------------------------------

describe("compileSessionPattern", () => {
  it("matches an exact key", () => {
    const p = compileSessionPattern("agent:main:sess-123");
    expect(p.test("agent:main:sess-123")).toBe(true);
    expect(p.test("agent:main:sess-456")).toBe(false);
  });

  it("single * does not cross colon", () => {
    const p = compileSessionPattern("agent:*:sess");
    expect(p.test("agent:main:sess")).toBe(true);
    expect(p.test("agent:subagent:sess")).toBe(true);
    expect(p.test("agent:a:b:sess")).toBe(false);
  });

  it("double ** crosses colons", () => {
    const p = compileSessionPattern("agent:main:**");
    expect(p.test("agent:main:sess-abc123")).toBe(true);
    expect(p.test("agent:main:a:b:c")).toBe(true);
    expect(p.test("agent:other:sess-abc123")).toBe(false);
  });

  it("double ** at start matches any prefix", () => {
    const p = compileSessionPattern("**:subagent:**");
    expect(p.test("claude-code:subagent:sess-abc")).toBe(true);
    expect(p.test("mybot:subagent:sess-xyz")).toBe(true);
    expect(p.test("mybot:main:sess-xyz")).toBe(false);
  });

  it("matches lossless-claw cron pattern", () => {
    const p = compileSessionPattern("agent:*:cron:**");
    expect(p.test("agent:mybot:cron:sess-123")).toBe(true);
    expect(p.test("agent:mybot:subagent:sess-123")).toBe(false);
  });

  it("matches lossless-claw subagent pattern", () => {
    const p = compileSessionPattern("agent:*:subagent:**");
    expect(p.test("agent:main:subagent:sess-abc")).toBe(true);
    expect(p.test("agent:x:subagent:sess-123")).toBe(true);
    expect(p.test("agent:a:b:subagent:sess")).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// matchesSessionPattern
// ---------------------------------------------------------------------------

describe("matchesSessionPattern", () => {
  it("returns true when any pattern matches", () => {
    const patterns = compileSessionPatterns(["agent:main:**", "agent:*:cron:**"]);
    expect(matchesSessionPattern("agent:main:sess-abc", patterns)).toBe(true);
    expect(matchesSessionPattern("agent:mybot:cron:sess-xyz", patterns)).toBe(true);
  });

  it("returns false when no pattern matches", () => {
    const patterns = compileSessionPatterns(["agent:main:**"]);
    expect(matchesSessionPattern("agent:subagent:sess-abc", patterns)).toBe(false);
  });

  it("returns false for empty pattern list", () => {
    expect(matchesSessionPattern("agent:main:sess", [])).toBe(false);
  });

  it("lossless-claw ignoreSessionPatterns example", () => {
    const patterns = compileSessionPatterns(["agent:main:**", "agent:*:cron:**"]);
    expect(matchesSessionPattern("agent:main:sess-abc123", patterns)).toBe(true);
    expect(matchesSessionPattern("agent:mybot:cron:sess-123", patterns)).toBe(true);
    expect(matchesSessionPattern("agent:mybot:subagent:sess-123", patterns)).toBe(false);
  });

  it("lossless-claw statelessSessionPatterns example", () => {
    const patterns = compileSessionPatterns(["agent:*:subagent:**", "agent:*:heartbeat:**"]);
    expect(matchesSessionPattern("agent:main:subagent:sess-abc", patterns)).toBe(true);
    expect(matchesSessionPattern("agent:main:heartbeat:sess-abc", patterns)).toBe(true);
    expect(matchesSessionPattern("agent:main:sess-abc", patterns)).toBe(false);
  });
});

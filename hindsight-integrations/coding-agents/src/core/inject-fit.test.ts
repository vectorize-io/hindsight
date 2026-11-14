import { describe, expect, it } from "vitest";
import { buildSystemInjection, fitMemory, MAX_INJECTED_MEMORY_CHARS } from "./inject.js";

describe("the injected memory fits Claude Code's additionalContext limit", () => {
  it("keeps a whole block under 10,000 characters however much was recalled", () => {
    const recalled = Array.from({ length: 200 }, (_, i) => `- memory ${i}: ${"x".repeat(60)}`).join(
      "\n"
    );
    const block = buildSystemInjection(recalled);
    expect(block.length).toBeLessThan(10_000);
    expect(block).toContain("- memory 0:");
    expect(block).not.toContain("- memory 199:");
    expect(block.endsWith("</hindsight_memory>")).toBe(true);
  });

  it("cuts at a line boundary and says so", () => {
    const fitted = fitMemory(
      Array.from({ length: 100 }, (_, i) => `line ${i} ${"y".repeat(100)}`).join("\n")
    );
    expect(fitted.length).toBeLessThanOrEqual(MAX_INJECTED_MEMORY_CHARS);
    expect(fitted).toMatch(/y\n\(… more memory left out to fit the context limit\)$/);
  });

  it("leaves a short memory untouched", () => {
    expect(fitMemory("one fact")).toBe("one fact");
  });
});

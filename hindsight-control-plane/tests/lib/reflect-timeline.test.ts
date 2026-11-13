import { describe, expect, it } from "vitest";
import { buildReflectTimeline, traceIterations, traceWallMs } from "@/lib/reflect-timeline";

const label = (items: ReturnType<typeof buildReflectTimeline>) =>
  items.map((i) =>
    i.kind === "llm"
      ? `${i.step.title}@${i.offset_ms}`
      : `tools:${i.tools.map((t) => t.tool)}@${i.offset_ms}`
  );

describe("buildReflectTimeline", () => {
  it("orders a decision-model run by start time, not by pairing LLM call N with iteration N", () => {
    const llm = [
      { scope: "round_query", duration_ms: 500, started_at_ms: 1000 },
      { scope: "decision_prune", duration_ms: 400, started_at_ms: 1900 },
      { scope: "decision_sufficiency", duration_ms: 200, started_at_ms: 2300 },
      { scope: "agent_2", duration_ms: 800, started_at_ms: 2500 },
      { scope: "agent_3", duration_ms: 900, started_at_ms: 3700 },
    ];
    const tools = [
      { tool: "search_observations", iteration: 1, duration_ms: 300, started_at_ms: 1500 },
      { tool: "recall", iteration: 1, duration_ms: 350, started_at_ms: 1500 },
      { tool: "recall", iteration: 2, duration_ms: 300, started_at_ms: 3300 },
    ];

    expect(label(buildReflectTimeline(llm, tools))).toEqual([
      "traceRoundQuery@0",
      "tools:search_observations,recall@500",
      "tracePrune@900",
      "traceSufficiency@1300",
      "agentDecided@1500",
      "tools:recall@2300",
      // The last free turn's done is the answer when no closing call follows.
      "responseGenerated@2700",
    ]);
    expect(traceIterations(tools)).toBe(2);
    expect(traceWallMs(llm, tools)).toBe(3600);
  });

  it("keeps an agent turn that called done before a closing answer as 'has enough'", () => {
    const llm = [
      { scope: "agent_1", duration_ms: 100, started_at_ms: 0 },
      { scope: "agent_2", duration_ms: 100, started_at_ms: 300 },
      { scope: "closing_done", duration_ms: 100, started_at_ms: 400 },
    ];
    const tools = [{ tool: "recall", iteration: 1, duration_ms: 200, started_at_ms: 100 }];

    expect(label(buildReflectTimeline(llm, tools))).toEqual([
      "agentDecided@0",
      "tools:recall@100",
      "traceAgentDone@300",
      "responseGenerated@400",
    ]);
  });
});

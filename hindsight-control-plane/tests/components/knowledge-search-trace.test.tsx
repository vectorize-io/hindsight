// @vitest-environment jsdom
/**
 * A traced knowledge search reads like the Recall trace: each candidate search's own list, then
 * each step with its partial list and scores, down to how max_tokens was filled.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string, values?: Record<string, unknown>) =>
    values ? `${key}:${JSON.stringify(values)}` : key,
}));

const { KnowledgeSearchTraceView } = await import("@/components/knowledge-search-trace");

afterEach(cleanup);

const hit = (rank: number) => ({
  rank,
  outcome: (rank === 1 ? "returned" : rank === 2 ? "merged" : "over_budget") as
    | "returned"
    | "merged"
    | "over_budget",
  result_rank: rank === 1 ? 1 : null,
  merged_into_rank: rank === 2 ? 1 : null,
  document_id: `d${rank}`,
  filename: `doc-${rank}.md`,
  index: rank,
  media_type: "text" as const,
  page: null,
  start_ms: null,
  end_ms: null,
  snippet: `snippet ${rank}`,
  score: 1 - rank / 100,
  found_by: ["vector"],
  similarity: 0.8,
  roles: {},
  passage_score: 0.8,
  with_document: 0.85 - rank / 1000,
  keyword: rank === 1 ? 2.5 : 0,
  reference_page: false,
});

const trace = (n: number) => ({
  budget: "mid" as const,
  depth: n,
  max_tokens: 4096,
  tokens_used: 1200,
  candidates: n,
  keyword_weight: 0.2,
  keyword_terms: { invoic: 3.14 },
  timings_ms: { embed: 12.4, "store.search": 80 },
  candidate_searches: [
    {
      name: "keyword" as const,
      results: Array.from({ length: n }, (_, i) => ({
        document_id: `d${i + 1}`,
        index: i + 1,
        score: 0.5 - i / 1000,
      })),
    },
  ],
  hits: Array.from({ length: n }, (_, i) => hit(i + 1)),
  budget_steps: [
    {
      action: "window" as const,
      rank: 1,
      document_id: "d1",
      first: 0,
      last: 2,
      tokens: 900,
      total: 900,
      into_rank: null,
    },
    {
      action: "merged" as const,
      rank: 2,
      document_id: "d1",
      first: 0,
      last: 3,
      tokens: 300,
      total: 1200,
      into_rank: 1,
    },
  ],
});

describe("KnowledgeSearchTraceView", () => {
  it("opens a candidate search to its own list and scores", () => {
    render(<KnowledgeSearchTraceView trace={trace(3)} />);
    expect(screen.queryByText("snippet 1")).toBeNull();
    fireEvent.click(screen.getByText("arm_keyword"));
    expect(screen.getByText("snippet 1")).toBeTruthy();
    expect(screen.getByText("armScore 0.500")).toBeTruthy();
  });

  it("shows how max_tokens was filled, step by step with the running total", () => {
    render(<KnowledgeSearchTraceView trace={trace(3)} />);
    fireEvent.click(screen.getByText("budgetTitle"));
    expect(screen.getByText('action_window:{"rank":0}')).toBeTruthy();
    expect(screen.getByText('action_merged:{"rank":1}')).toBeTruthy();
    expect(screen.getByText('total:{"total":1200,"max":4096}')).toBeTruthy();
  });

  it("lists the final ranking with each hit's meaning, keyword and final score", () => {
    render(<KnowledgeSearchTraceView trace={trace(12)} />);
    fireEvent.click(screen.getByText("rankingTitle"));
    expect(screen.getByText("score 0.990")).toBeTruthy();
    expect(screen.getByText("keyword 2.50")).toBeTruthy();
    expect(screen.queryByText("doc-12.md")).toBeNull();
    fireEvent.click(screen.getByText('viewAll:{"count":12}'));
    expect(screen.getByText("doc-12.md")).toBeTruthy();
  });
});

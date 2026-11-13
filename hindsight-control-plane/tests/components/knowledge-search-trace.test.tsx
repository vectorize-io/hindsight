// @vitest-environment jsdom
/**
 * A traced knowledge search shows every ranked candidate, its score step by step and what the
 * token budget did with it.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string, values?: Record<string, unknown>) =>
    values ? `${key}:${JSON.stringify(values)}` : key,
}));

const { KnowledgeSearchTraceView } = await import("@/components/knowledge-search-trace");

afterEach(cleanup);

const hit = (rank: number, outcome: "returned" | "merged" | "over_budget") => ({
  rank,
  outcome,
  result_rank: outcome === "returned" ? rank : null,
  merged_into_rank: outcome === "merged" ? 1 : null,
  document_id: `d${rank}`,
  filename: `doc-${rank}.md`,
  index: rank,
  media_type: "text" as const,
  page: null,
  start_ms: null,
  end_ms: null,
  snippet: `snippet ${rank}`,
  score: 1 - rank / 100,
  found_by: ["vector", "keyword"],
  similarity: 0.8,
  roles: {},
  passage_score: 0.8,
  with_document: 0.8,
  keyword: rank === 1 ? 2.5 : 0,
  reference_page: false,
});

const trace = (n: number) => ({
  budget: "mid" as const,
  max_tokens: 4096,
  tokens_used: 1200,
  candidates: 300,
  language: "english",
  keyword_weight: 0.2,
  keyword_terms: { invoic: 3.14 },
  timings_ms: { embed: 12.4, "store.search": 80 },
  hits: Array.from({ length: n }, (_, i) =>
    hit(i + 1, i === 0 ? "returned" : i === 1 ? "merged" : "over_budget")
  ),
});

describe("KnowledgeSearchTraceView", () => {
  it("shows the search's facts and each candidate's score and outcome", () => {
    render(<KnowledgeSearchTraceView trace={trace(3)} />);
    expect(screen.getByText("300")).toBeTruthy();
    expect(screen.getByText("1200 / 4096")).toBeTruthy();
    expect(screen.getByText("invoic")).toBeTruthy();
    expect(screen.getByText("0.990")).toBeTruthy();
    expect(screen.getByText("2.50")).toBeTruthy();
    expect(screen.getByText('returned:{"rank":1}')).toBeTruthy();
    expect(screen.getByText('merged:{"rank":1}')).toBeTruthy();
    expect(screen.getByText("overBudget")).toBeTruthy();
  });

  it("shows the first 50 candidates, and all of them on request", () => {
    render(<KnowledgeSearchTraceView trace={trace(60)} />);
    expect(screen.queryByText("doc-60.md")).toBeNull();
    fireEvent.click(screen.getByText('showAll:{"count":60}'));
    expect(screen.getByText("doc-60.md")).toBeTruthy();
  });
});

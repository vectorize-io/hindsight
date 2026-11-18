// @vitest-environment jsdom
import { describe, expect, it } from "vitest";

import { tooltipHtml } from "@/components/constellation";
import type { GraphNode } from "@/components/graph-data";

const MARKUP = "<b>bold</b>";

/** Render the tooltip the way the component does. */
function render(node: GraphNode, heatColor = "#ef4444"): HTMLElement {
  const tip = document.createElement("div");
  tip.innerHTML = tooltipHtml(node, heatColor, 3, false);
  return tip;
}

describe("constellation tooltip", () => {
  it("shows every stored field as text, never as markup", () => {
    const tip = render({
      id: `doc:${MARKUP}`,
      label: MARKUP,
      metadata: {
        text: MARKUP,
        entities: `${MARKUP}, Alice`,
        fact_type: MARKUP,
        context: MARKUP,
        tags: [MARKUP],
        occurred_start: MARKUP,
        occurred_end: `${MARKUP}x`,
        mentioned_at: MARKUP,
        proof_count: MARKUP,
        document_id: MARKUP,
      },
    } as GraphNode);

    expect(tip.querySelector("b")).toBeNull();
    expect(tip.textContent).toContain(`doc:${MARKUP}`);
    expect(tip.textContent).toContain("Alice");
  });

  it("keeps a node colour inside its style attribute", () => {
    const tip = render({ id: "n1", metadata: { text: "hello" } } as GraphNode, `red"><b>x</b>`);
    expect(tip.querySelector("b")).toBeNull();
    expect(tip.textContent).toContain("hello");
  });

  it("keeps the ordinary fields readable", () => {
    const tip = render({
      id: "unit-1",
      metadata: { text: "Alice plays the cello", mentioned_at: "2026-11-18T10:30:00Z", proof_count: 4 },
    } as GraphNode);
    expect(tip.textContent).toContain("Alice plays the cello");
    expect(tip.textContent).toContain("2026-11-18 10:30");
    expect(tip.textContent).toContain("4 sources");
  });
});

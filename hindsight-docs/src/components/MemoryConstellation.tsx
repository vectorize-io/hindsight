import React, {type ReactNode} from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import {Constellation} from './constellation/Constellation';
import {buildMockBank} from './constellation/mock-bank';
import type {GraphLink, GraphNode} from './constellation/graph-data';

/**
 * The control plane's memory constellation, running live on the docs site.
 *
 * The hero used to show a screenshot of this view. A picture cannot be
 * hovered, so the one claim the hero exists to make — that a bank is a graph
 * you can actually look at — had to be taken on faith. This is the same
 * component the control plane renders (src/components/constellation/
 * Constellation.tsx is a near-copy of hindsight-control-plane/src/components/
 * constellation.tsx: the control plane's next-intl lookup is a local string
 * table, and dark mode reads Docusaurus's `data-theme` instead of a `dark`
 * class), called the way the control plane's Memories view calls it — one flat
 * field, coloured by when each memory was mentioned, sized by how many facts
 * back it — over a synthetic bank rather than a real one.
 *
 * The data is mocked and deterministic — see constellation/mock-bank.ts. The
 * docs site has no dataplane to query, and a hero figure that reshuffles on
 * every reload is one nobody can point at in a conversation.
 */

const BANK = buildMockBank();

const MENTIONED = BANK.nodes.map((n) => Date.parse(n.metadata?.mentioned_at));
const MIN_T = Math.min(...MENTIONED);
const MAX_T = Math.max(...MENTIONED);
const day = (t: number) => new Date(t).toISOString().slice(0, 10);

// The Memories view's own mappings (data-view.tsx).
const recencyHeat = (node: GraphNode) =>
  (Date.parse(node.metadata?.mentioned_at) - MIN_T) / (MAX_T - MIN_T);
// Observations are sized by their source facts (the Observations tab); facts
// have none, so they keep the size the World/Experience tabs give them — the
// component's default, by link count.
const LINK_COUNT = new Map<string, number>();
for (const l of BANK.links) {
  LINK_COUNT.set(l.source, (LINK_COUNT.get(l.source) ?? 0) + 1);
  LINK_COUNT.set(l.target, (LINK_COUNT.get(l.target) ?? 0) + 1);
}
const sourceFactsSize = (node: GraphNode) =>
  node.metadata?.proof_count
    ? 3 + Math.min(Math.sqrt(node.metadata.proof_count - 1) * 2, 11)
    : 2.5 + Math.min((LINK_COUNT.get(node.id) ?? 0) * 0.15, 2.5);
const nodeColor = (node: GraphNode) => node.color || '#0074d9';
const linkColor = (link: GraphLink) => {
  if (link.type === 'temporal') return '#009296';
  if (link.type === 'entity') return '#f59e0b';
  if (link.type === 'caused_by') return '#8b5cf6';
  return '#0074d9';
};

export default function MemoryConstellation({
  height = 460,
}: {
  height?: number;
}): ReactNode {
  return (
    /* Canvas-only component: it measures the DOM and reads devicePixelRatio on
       mount, so there is nothing for the static build to render. */
    <BrowserOnly fallback={<div style={{height}} />}>
      {() => (
        <Constellation
          data={BANK}
          height={height}
          nodeColorFn={nodeColor}
          linkColorFn={linkColor}
          nodeSizeFn={sourceFactsSize}
          sizeLegendLabel="source facts"
          nodeHeatFn={recencyHeat}
          heatLegendLabel="recency · mentioned"
          heatLegendEndpoints={[day(MIN_T), day(MAX_T)]}
          allowFullscreen={false}
        />
      )}
    </BrowserOnly>
  );
}

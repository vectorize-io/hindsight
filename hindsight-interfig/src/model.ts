// The figure format and the pure helpers that read it. No React rendering here, so it can be tested directly.
import type { ReactNode } from 'react';

/** `store` draws a database cylinder: use it for data at rest, and plain boxes for the things that process it.
 *  A content card grows to fit the most any step puts in it; `lines` only sets a minimum. `width` overrides the card width. */
export type FigNode = {
  id: string;
  label: ReactNode;
  sub?: ReactNode;
  shape?: 'box' | 'decision' | 'store';
  lines?: number;
  width?: number;
};
export type FigGroup = {
  id?: string;
  label?: ReactNode;
  direction?: 'row' | 'column';
  gap?: number;
  /** Where children line up across the group's direction. Default: center (columns stretch). */
  align?: 'start' | 'center' | 'end';
  children: (FigNode | FigGroup)[];
};
/** `around` routes the edge over or under the boxes in between (loops, skip-ahead edges).
 *  `quiet` edges are only drawn while a step uses them: for long edges that would cut across everything. */
export type FigEdge = {
  id?: string;
  from: string;
  to: string;
  label?: ReactNode;
  around?: 'above' | 'below';
  quiet?: boolean;
};
/** One packet on one edge. `back` runs it in reverse; `data` rides along with it as a small card. */
export type FigHop = string | { edge: string; back?: boolean; data?: ReactNode };
/** One moment of a step. Packets cross `edges` (an array runs them at the same time), `say` narrates it,
 *  and `show` fills the content card inside those boxes (it stays until the step ends). No edges = a pause. */
export type FigBeat = {
  edges?: FigHop | FigHop[];
  say?: ReactNode;
  show?: Record<string, FigContent>;
  light?: string[];
  ms?: number;
};
/** One line of a content card: a colored `tag`, the `text`, a muted `meta` after it, and a `mark` (✓, new…) on the right. */
export type FigRow = {
  tag?: string;
  tone?: FigTone;
  text: ReactNode;
  meta?: ReactNode;
  mark?: ReactNode;
  mono?: boolean;
};
export type FigTone = 'blue' | 'purple' | 'green' | 'orange' | 'gray';
/** What a content card shows: rows, or anything React can render. */
export type FigContent = FigRow[] | ReactNode;
/** One story the figure can tell: the `flow` beats play in order. A plain hop or hop array is a beat with just edges. */
export type FigStep = {
  label: ReactNode;
  caption?: ReactNode;
  flow: (FigHop | FigHop[] | FigBeat)[];
  nodes?: string[];
};
export type FigTheme = Partial<Record<'accent' | 'fg' | 'muted' | 'bg' | 'surface' | 'border' | 'font', string>>;
export type FlowProps = {
  layout: FigGroup;
  edges: FigEdge[];
  steps?: FigStep[];
  /** Colors. Can also be set from CSS with --fig-accent, --fig-bg, ... on `.interfig`. */
  theme?: FigTheme;
  /** Milliseconds a packet takes to cross one edge. */
  speed?: number;
  autoplay?: boolean;
};
/** A ready-made figure: what it shows, where it is used, and what to draw. */
export type Figure = { title: string; source?: string; props: FlowProps };

/** A beat with its hops spelled out: every hop has an edge id and a direction. */
export type Beat = Omit<FigBeat, 'edges'> & { hops: { edge: string; back: boolean; data?: ReactNode }[] };
export const toBeat = (b: FigHop | FigHop[] | FigBeat): Beat => {
  const isBeat = typeof b === 'object' && !Array.isArray(b) && !('edge' in b);
  const { edges, ...rest }: FigBeat = isBeat ? b : { edges: b };
  const hops = edges == null ? [] : Array.isArray(edges) ? edges : [edges];
  return {
    ...rest,
    hops: hops.map((h) => (typeof h === 'string' ? { edge: h, back: false } : { back: false, ...h })),
  };
};

/** Content given as rows (as opposed to any other React node). */
export const isRows = (c: FigContent): c is FigRow[] =>
  Array.isArray(c) && c.every((r) => r != null && typeof r === 'object' && 'text' in r);

export const edgeId = (e: FigEdge) => e.id ?? `${e.from}->${e.to}`;
export const isGroup = (x: FigNode | FigGroup): x is FigGroup => 'children' in x;
export const decisions = (g: FigGroup): string[] =>
  g.children.flatMap((c) => (isGroup(c) ? decisions(c) : c.shape === 'decision' ? [c.id] : []));

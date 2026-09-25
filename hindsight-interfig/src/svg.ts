// The same figure, as one self-contained animated SVG: no scripts, no fonts to fetch, so it plays
// inside a markdown image on GitHub, GitLab and Notion, where the React player cannot go. It gives
// up what needs a reader: no hover, no tabs, no pause — every step runs in one loop.
//
// It reads the same spec and the same `route()` as the React renderer, so the two cannot drift.
//
// ponytail: text is measured by character count rather than by a browser, so this runs anywhere with
// plain node. Wrapping is therefore approximate; widen a card if a line lands short.
import { route, type Rect } from './geometry.ts';
import {
  edgeId,
  isGroup,
  isRows,
  toBeat,
  type Beat,
  type FigContent,
  type FigGroup,
  type FigNode,
  type FigRow,
  type FigTheme,
  type FigTone,
  type FlowProps,
} from './model.ts';

/** Sizes copied from the React renderer so both lay a figure out the same way. */
const CARD_WIDTH = 166,
  LINE = 15,
  CARD_PAD = 6,
  CARD_SIDE = 8,
  ROW_GAP = 4;
const LABEL_LINE = 18,
  SUB_LINE = 15,
  NODE_MIN_W = 100,
  NODE_MAX_W = 190;
const FRAME_TOP = 37,
  FRAME_SIDE = 18,
  FRAME_BOTTOM = 18;
/** Characters are ~0.53em wide in this font stack; enough to place a box, not to typeset a page. */
const em = (fontSize: number) => fontSize * 0.53;
/** The React player runs 1.25x faster than a figure's own timings. Match it so the two feel the same. */
const BASE_RATE = 1.25;

const TONES: Record<FigTone, string> = {
  blue: '#3b82f6',
  purple: '#8b5cf6',
  green: '#10b981',
  orange: '#f59e0b',
  gray: '#8b949e',
};

/** Only text survives the trip to SVG: a React element has no string form here. */
const str = (n: unknown): string => (typeof n === 'string' || typeof n === 'number' ? String(n) : '');
/** A MiniGraph cannot be drawn without React, but its links read fine as text: "Alice → Google". */
const graphText = (c: unknown): string => {
  const props = (c as { props?: { links?: [string, string][] } } | null)?.props;
  return Array.isArray(props?.links) ? props.links.map(([a, b]) => `${a} → ${b}`).join('\n') : '';
};
/** What a card actually holds: rows, or the best text we can get out of it. */
const content = (c: FigContent): FigRow[] | string => (isRows(c) ? c : str(c) || graphText(c));
const esc = (s: string) => s.replace(/[<>&"]/g, (c) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;' })[c]!);
const n2 = (v: number) => Math.round(v * 10) / 10;
const pct = (v: number) => n2(v * 100) + '%';

/** Break a string into lines that fit `width`, keeping the newlines it already has. */
function wrap(s: string, width: number, fontSize: number): string[] {
  const max = Math.max(4, Math.floor(width / em(fontSize)));
  const out: string[] = [];
  for (const para of s.split('\n')) {
    let line = '';
    for (const word of para.split(' ')) {
      if (!line) line = word;
      else if (line.length + 1 + word.length <= max) line += ' ' + word;
      else {
        out.push(line);
        line = word;
      }
    }
    out.push(line);
  }
  return out;
}

type Row = { row: FigRow; heads: boolean; lines: string[] };
/** A card's rows, wrapped for `width`, with the height they need. */
function layoutCard(c: FigContent, width: number): { rows: Row[]; height: number } {
  const inner = width - CARD_SIDE * 2;
  if (c == null) return { rows: [], height: LINE + CARD_PAD * 2 };
  const body = content(c);
  if (!Array.isArray(body)) {
    const lines = wrap(body || ' ', inner, 11);
    return { rows: [{ row: { text: body }, heads: false, lines }], height: lines.length * LINE + CARD_PAD * 2 };
  }
  const rows = body.map((row) => {
    // A word-sized tag heads its row so the text keeps the full width; a number or no tag sits inline.
    const heads = (row.tag?.length ?? 0) > 2;
    const tagW = row.tag && !heads ? row.tag.length * em(9) + 13 : 0;
    const markW = row.mark && !heads ? str(row.mark).length * em(11) + 6 : 0;
    const body = str(row.text) + (row.meta != null ? ' · ' + str(row.meta) : '');
    return { row, heads, lines: wrap(body, inner - tagW - markW, row.mono ? 10.5 : 11) };
  });
  const height =
    rows.reduce((h, r) => h + (r.heads ? LINE : 0) + r.lines.length * LINE, 0) + ROW_GAP * Math.max(0, rows.length - 1) + CARD_PAD * 2;
  return { rows, height };
}

type Placed = Rect & { item: FigNode | FigGroup };
type Sizes = { cards: Map<string, FigContent[]>; cardH: Map<string, number>; minH: (id: string) => number };

function nodeWidth(item: FigNode, carded: boolean): number {
  if (item.width != null) return item.width;
  if (carded) return CARD_WIDTH;
  const label = str(item.label).length * em(14) + 32;
  const sub = str(item.sub).length * em(12) + 32;
  return Math.min(NODE_MAX_W, Math.max(NODE_MIN_W, label, sub));
}

function size(item: FigNode | FigGroup, s: Sizes): { w: number; h: number } {
  if (!isGroup(item)) {
    const contents = s.cards.get(item.id);
    const w = nodeWidth(item, contents != null);
    const top = item.shape === 'store' ? 24 : 10;
    const card = contents ? 8 + s.cardH.get(item.id)! : 0;
    const h = Math.max(top + LABEL_LINE + (item.sub ? SUB_LINE : 0) + card + 10, s.minH(item.id));
    return item.shape === 'decision' ? { w: w + 70, h: h + 24 } : { w, h };
  }
  const kids = item.children.map((c) => size(c, s));
  const gap = item.gap ?? (item.direction === 'column' ? 28 : 56);
  const along = kids.reduce((n, k) => n + (item.direction === 'column' ? k.h : k.w), 0) + gap * (kids.length - 1);
  const across = Math.max(...kids.map((k) => (item.direction === 'column' ? k.w : k.h)));
  const inner = item.direction === 'column' ? { w: across, h: along } : { w: along, h: across };
  return item.label != null ? { w: inner.w + FRAME_SIDE * 2, h: inner.h + FRAME_TOP + FRAME_BOTTOM } : inner;
}

function place(item: FigNode | FigGroup, x: number, y: number, s: Sizes, out: Placed[]): void {
  const { w, h } = size(item, s);
  out.push({ x, y, w, h, item });
  if (!isGroup(item)) return;
  const framed = item.label != null;
  let ix = x + (framed ? FRAME_SIDE : 0);
  let iy = y + (framed ? FRAME_TOP : 0);
  const innerW = w - (framed ? FRAME_SIDE * 2 : 0),
    innerH = h - (framed ? FRAME_TOP + FRAME_BOTTOM : 0);
  const gap = item.gap ?? (item.direction === 'column' ? 28 : 56);
  for (const child of item.children) {
    const c = size(child, s);
    const align = item.align ?? 'center';
    const off = (room: number, used: number) => (align === 'start' ? 0 : align === 'end' ? room - used : (room - used) / 2);
    if (item.direction === 'column') {
      place(child, ix + off(innerW, c.w), iy, s, out);
      iy += c.h + gap;
    } else {
      place(child, ix, iy + off(innerH, c.h), s, out);
      ix += c.w + gap;
    }
  }
}

/** One stretch of the loop: the figure holds still, showing beat `bi` of step `si`. */
type Seg = { t0: number; t1: number; si: number; bi: number };

export type SvgOptions = { speed?: number; padding?: number; theme?: FigTheme };

export function toSvg(fig: FlowProps, opts: SvgOptions = {}): string {
  const speed = (opts.speed ?? fig.speed ?? 900) / 1000 / BASE_RATE;
  const pad = opts.padding ?? 24;
  const steps = fig.steps ?? [];
  const beats: Beat[][] = steps.map((s) => s.flow.map(toBeat));

  // Every content a box will ever show, so its card can be sized to the biggest one up front.
  const cards = new Map<string, FigContent[]>();
  for (const b of beats.flat()) for (const [id, c] of Object.entries(b.show ?? {})) cards.set(id, [...(cards.get(id) ?? []), c]);

  // Boxes with many edges on one side get taller so the edges and their labels have room.
  const out: Record<string, number> = {},
    inn: Record<string, number> = {};
  for (const e of fig.edges) {
    out[e.from] = (out[e.from] ?? 0) + 1;
    inn[e.to] = (inn[e.to] ?? 0) + 1;
  }
  const minH = (id: string) => {
    const n = Math.max(out[id] ?? 0, inn[id] ?? 0);
    return n > 2 ? n * 30 : 0;
  };
  const cardH = new Map<string, number>();
  const sizes: Sizes = { cards, cardH, minH };
  for (const [id, contents] of cards) {
    const width = CARD_WIDTH; // refined below once the node's own width is known
    const widths = [width];
    cardH.set(id, Math.max(LINE + CARD_PAD * 2, ...contents.map((c) => layoutCard(c, Math.min(...widths) - 20).height)));
  }

  const placed: Placed[] = [];
  place(fig.layout, pad, pad, sizes, placed);
  const nodes = placed.filter((p) => !isGroup(p.item)) as (Rect & { item: FigNode })[];
  // A node's own width can differ from CARD_WIDTH (`width` in the spec), so re-measure once placed.
  for (const p of nodes) {
    if (!cards.has(p.item.id)) continue;
    const inner = p.w - 20;
    cardH.set(p.item.id, Math.max(...cards.get(p.item.id)!.map((c) => layoutCard(c, inner).height), LINE + CARD_PAD * 2));
  }
  placed.length = 0;
  place(fig.layout, pad, pad, sizes, placed);

  const rects: Record<string, Rect> = {};
  for (const p of placed) if (p.item.id) rects[p.item.id] = p;
  const tips = new Set(placed.filter((p) => !isGroup(p.item) && p.item.shape === 'decision').map((p) => p.item.id!));
  const ids = fig.edges.map(edgeId);
  const routed = route(
    fig.edges.map((e, i) => ({ id: ids[i], from: e.from, to: e.to, around: e.around })),
    rects,
    tips,
  );
  const byId = Object.fromEntries(routed.map((r) => [r.id, r]));

  // The timeline: every beat of every step, in order, with the player's hold at the end of each step.
  const segs: Seg[] = [];
  const hops: { si: number; bi: number; lane: number; edge: string; back: boolean; data?: unknown; t0: number; t1: number }[] = [];
  let t = 0;
  beats.forEach((stepBeats, si) => {
    stepBeats.forEach((b, bi) => {
      const dur = (b.ms ?? fig.speed ?? 900) / 1000 / BASE_RATE;
      const last = bi === stepBeats.length - 1;
      b.hops.forEach((h, lane) => {
        if (byId[h.edge]) hops.push({ si, bi, lane, edge: h.edge, back: h.back, data: h.data, t0: t, t1: t + dur });
      });
      segs.push({ t0: t, t1: t + dur + (last ? speed * 1.5 : 0), si, bi });
      t += dur + (last ? speed * 1.5 : 0);
    });
  });
  const total = t || 1;

  // What the figure shows during each segment, matching the React player exactly.
  const shownAt = segs.map(
    ({ si, bi }) => Object.assign({}, ...beats[si].slice(0, bi + 1).map((b) => b.show)) as Record<string, FigContent>,
  );
  const litEdges = segs.map(({ si, bi }) => new Set(beats[si].slice(0, bi + 1).flatMap((b) => b.hops.map((h) => h.edge))));
  const litNodes = segs.map((seg, i) => {
    const on = new Set<string>([...(steps[seg.si].nodes ?? []), ...Object.keys(shownAt[i]), ...(beats[seg.si][seg.bi].light ?? [])]);
    fig.edges.forEach((e, ei) => {
      if (litEdges[i].has(ids[ei])) on.add(e.from).add(e.to);
    });
    return on;
  });
  const captions = segs.map(({ si, bi }) => {
    const said = beats[si].slice(0, bi + 1).findLastIndex((b) => b.say != null);
    return str(said === -1 ? steps[si].caption : beats[si][said].say);
  });

  // One class per on/off pattern over the segments, so 30 boxes share a handful of keyframes.
  const css: string[] = [];
  const seen = new Map<string, string>();
  /** The class that switches this element between `onCss` and `offCss` as the loop plays. */
  const anim = (on: boolean[], onCss: string, offCss: string, prefix: string): string => {
    if (!segs.length || on.every((x) => !x)) return '';
    const key = prefix + on.map((x) => (x ? 1 : 0)).join('');
    if (!seen.has(key)) {
      const name = `a${seen.size}`;
      seen.set(key, name);
      const frames = segs.map((s, i) => `${pct(s.t0 / total)},${pct(s.t1 / total - 0.0001)} { ${on[i] ? onCss : offCss} }`).join(' ');
      css.push(`@keyframes ${name} { ${frames} }\n.${name} { animation: ${name} ${n2(total)}s infinite step-end; }`);
    }
    return seen.get(key)!;
  };
  /** One `class` attribute from the static classes and the animated one — two would be invalid XML. */
  const cls = (...names: (string | false | undefined)[]) => {
    const list = names.filter(Boolean).join(' ');
    return list ? ` class="${list}"` : '';
  };

  const boxes = placed.map((p) => {
    const { item } = p;
    if (isGroup(item)) {
      if (item.label == null) return '';
      const on = item.id ? segs.map((_, i) => litNodes[i].has(item.id!)) : [];
      return (
        `<rect x="${n2(p.x)}" y="${n2(p.y)}" width="${n2(p.w)}" height="${n2(p.h)}" rx="14" fill="var(--surface)" stroke="var(--border)"` +
        cls(anim(on, 'stroke: var(--accent)', 'stroke: var(--border)', 'n')) +
        `/><text x="${n2(p.x + FRAME_SIDE)}" y="${n2(p.y + 20)}" class="frame">${esc(str(item.label).toUpperCase())}</text>`
      );
    }
    const on = segs.map((_, i) => litNodes[i].has(item.id));
    const stroke = cls(anim(on, 'stroke: var(--accent)', 'stroke: var(--border)', 'n'));
    const cx = p.x + p.w / 2;
    const contents = cards.get(item.id);
    const cardTop = p.y + p.h - 10 - (contents ? cardH.get(item.id)! : 0);
    const labelY = item.shape === 'store' ? p.y + 24 + 13 : contents ? p.y + 10 + 13 : p.y + p.h / 2 + (item.sub ? -2 : 5);
    const shape =
      item.shape === 'decision'
        ? `<polygon points="${n2(cx)},${n2(p.y)} ${n2(p.x + p.w)},${n2(p.y + p.h / 2)} ${n2(cx)},${n2(p.y + p.h)} ${n2(p.x)},${n2(p.y + p.h / 2)}" fill="var(--bg)" stroke="var(--border)"${stroke}/>`
        : item.shape === 'store'
          ? `<path d="M${n2(p.x)} ${n2(p.y + 12)} a ${n2(p.w / 2)} 12 0 0 1 ${n2(p.w)} 0 v ${n2(p.h - 24)} a ${n2(p.w / 2)} 12 0 0 1 ${n2(-p.w)} 0 z" fill="var(--bg)" stroke="var(--border)"${stroke}/>` +
            `<path d="M${n2(p.x)} ${n2(p.y + 12)} a ${n2(p.w / 2)} 12 0 0 0 ${n2(p.w)} 0" fill="none" stroke="var(--border)"${stroke}/>`
          : `<rect x="${n2(p.x)}" y="${n2(p.y)}" width="${n2(p.w)}" height="${n2(p.h)}" rx="10" fill="var(--bg)" stroke="var(--border)"${stroke}/>`;
    const label = `<text x="${n2(cx)}" y="${n2(labelY)}" class="label">${esc(str(item.label))}</text>`;
    const sub = item.sub ? `<text x="${n2(cx)}" y="${n2(labelY + SUB_LINE)}" class="sub">${esc(str(item.sub))}</text>` : '';
    return shape + label + sub + (contents ? card(p as Rect & { item: FigNode }, cardTop, contents) : '');
  });

  /** The dashed content card: one group per content it will ever hold, each visible on its own beats. */
  function card(p: Rect & { item: FigNode }, top: number, contents: FigContent[]): string {
    const id = p.item.id;
    const x = p.x + 10,
      w = p.w - 20,
      h = cardH.get(id)!;
    const filled = segs.map((_, i) => shownAt[i][id] != null);
    const box =
      `<rect x="${n2(x)}" y="${n2(top)}" width="${n2(w)}" height="${n2(h)}" rx="6" fill="var(--surface)" stroke="var(--border)" stroke-dasharray="3 3"` +
      cls(anim(filled, 'stroke: var(--accent); fill: var(--card-on)', 'stroke: var(--border); fill: var(--surface)', 'c')) +
      `/>`;
    // Every distinct content gets a layer; the segments decide which one is showing.
    const layers = contents
      .map((c, ci) => {
        const on = segs.map((_, i) => contents.indexOf(shownAt[i][id] as FigContent) === ci);
        if (on.every((x) => !x)) return '';
        return `<g opacity="0"${cls(anim(on, 'opacity: 1', 'opacity: 0', 'v'))}>${rows(c, x, top, w)}</g>`;
      })
      .join('');
    const empty = segs.map((_, i) => shownAt[i][id] == null);
    const dash = empty.some(Boolean)
      ? `<text x="${n2(x + CARD_SIDE)}" y="${n2(top + CARD_PAD + 11)}" opacity="0"${cls('row', 'muted', anim(empty, 'opacity: 1', 'opacity: 0', 'v'))}>—</text>`
      : '';
    return box + layers + dash;
  }

  /** One card's rows: a colored tag pill, the text, its muted meta, and the mark on the right. */
  function rows(c: FigContent, x: number, top: number, w: number): string {
    const { rows } = layoutCard(c, w);
    let y = top + CARD_PAD;
    return rows
      .map(({ row, heads, lines }) => {
        const tone = TONES[row.tone ?? 'blue'];
        const left = x + CARD_SIDE;
        const parts: string[] = [];
        const tagW = row.tag ? row.tag.length * em(9) + 8 : 0;
        const pill = (px: number, py: number) =>
          row.tag
            ? `<rect x="${n2(px)}" y="${n2(py)}" width="${n2(tagW)}" height="14" rx="4" fill="${tone}" fill-opacity="0.15"/>` +
              `<text x="${n2(px + tagW / 2)}" y="${n2(py + 10.5)}" class="tag" fill="${tone}">${esc(row.tag.toUpperCase())}</text>`
            : '';
        const mark = row.mark ? `<text x="${n2(x + w - CARD_SIDE)}" y="${n2(y + 11)}" class="mark">${esc(str(row.mark))}</text>` : '';
        if (heads) {
          parts.push(pill(left, y + 1), mark);
          y += LINE;
        }
        const indent = heads ? 0 : tagW ? tagW + 5 : 0;
        if (!heads) parts.push(pill(left, y + 1), mark);
        lines.forEach((line, li) => {
          parts.push(
            `<text x="${n2(left + (li === 0 ? indent : 0))}" y="${n2(y + 11)}"${cls('row', row.mono && 'mono')}>${esc(line)}</text>`,
          );
          y += LINE;
        });
        y += ROW_GAP;
        return parts.join('');
      })
      .join('');
  }

  const edgeSvg = routed.map((r) => {
    const e = fig.edges[ids.indexOf(r.id)];
    const on = segs.map((_, i) => litEdges[i].has(r.id));
    const hidden = e.quiet && !on.every(Boolean);
    const shown = hidden ? cls(anim(on, 'opacity: 1', 'opacity: 0', 'q')) : '';
    const lit = cls(anim(on, 'stroke: var(--accent); stroke-width: 2', 'stroke: var(--muted); stroke-width: 1.25', 'e'));
    const path = `<path id="p-${esc(r.id)}" d="${r.d}" fill="none" stroke="var(--muted)" stroke-width="1.25" marker-end="url(#arrow)"${lit}/>`;
    // A quiet edge is only drawn while a step uses it, so wrap the whole thing rather than the stroke.
    const label =
      e.label == null
        ? ''
        : (() => {
            const lw = str(e.label).length * em(11) + 14;
            return (
              `<rect x="${n2(r.mid.x - lw / 2)}" y="${n2(r.mid.y - 9)}" width="${n2(lw)}" height="18" rx="9" fill="var(--bg)" stroke="var(--border)"` +
              cls(anim(on, 'fill: var(--accent); stroke: var(--accent)', 'fill: var(--bg); stroke: var(--border)', 'l')) +
              `/><text x="${n2(r.mid.x)}" y="${n2(r.mid.y + 4)}"${cls('edgelabel', anim(on, 'fill: #fff', 'fill: var(--muted)', 'x'))}>${esc(str(e.label))}</text>`
            );
          })();
    return hidden ? `<g opacity="0"${shown}>${path}${label}</g>` : path + label;
  });

  // A packet per hop: it waits offstage, crosses its edge during its beat, then leaves. Any `data`
  // rides above it in a chip, which is how the figure says what is moving.
  const packets = hops.map((h, i) => {
    const t0 = h.t0 / total,
      t1 = h.t1 / total;
    const name = `p${i}`;
    css.push(
      `@keyframes ${name} { 0%,${pct(t0)} { opacity: 0 } ${pct(t0 + 0.0001)},${pct(t1 - 0.0001)} { opacity: 1 } ${pct(t1)},100% { opacity: 0 } }\n` +
        `.${name} { animation: ${name} ${n2(total)}s infinite step-end; }`,
    );
    const chip = (() => {
      const label = str(h.data);
      if (!label) return '';
      const lines = wrap(label, 210, 11.5);
      const w = Math.max(...lines.map((l) => l.length)) * em(11.5) + 18;
      const boxH = lines.length * 15 + 8;
      return (
        `<rect x="${n2(-w / 2)}" y="${n2(-boxH - 12)}" width="${n2(w)}" height="${n2(boxH)}" rx="8" fill="var(--accent)"/>` +
        lines.map((l, li) => `<text x="0" y="${n2(-boxH - 12 + 15 * li + 15)}" class="chip">${esc(l)}</text>`).join('')
      );
    })();
    return (
      `<g${cls(name)} opacity="0"><circle r="10" fill="var(--accent)" opacity="0.2"/><circle r="4.5" fill="var(--accent)"/>${chip}` +
      `<animateMotion dur="${n2(total)}s" repeatCount="indefinite" keyTimes="0;${n2(t0)};${n2(t1)};1" keyPoints="${h.back ? '1;1;0;0' : '0;0;1;1'}" calcMode="linear">` +
      `<mpath href="#p-${esc(h.edge)}" xlink:href="#p-${esc(h.edge)}"/></animateMotion></g>`
    );
  });

  const bounds = placed[0];
  const arcs = fig.edges.some((e) => e.around);
  const capLines = [...new Set(captions)].flatMap((c) => wrap(c, Math.max(560, bounds.w), 13.5).length);
  const capH = steps.length ? 30 + Math.max(0, ...capLines) * 20 : 0;
  const W = Math.max(bounds.w + pad * 2, 560);
  const H = bounds.h + pad * 2 + capH + (arcs ? 44 : 0);
  const shift = (W - (bounds.w + pad * 2)) / 2;

  // The step label and its narration, both switching with the beats.
  const labels = [...new Set(steps.map((s, i) => i))].map((si) => {
    const on = segs.map((s) => s.si === si);
    return `<text x="${n2(W / 2)}" y="${n2(H - capH + 16)}" opacity="0"${cls('steplabel', anim(on, 'opacity: 1', 'opacity: 0', 's'))}>${esc(str(steps[si].label))}</text>`;
  });
  const said = [...new Set(captions)].map((text) => {
    const on = captions.map((c) => c === text);
    const lines = wrap(text, Math.max(560, bounds.w), 13.5);
    return (
      `<g opacity="0"${cls(anim(on, 'opacity: 1', 'opacity: 0', 'y'))}>` +
      lines.map((l, li) => `<text x="${n2(W / 2)}" y="${n2(H - capH + 42 + li * 20)}" class="caption">${esc(l)}</text>`).join('') +
      `</g>`
    );
  });

  const t0 = { accent: '#0074d9', fg: '#1c1e21', muted: '#606770', bg: '#ffffff', surface: '#f5f7fa', border: '#d0d7de', ...opts.theme };
  return `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="${n2(W)}" height="${n2(H)}" viewBox="0 0 ${n2(W)} ${n2(H)}" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif">
<style>
svg { --accent:${t0.accent}; --fg:${t0.fg}; --muted:${t0.muted}; --bg:${t0.bg}; --surface:${t0.surface}; --border:${t0.border}; --card-on:#eef5fd; }
@media (prefers-color-scheme: dark) { svg { --accent:#3396e8; --fg:#e3e3e3; --muted:#9aa0a6; --bg:#1b1b1d; --surface:#242526; --border:#3a3b3c; --card-on:#1d2733; } }
.label { fill: var(--fg); font-size: 14px; font-weight: 500; text-anchor: middle; }
.sub { fill: var(--muted); font-size: 12px; text-anchor: middle; }
.frame { fill: var(--muted); font-size: 11px; font-weight: 600; letter-spacing: .04em; }
.edgelabel { fill: var(--muted); font-size: 11px; text-anchor: middle; font-family: ui-monospace, Menlo, monospace; }
.row { fill: var(--fg); font-size: 11px; }
.row.mono { font-size: 10.5px; font-family: ui-monospace, Menlo, monospace; }
.muted { fill: var(--muted); }
.tag { font-size: 9px; font-weight: 600; letter-spacing: .03em; text-anchor: middle; }
.mark { fill: var(--accent); font-size: 11px; font-weight: 600; text-anchor: end; }
.chip { fill: #fff; font-size: 11.5px; text-anchor: middle; }
.steplabel { fill: var(--fg); font-size: 13px; font-weight: 600; text-anchor: middle; font-family: ui-monospace, Menlo, monospace; }
.caption { fill: var(--muted); font-size: 13.5px; text-anchor: middle; }
${css.join('\n')}
</style>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 1 L 9 5 L 0 9 z" fill="var(--muted)"/></marker></defs>
<rect width="100%" height="100%" fill="var(--bg)"/>
<g transform="translate(${n2(shift)} ${arcs ? 44 : 0})">
${boxes.filter(Boolean).join('\n')}
${edgeSvg.join('\n')}
${packets.join('\n')}
</g>
${labels.join('\n')}
${said.join('\n')}
</svg>
`;
}

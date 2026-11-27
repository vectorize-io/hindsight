/**
 * The year's memory constellation for Hindsight Wrapped: the real memories of
 * the top banks (from the bank graph endpoint) and their links, laid out once
 * and kept as dots only: no text, no names, so it can go on the share card.
 */

/** 0 world, 1 experience, 2 observation. */
export type MapKind = 0 | 1 | 2;

export interface MemoryMap {
  /** Position in 0..1, kind, and its order in time (0 the earliest, 1 the latest). */
  nodes: { x: number; y: number; k: MapKind; t: number }[];
  links: [number, number][];
}

/** The part of a bank graph response the map reads. */
export interface BankGraph {
  edges?: { data: { source: string; target: string; weight?: number } }[];
  table_rows?: { id: string; fact_type?: string; mentioned_at?: string; created_at?: string }[];
}

const KINDS: Record<string, MapKind> = { world: 0, experience: 1, observation: 2 };
// Each memory keeps its strongest few links: the graph endpoint links almost everything.
const LINKS_PER_NODE = 3;
const ITERATIONS = 120;

/** A small seeded random, so the same year always lays out the same way. */
export function seeded(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * One constellation from several banks' graphs: the memories written in
 * [start, end), each bank a cluster of its own, linked by their strongest links,
 * then relaxed with a small force layout and scaled into 0..1.
 */
export function buildMemoryMap(graphs: BankGraph[], start: Date, end: Date): MemoryMap {
  const span = end.getTime() - start.getTime();
  const nodes: (MemoryMap["nodes"][number] & { vx: number; vy: number })[] = [];
  const index = new Map<string, number>();
  const links: [number, number][] = [];
  const rand = seeded(graphs.length * 7919 + 1);

  graphs.forEach((g, bank) => {
    const a = (bank / graphs.length) * Math.PI * 2;
    const cx = graphs.length > 1 ? Math.cos(a) : 0;
    const cy = graphs.length > 1 ? Math.sin(a) : 0;
    for (const row of g.table_rows ?? []) {
      const when = new Date(row.mentioned_at ?? row.created_at ?? "");
      const ms = when.getTime() - start.getTime();
      if (!(ms >= 0 && ms < span) || index.has(row.id)) continue;
      index.set(row.id, nodes.length);
      nodes.push({
        x: cx + (rand() - 0.5) * 0.8,
        y: cy + (rand() - 0.5) * 0.8,
        k: KINDS[row.fact_type ?? ""] ?? 0,
        t: ms / span,
        vx: 0,
        vy: 0,
      });
    }
    const best = new Map<number, { j: number; w: number }[]>();
    for (const { data: e } of g.edges ?? []) {
      const i = index.get(e.source);
      const j = index.get(e.target);
      if (i === undefined || j === undefined || i === j) continue;
      const list = best.get(i) ?? [];
      list.push({ j, w: e.weight ?? 0 });
      best.set(i, list);
    }
    const seen = new Set<string>();
    for (const [i, list] of best) {
      list.sort((p, q) => q.w - p.w);
      for (const { j } of list.slice(0, LINKS_PER_NODE)) {
        const key = i < j ? `${i}-${j}` : `${j}-${i}`;
        if (!seen.has(key)) {
          seen.add(key);
          links.push([i, j]);
        }
      }
    }
  });

  // ponytail: O(n²) repulsion per step; fine for the few hundred memories fetched.
  const n = nodes.length;
  for (let it = 0; it < ITERATIONS; it++) {
    const cool = 1 - it / ITERATIONS;
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) {
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const d2 = dx * dx + dy * dy + 1e-4;
        const f = 0.0006 / d2;
        nodes[i].vx += dx * f;
        nodes[i].vy += dy * f;
        nodes[j].vx -= dx * f;
        nodes[j].vy -= dy * f;
      }
    for (const [i, j] of links) {
      const dx = nodes[j].x - nodes[i].x;
      const dy = nodes[j].y - nodes[i].y;
      nodes[i].vx += dx * 0.02;
      nodes[i].vy += dy * 0.02;
      nodes[j].vx -= dx * 0.02;
      nodes[j].vy -= dy * 0.02;
    }
    for (const p of nodes) {
      // A pull towards the middle keeps the clusters together.
      p.vx -= p.x * 0.01;
      p.vy -= p.y * 0.01;
      const v = Math.hypot(p.vx, p.vy);
      const cap = 0.05 * cool + 0.002;
      const s = v > cap ? cap / v : 1;
      p.x += p.vx * s;
      p.y += p.vy * s;
      p.vx *= 0.5;
      p.vy *= 0.5;
    }
  }

  let [minX, minY, maxX, maxY] = [Infinity, Infinity, -Infinity, -Infinity];
  for (const p of nodes) {
    minX = Math.min(minX, p.x);
    maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
  }
  const size = Math.max(maxX - minX, maxY - minY, 1e-6);
  const round = (v: number) => Math.round(v * 1000) / 1000;
  // `t` becomes the memory's rank in time: the recent memories the graph returns
  // bunch up in the last weeks, and the map should still light up one by one.
  const order = nodes.map((p, i) => [p.t, i]).sort((a, b) => a[0] - b[0]);
  order.forEach(([, i], r) => (nodes[i].t = n > 1 ? r / (n - 1) : 0));
  return {
    nodes: nodes.map((p) => ({
      x: round(0.5 + (p.x - (minX + maxX) / 2) / size),
      y: round(0.5 + (p.y - (minY + maxY) / 2) / size),
      k: p.k,
      t: round(p.t),
    })),
    links,
  };
}

const KIND_COLORS = ["#5BB0FF", "#2DD4BF", "#E8EEFF"];

/** A soft glowing dot per kind, drawn once and stamped for every memory. */
const sprites = new Map<string, HTMLCanvasElement>();
function sprite(color: string): HTMLCanvasElement {
  let c = sprites.get(color);
  if (c) return c;
  c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d")!;
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, "#fff");
  grad.addColorStop(0.12, color);
  grad.addColorStop(0.35, `${color}55`);
  grad.addColorStop(1, `${color}00`);
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  sprites.set(color, c);
  return c;
}

/**
 * Paints the map into a box. `upTo` (0..1) shows only the memories written so
 * far in the year, so animating it lights the year up in order; `time` (s)
 * makes the lit memories twinkle.
 */
export function paintMemoryMap(
  ctx: CanvasRenderingContext2D,
  map: MemoryMap,
  box: { x: number; y: number; w: number; h: number },
  upTo = 1,
  time = 0
) {
  const side = Math.min(box.w, box.h);
  const ox = box.x + (box.w - side) / 2;
  const oy = box.y + (box.h - side) / 2;
  const px = (v: number) => ox + v * side;
  const py = (v: number) => oy + v * side;
  const dot = Math.max(12, side / 26);

  ctx.lineWidth = Math.max(0.6, side / 900);
  for (const [i, j] of map.links) {
    const a = map.nodes[i];
    const b = map.nodes[j];
    const lit = Math.min(upTo - a.t, upTo - b.t);
    if (lit < 0) continue;
    ctx.strokeStyle = `rgba(150,195,255,${0.06 + 0.16 * Math.min(1, lit * 12)})`;
    ctx.beginPath();
    ctx.moveTo(px(a.x), py(a.y));
    ctx.lineTo(px(b.x), py(b.y));
    ctx.stroke();
  }
  map.nodes.forEach((p, i) => {
    const age = upTo - p.t;
    if (age < 0) return;
    // A memory flares as it appears, then settles into a gentle twinkle.
    const flare = Math.max(0, 1 - age * 15);
    const twinkle = 0.75 + 0.25 * Math.sin(time * 1.7 + i * 2.39);
    const s = dot * (1 + flare * 1.6);
    ctx.globalAlpha = Math.min(1, twinkle + flare);
    ctx.drawImage(sprite(KIND_COLORS[p.k]), px(p.x) - s / 2, py(p.y) - s / 2, s, s);
  });
  ctx.globalAlpha = 1;
}

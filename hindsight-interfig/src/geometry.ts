export type Rect = { x: number; y: number; w: number; h: number };
export type Side = 'l' | 'r' | 't' | 'b';
export type Routed = { id: string; d: string; mid: { x: number; y: number } };

export type Around = 'above' | 'below';
type Pick = { id: string; a: Rect; b: Rect; sa: Side; sb: Side; from: string; to: string; around?: Around };

const cx = (r: Rect) => r.x + r.w / 2;
const cy = (r: Rect) => r.y + r.h / 2;

// Turns edges between measured boxes into curved SVG paths.
// Boxes stacked on top of each other connect bottom->top, otherwise side->side.
// Several edges leaving the same side of a box are spread out so they don't overlap,
// except on `tips` boxes (diamonds), where they all meet at the point.
// `around` makes an edge leave and enter from the top or bottom, arcing over whatever sits between.
export function route(
  edges: { id: string; from: string; to: string; around?: Around }[],
  rects: Record<string, Rect>,
  tips: Set<string> = new Set(),
): Routed[] {
  const picks: Pick[] = [];
  for (const e of edges) {
    const a = rects[e.from],
      b = rects[e.to];
    if (!a || !b) continue;
    const stacked = a.x < b.x + b.w && b.x < a.x + a.w;
    const [sa, sb]: Side[] = e.around
      ? e.around === 'above'
        ? ['t', 't']
        : ['b', 'b']
      : stacked
        ? a.y < b.y
          ? ['b', 't']
          : ['t', 'b']
        : a.x < b.x
          ? ['r', 'l']
          : ['l', 'r'];
    picks.push({ ...e, a, b, sa, sb });
  }

  // For every (box, side), list the edge ends that sit there, sorted by where the other end is.
  const ends = new Map<string, { pick: Pick; start: boolean }[]>();
  for (const pick of picks) {
    for (const start of [true, false]) {
      const key = (start ? pick.from : pick.to) + ':' + (start ? pick.sa : pick.sb);
      if (!ends.has(key)) ends.set(key, []);
      ends.get(key)!.push({ pick, start });
    }
  }
  const anchor = new Map<string, { x: number; y: number }>();
  for (const [key, list] of ends) {
    const tip = tips.has(key.slice(0, key.lastIndexOf(':')));
    const side = list[0].start ? list[0].pick.sa : list[0].pick.sb;
    const horiz = side === 't' || side === 'b';
    const other = (x: { pick: Pick; start: boolean }) => {
      const r = x.start ? x.pick.b : x.pick.a;
      return horiz ? cx(r) : cy(r);
    };
    list.sort((p, q) => other(p) - other(q)); // stable: ties keep declaration order
    list.forEach((x, i) => {
      const r = x.start ? x.pick.a : x.pick.b;
      const f = tip ? 0.5 : (i + 1) / (list.length + 1);
      const pt =
        side === 'l'
          ? { x: r.x, y: r.y + r.h * f }
          : side === 'r'
            ? { x: r.x + r.w, y: r.y + r.h * f }
            : side === 't'
              ? { x: r.x + r.w * f, y: r.y }
              : { x: r.x + r.w * f, y: r.y + r.h };
      anchor.set(x.pick.id + (x.start ? ':s' : ':e'), pt);
    });
  }

  return picks.map((p) => {
    const s = anchor.get(p.id + ':s')!,
      e = anchor.get(p.id + ':e')!;
    if (p.around) {
      // ponytail: arcs 50px past the two ends' boxes; a taller box in between can still be crossed.
      const y = p.around === 'above' ? Math.min(p.a.y, p.b.y) - 50 : Math.max(p.a.y + p.a.h, p.b.y + p.b.h) + 50;
      return {
        id: p.id,
        d: `M ${s.x} ${s.y} C ${s.x} ${y}, ${e.x} ${y}, ${e.x} ${e.y}`,
        mid: { x: (s.x + e.x) / 2, y: (s.y + 6 * y + e.y) / 8 },
      };
    }
    const horiz = p.sa === 'l' || p.sa === 'r';
    const k = (horiz ? Math.abs(e.x - s.x) : Math.abs(e.y - s.y)) / 2;
    const sign = p.sa === 'r' || p.sa === 'b' ? 1 : -1;
    const c1 = horiz ? `${s.x + sign * k} ${s.y}` : `${s.x} ${s.y + sign * k}`;
    const c2 = horiz ? `${e.x - sign * k} ${e.y}` : `${e.x} ${e.y - sign * k}`;
    return { id: p.id, d: `M ${s.x} ${s.y} C ${c1}, ${c2}, ${e.x} ${e.y}`, mid: { x: (s.x + e.x) / 2, y: (s.y + e.y) / 2 } };
  });
}

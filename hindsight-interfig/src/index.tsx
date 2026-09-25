'use client';
import { useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from 'react';
import { route, type Rect, type Routed } from './geometry';
import {
  decisions,
  edgeId,
  isGroup,
  isRows,
  toBeat,
  type FigContent,
  type FigGroup,
  type FigNode,
  type FigTheme,
  type FigTone,
  type FlowProps,
} from './model';

export type * from './model';

const DEFAULTS: Required<FigTheme> = {
  accent: '#0074d9',
  fg: '#1c1e21',
  muted: '#606770',
  bg: '#ffffff',
  surface: '#f5f7fa',
  border: '#d0d7de',
  font: 'inherit',
};
const v = (k: keyof FigTheme) => `var(--fig-${k}, ${DEFAULTS[k]})`;
const MONO = 'var(--ifm-font-family-monospace, ui-monospace, SFMono-Regular, Menlo, monospace)';
const glow = `0 0 0 3px color-mix(in srgb, ${v('accent')} 18%, transparent)`;

/** Boxes that get a content card: the ones some step fills with `show`. They get a fixed width so the layout never jumps. */
const CARD_WIDTH = 166;

const TONES: Record<FigTone, string> = {
  blue: '#3b82f6',
  purple: '#8b5cf6',
  green: '#10b981',
  orange: '#f59e0b',
  gray: '#8b949e',
};
const cardBody = (c: FigContent): ReactNode => {
  if (c == null) return '—';
  if (!isRows(c)) return c;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, whiteSpace: 'normal' }}>
      {c.map((r, i) => {
        const tone = TONES[r.tone ?? 'blue'];
        const tag = r.tag != null && (
          <span
            style={{
              flexShrink: 0,
              fontSize: 9,
              fontWeight: 600,
              letterSpacing: '.03em',
              textTransform: 'uppercase',
              padding: '0 4px',
              borderRadius: 4,
              lineHeight: '14px',
              color: tone,
              background: `color-mix(in srgb, ${tone} 15%, transparent)`,
            }}
          >
            {r.tag}
          </span>
        );
        const mark = r.mark != null && (
          <span style={{ flexShrink: 0, marginLeft: 'auto', color: v('accent'), fontWeight: 600 }}>{r.mark}</span>
        );
        const text = (
          <span
            style={{
              minWidth: 0,
              fontFamily: r.mono ? MONO : undefined,
              fontSize: r.mono ? 10.5 : undefined,
            }}
          >
            {r.text}
            {r.meta != null && <span style={{ color: v('muted') }}> · {r.meta}</span>}
          </span>
        );
        // A word-sized tag heads its row so the text keeps the full width; a number or no tag sits inline.
        return (r.tag?.length ?? 0) > 2 ? (
          <div key={i}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 1 }}>
              {tag}
              {mark}
            </div>
            {text}
          </div>
        ) : (
          <div key={i} style={{ display: 'flex', gap: 5, alignItems: 'baseline' }}>
            {tag}
            {text}
            {mark}
          </div>
        );
      })}
    </div>
  );
};

/** 1× plays a touch faster than the figures' own timings: they were written to be read slowly. */
const BASE_RATE = 1.25;

export function Flow({ layout, edges, steps = [], theme, speed = 900, autoplay = true }: FlowProps) {
  const root = useRef<HTMLDivElement>(null);
  const outer = useRef<HTMLDivElement>(null);
  const [fit, setFit] = useState({ scale: 1, height: 0 });
  const [full, setFull] = useState(false);
  useEffect(() => {
    if (!full) return;
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setFull(false);
    addEventListener('keydown', onKey);
    return () => removeEventListener('keydown', onKey);
  }, [full]);
  const dots = useRef<(SVGGElement | null)[]>([]);
  const chips = useRef<(HTMLDivElement | null)[]>([]);
  const bar = useRef<HTMLDivElement>(null); // the active tab's progress line
  const [beat, setBeat] = useState(0);
  const paths = useRef<Record<string, SVGPathElement | null>>({});
  const [routed, setRouted] = useState<Routed[]>([]);
  const [active, setActive] = useState<number | null>(steps.length ? 0 : null);
  const [playing, setPlaying] = useState(autoplay);
  // The step's clock lives outside React: pausing freezes it, resizing keeps it, only a new step resets it.
  const playingRef = useRef(playing);
  playingRef.current = playing;
  const [rate, setRate] = useState<1 | 2>(1);
  const rateRef = useRef(rate);
  rateRef.current = rate;
  const clock = useRef<{ beats: unknown; elapsed: number }>({ beats: null, elapsed: 0 });
  const [hover, setHover] = useState<string | null>(null);

  const ids = useMemo(() => edges.map(edgeId), [edges]);
  const tips = useMemo(() => new Set(decisions(layout)), [layout]);
  const step = active == null ? null : steps[active];
  const beats = useMemo(() => (step?.flow ?? []).map(toBeat), [step]);
  // Every content each box's card will ever show. The card is sized to the largest, so text never gets cut and nothing jumps.
  const carded = useMemo(() => {
    const all = new Map<string, FigContent[]>();
    for (const b of steps.flatMap((s) => s.flow.map(toBeat)))
      for (const [id, c] of Object.entries(b.show ?? {})) all.set(id, [...(all.get(id) ?? []), c]);
    return all;
  }, [steps]);
  const cur = beats[beat];
  // The narration: the latest `say` so far (a beat without one keeps the line before), else the step caption.
  const saidAt = beats.slice(0, beat + 1).findLastIndex((b) => b.say != null);
  const said = saidAt === -1 ? step?.caption : beats[saidAt].say;
  // Content cards: everything the step has shown so far, latest wins.
  const shown = useMemo(
    () => Object.assign({}, ...beats.slice(0, beat + 1).map((b) => b.show)) as Record<string, FigContent>,
    [beats, beat],
  );
  // The beat that last changed each card, so only a card whose content just changed animates in.
  const shownAt = useMemo(() => {
    const at: Record<string, number> = {};
    beats.slice(0, beat + 1).forEach((b, i) => Object.keys(b.show ?? {}).forEach((k) => (at[k] = i)));
    return at;
  }, [beats, beat]);

  // Boxes with many edges on one side get taller so the edges and their labels have room.
  const minHeight = useMemo(() => {
    const out: Record<string, number> = {},
      inn: Record<string, number> = {};
    for (const e of edges) {
      out[e.from] = (out[e.from] ?? 0) + 1;
      inn[e.to] = (inn[e.to] ?? 0) + 1;
    }
    return (id: string) => {
      const n = Math.max(out[id] ?? 0, inn[id] ?? 0);
      return n > 2 ? n * 30 : undefined;
    };
  }, [edges]);

  // Measure every box and redraw the edges whenever anything changes size.
  useEffect(() => {
    const el = root.current,
      box = outer.current;
    if (!el || !box) return;
    const measure = () => {
      // Too wide for its container? Shrink it, like mermaid does. Below half size, scroll instead.
      const scale = Math.max(0.5, Math.min(1, box.clientWidth / el.offsetWidth));
      setFit({ scale, height: el.offsetHeight * scale });
      const base = el.getBoundingClientRect();
      const k = base.width / el.offsetWidth; // the scale currently on screen; rects are measured unscaled
      const rects: Record<string, Rect> = {};
      el.querySelectorAll<HTMLElement>('[data-fig]').forEach((n) => {
        const r = n.getBoundingClientRect();
        rects[n.dataset.fig!] = {
          x: (r.left - base.left) / k,
          y: (r.top - base.top) / k,
          w: r.width / k,
          h: r.height / k,
        };
      });
      setRouted(
        route(
          edges.map((e, i) => ({ id: ids[i], from: e.from, to: e.to, around: e.around })),
          rects,
          tips,
        ),
      );
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    ro.observe(box);
    el.querySelectorAll('[data-fig]').forEach((n) => ro.observe(n));
    return () => ro.disconnect();
  }, [edges, ids, layout, tips]);

  // Play the step's beats: each moves its packets (and their data cards) along its edges, then the next step starts.
  useEffect(() => {
    const gs = dots.current,
      cs = chips.current;
    if (clock.current.beats !== beats) {
      clock.current = { beats, elapsed: 0 };
      setBeat(0);
    }
    if (!beats.length || !routed.length) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      if (bar.current) bar.current.style.transform = 'none';
      return void setBeat(beats.length - 1);
    }
    const ends = beats.reduce<number[]>((acc, b) => [...acc, (acc.at(-1) ?? 0) + (b.ms ?? speed)], []);
    const total = ends.at(-1)! + speed * 1.5; // hold on the last beat before moving on
    let raf = 0,
      last = performance.now(),
      shownBeat = -1;
    const tick = (now: number) => {
      if (playingRef.current) clock.current.elapsed += (now - last) * rateRef.current * BASE_RATE;
      last = now;
      if (clock.current.elapsed >= total) {
        if (steps.length > 1) return setActive((a) => ((a ?? 0) + 1) % steps.length);
        clock.current.elapsed = 0;
      }
      const t = clock.current.elapsed;
      if (bar.current) bar.current.style.transform = `scaleX(${Math.min(1, t / total)})`;
      const next = ends.findIndex((e) => t < e);
      const i = next === -1 ? beats.length - 1 : next;
      if (i !== shownBeat) setBeat((shownBeat = i));
      const start = i ? ends[i - 1] : 0;
      const f = Math.min(1, (t - start) / ((beats[i].ms ?? speed) * 0.8)); // arrive a little early, rest at the end
      const eased = f < 0.5 ? 2 * f * f : 1 - (-2 * f + 2) ** 2 / 2;
      const hops = t < ends.at(-1)! ? beats[i].hops : [];
      gs.forEach((g, j) => {
        const p = hops[j] && paths.current[hops[j].edge];
        const c = cs[j];
        if (!p) {
          if (g) g.style.opacity = '0';
          if (c) c.style.opacity = '0';
          return;
        }
        const pt = p.getPointAtLength((hops[j].back ? 1 - eased : eased) * p.getTotalLength());
        if (g) {
          g.setAttribute('transform', `translate(${pt.x} ${pt.y})`);
          g.style.opacity = '1';
        }
        if (c) {
          c.style.transform = `translate(${pt.x}px, ${pt.y}px) translate(-50%, calc(-100% - 12px))`;
          c.style.opacity = hops[j].data == null ? '0' : '1';
        }
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      gs.forEach((g) => g && (g.style.opacity = '0'));
      cs.forEach((c) => c && (c.style.opacity = '0'));
    };
  }, [beats, routed, speed, steps.length]);

  // What to light up: the step's edges and their ends, or whatever touches the hovered box.
  // A step lights the trail so far: every edge its beats have crossed, and every box it has filled.
  const trail = beats.slice(0, beat + 1).flatMap((b) => b.hops.map((h) => h.edge));
  const litEdges = new Set<string>(hover ? ids.filter((id, i) => edges[i].from === hover || edges[i].to === hover) : trail);
  const litNodes = new Set<string>(hover ? [hover] : [...(step?.nodes ?? []), ...Object.keys(shown), ...(cur?.light ?? [])]);
  edges.forEach((e, i) => {
    if (litEdges.has(ids[i])) litNodes.add(e.from).add(e.to);
  });
  const focus = hover != null || step != null;

  const vars = Object.fromEntries(Object.entries(theme ?? {}).map(([k, val]) => [`--fig-${k}`, val])) as CSSProperties;

  const renderItem = (item: FigNode | FigGroup, depth: number): ReactNode => {
    if (isGroup(item)) {
      const lit = item.id != null && litNodes.has(item.id);
      const framed = item.label != null;
      return (
        <div
          key={item.id ?? depth + String(item.label)}
          data-fig={item.id}
          style={{
            ...(framed && {
              background: v('surface'),
              border: `1px solid ${lit ? v('accent') : v('border')}`,
              boxShadow: lit ? glow : undefined,
              borderRadius: 14,
              padding: '10px 18px 18px',
            }),
            transition: 'border-color .25s, box-shadow .25s',
          }}
        >
          {item.label != null && (
            <div
              style={{
                fontSize: 12,
                fontWeight: 600,
                letterSpacing: '.04em',
                textTransform: 'uppercase',
                color: v('muted'),
                marginBottom: 12,
              }}
            >
              {item.label}
            </div>
          )}
          <div
            style={{
              display: 'flex',
              flexDirection: item.direction ?? 'row',
              gap: item.gap ?? (item.direction === 'column' ? 28 : 56),
              alignItems: item.align
                ? item.align === 'center'
                  ? 'center'
                  : 'flex-' + item.align
                : item.direction === 'column'
                  ? 'stretch'
                  : 'center',
              justifyContent: 'center',
            }}
          >
            {item.children.map((c) => renderItem(c, depth + 1))}
          </div>
        </div>
      );
    }
    const lit = litNodes.has(item.id);
    const diamond = item.shape === 'decision';
    const store = item.shape === 'store';
    const card = carded.has(item.id);
    return (
      <div
        key={item.id}
        data-fig={item.id}
        onMouseEnter={() => setHover(item.id)}
        onMouseLeave={() => setHover(null)}
        style={{
          position: 'relative',
          isolation: 'isolate',
          minWidth: 100,
          maxWidth: card ? undefined : 190,
          width: item.width ?? (card ? CARD_WIDTH : undefined),
          minHeight: minHeight(item.id),
          boxSizing: 'border-box',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          padding: diamond ? '22px 34px' : `${store ? 24 : 10}px ${card ? 10 : 16}px 10px`,
          textAlign: 'center',
          background: diamond ? undefined : v('bg'),
          color: v('fg'),
          border: diamond ? undefined : `1px solid ${lit ? v('accent') : v('border')}`,
          boxShadow: diamond ? undefined : lit ? glow : '0 1px 2px rgba(0,0,0,.06)',
          opacity: focus && !lit ? 0.7 : 1,
          borderRadius: store ? '50% / 12px' : 10,
          fontSize: 14,
          fontWeight: 500,
          transition: 'border-color .25s, box-shadow .25s, opacity .25s',
          cursor: 'default',
        }}
      >
        {diamond && (
          <svg
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
            style={{
              position: 'absolute',
              inset: 0,
              width: '100%',
              height: '100%',
              overflow: 'visible',
              zIndex: -1,
            }}
          >
            <polygon
              points="50,0 100,50 50,100 0,50"
              fill={v('bg')}
              stroke={lit ? v('accent') : v('border')}
              strokeWidth={lit ? 1.6 : 1}
              vectorEffect="non-scaling-stroke"
              style={{ transition: 'stroke .25s' }}
            />
          </svg>
        )}
        {store && (
          // the cylinder's top rim
          <div
            style={{
              position: 'absolute',
              top: -1,
              left: -1,
              right: -1,
              height: 24,
              boxSizing: 'border-box',
              borderRadius: '50%',
              border: `1px solid ${lit ? v('accent') : v('border')}`,
              transition: 'border-color .25s',
            }}
          />
        )}
        {item.label}
        {item.sub != null && <div style={{ fontSize: 12, fontWeight: 400, color: v('muted'), marginTop: 2 }}>{item.sub}</div>}
        {card && (
          <div
            key={`${active}-${shownAt[item.id] ?? 'empty'}`}
            style={{
              marginTop: 8,
              padding: '6px 8px',
              minHeight: item.lines != null ? item.lines * 15 + 13 : undefined,
              boxSizing: 'border-box',
              display: 'grid',
              borderRadius: 6,
              textAlign: 'left',
              fontSize: 11,
              lineHeight: '15px',
              fontWeight: 400,
              whiteSpace: 'pre-wrap',
              overflowWrap: 'anywhere',
              color: shown[item.id] != null ? v('fg') : v('muted'),
              background: shown[item.id] != null ? `color-mix(in srgb, ${v('accent')} 8%, ${v('bg')})` : v('surface'),
              border: `1px dashed ${shown[item.id] != null ? v('accent') : v('border')}`,
              animation: shown[item.id] != null ? 'interfig-in .35s ease-out' : undefined,
            }}
          >
            {/* Invisible copies of every content stack in one grid cell and set the size; the real one sits on top. */}
            {carded.get(item.id)!.map((c, i) => (
              <div key={i} aria-hidden style={{ gridArea: '1 / 1', visibility: 'hidden' }}>
                {cardBody(c)}
              </div>
            ))}
            <div style={{ gridArea: '1 / 1' }}>{cardBody(shown[item.id])}</div>
          </div>
        )}
      </div>
    );
  };

  const iconBtn: CSSProperties = {
    display: 'grid',
    placeItems: 'center',
    width: 30,
    height: 30,
    padding: 0,
    borderRadius: 999,
    cursor: 'pointer',
    border: `1px solid ${v('border')}`,
    background: v('bg'),
    color: v('muted'),
  };

  return (
    <figure
      className="interfig"
      style={{
        ...vars,
        position: full ? 'fixed' : 'relative',
        inset: full ? 0 : undefined,
        zIndex: full ? 1000 : undefined,
        overflow: full ? 'auto' : undefined,
        margin: full ? 0 : '24px 0',
        padding: full ? '48px 24px 24px' : '20px 16px 16px',
        border: full ? 'none' : `1px solid ${v('border')}`,
        borderRadius: full ? 0 : 16,
        background: v('bg'),
        fontFamily: v('font'),
        color: v('fg'),
      }}
    >
      <button
        type="button"
        aria-label={full ? 'Close full screen' : 'Full screen'}
        title={full ? 'Close (Esc)' : 'Full screen'}
        onClick={() => setFull((f) => !f)}
        style={{
          ...iconBtn,
          position: 'absolute',
          top: 10,
          right: 10,
          zIndex: 3,
          width: 28,
          height: 28,
        }}
      >
        <Icon d={full ? 'M4 4l8 8M12 4l-8 8' : 'M9 3h4v4M7 13H3V9M13 3L9 7M3 13l4-4'} />
      </button>
      <style>{'@keyframes interfig-in{from{opacity:0;transform:translateY(3px)}to{opacity:1;transform:none}}'}</style>
      {/* The canvas: a dotted grid that runs to the figure's edges, with a line under it. */}
      <div
        style={{
          margin: full ? '-48px -24px 0' : '-20px -16px 0',
          padding: full ? '48px 24px 24px' : '24px 16px',
          borderRadius: full ? 0 : '16px 16px 0 0',
          borderBottom: `1px solid ${v('border')}`,
          backgroundImage: `radial-gradient(color-mix(in srgb, ${v('fg')} 16%, transparent) 1px, transparent 1.2px)`,
          backgroundSize: '14px 14px',
        }}
      >
        {/* ponytail: rows never wrap; wide figures shrink to fit, down to half size. Add a stacked mobile layout if that bites. */}
        <div
          ref={outer}
          style={{
            overflow: fit.scale > 0.5 ? 'hidden' : 'auto',
            height: fit.scale < 1 ? fit.height : undefined,
          }}
        >
          <div
            ref={root}
            style={{
              position: 'relative',
              width: 'max-content',
              margin: '0 auto',
              transform: fit.scale < 1 ? `scale(${fit.scale})` : undefined,
              transformOrigin: 'top left',
              padding: 4,
              // room for edges that arc over or under the boxes
              paddingTop: edges.some((e) => e.around === 'above') ? 44 : 4,
              paddingBottom: edges.some((e) => e.around === 'below') ? 44 : 4,
            }}
          >
            {renderItem(layout, 0)}
            <svg
              style={{
                position: 'absolute',
                inset: 0,
                width: '100%',
                height: '100%',
                overflow: 'visible',
                pointerEvents: 'none',
              }}
            >
              <defs>
                {(['off', 'on'] as const).map((k) => (
                  <marker
                    key={k}
                    id={`fig-arrow-${k}`}
                    viewBox="0 0 10 10"
                    refX="9"
                    refY="5"
                    markerWidth="7"
                    markerHeight="7"
                    orient="auto-start-reverse"
                  >
                    <path d="M 0 1 L 9 5 L 0 9 z" fill={k === 'on' ? v('accent') : v('muted')} />
                  </marker>
                ))}
              </defs>
              {routed.map((r) => {
                const on = litEdges.has(r.id);
                const hidden = !on && edges[ids.indexOf(r.id)].quiet;
                return (
                  <path
                    key={r.id}
                    ref={(p) => {
                      paths.current[r.id] = p;
                    }}
                    d={r.d}
                    fill="none"
                    stroke={on ? v('accent') : v('muted')}
                    strokeWidth={on ? 2 : 1.25}
                    strokeOpacity={hidden ? 0 : focus && !on ? 0.35 : 1}
                    markerEnd={hidden ? undefined : `url(#fig-arrow-${on ? 'on' : 'off'})`}
                    style={{ transition: 'stroke .25s, stroke-opacity .25s' }}
                  />
                );
              })}
              {Array.from({ length: Math.max(1, ...beats.map((b) => b.hops.length)) }, (_, j) => (
                <g
                  key={j}
                  ref={(g) => {
                    dots.current[j] = g;
                  }}
                  style={{ opacity: 0 }}
                >
                  <circle r={10} fill={v('accent')} opacity={0.2} />
                  <circle r={4.5} fill={v('accent')} />
                </g>
              ))}
            </svg>
            {Array.from({ length: Math.max(1, ...beats.map((b) => b.hops.length)) }, (_, j) => (
              <div
                key={j}
                ref={(c) => {
                  chips.current[j] = c;
                }}
                style={{
                  position: 'absolute',
                  left: 0,
                  top: 0,
                  zIndex: 2,
                  opacity: 0,
                  maxWidth: 220,
                  width: 'max-content',
                  padding: '4px 9px',
                  borderRadius: 8,
                  fontSize: 11.5,
                  lineHeight: '15px',
                  pointerEvents: 'none',
                  background: v('accent'),
                  color: '#fff',
                  boxShadow: '0 4px 14px rgba(0,0,0,.18)',
                  transition: 'opacity .2s',
                }}
              >
                {cur?.hops[j]?.data}
              </div>
            ))}
            {routed.map((r) => {
              const e = edges[ids.indexOf(r.id)];
              if (e.label == null) return null;
              const on = litEdges.has(r.id);
              return (
                <div
                  key={r.id}
                  style={{
                    position: 'absolute',
                    left: r.mid.x,
                    top: r.mid.y,
                    transform: 'translate(-50%, -50%)',
                    fontSize: 11,
                    lineHeight: '16px',
                    padding: '0 7px',
                    borderRadius: 999,
                    whiteSpace: 'nowrap',
                    pointerEvents: 'none',
                    background: on ? v('accent') : v('bg'),
                    color: on ? '#fff' : v('muted'),
                    border: `1px solid ${on ? v('accent') : v('border')}`,
                    opacity: !on && e.quiet ? 0 : focus && !on ? 0.6 : 1,
                    fontFamily: MONO,
                    transition: 'background .25s, color .25s, opacity .25s',
                  }}
                >
                  {e.label}
                </div>
              );
            })}
          </div>
        </div>
      </div>
      {steps.length > 0 && (
        <figcaption style={{ marginTop: 14, textAlign: 'center' }}>
          <div style={{ display: 'flex', gap: 10, justifyContent: 'center', alignItems: 'center' }}>
            <button
              type="button"
              aria-label={playing ? 'Pause' : 'Play'}
              title={playing ? 'Pause' : 'Play'}
              style={iconBtn}
              onClick={() => setPlaying((p) => !p)}
            >
              {/* pause: two bars; play: a triangle */}
              <Icon d={playing ? 'M5.5 4v8M10.5 4v8' : 'M5 3.5v9l7.5-4.5z'} fill={!playing} />
            </button>
            {/* Tabs in a quiet track; the active one carries a progress line for the step that is playing. */}
            <div
              role="tablist"
              style={{
                display: 'inline-flex',
                gap: 2,
                padding: 3,
                borderRadius: 10,
                background: v('surface'),
                border: `1px solid ${v('border')}`,
              }}
            >
              {steps.map((s, i) => {
                const on = active === i;
                return (
                  <button
                    key={i}
                    type="button"
                    role="tab"
                    aria-selected={on}
                    onClick={() => {
                      clock.current.elapsed = 0; // replay from the start, even when it is already the active step
                      setActive(i);
                      setPlaying(true);
                      setBeat(0);
                    }}
                    style={{
                      position: 'relative',
                      overflow: 'hidden',
                      font: 'inherit',
                      fontFamily: MONO,
                      fontSize: 12.5,
                      padding: '5px 14px',
                      borderRadius: 7,
                      border: 'none',
                      cursor: 'pointer',
                      background: on ? v('bg') : 'transparent',
                      color: on ? v('fg') : v('muted'),
                      boxShadow: on ? '0 1px 2px rgba(0,0,0,.08)' : 'none',
                      transition: 'background .2s, color .2s',
                    }}
                  >
                    {s.label}
                    {on && (
                      <div
                        ref={bar}
                        style={{
                          position: 'absolute',
                          left: 0,
                          right: 0,
                          bottom: 0,
                          height: 2,
                          background: v('accent'),
                          transformOrigin: 'left',
                          transform: 'scaleX(0)',
                          opacity: playing ? 1 : 0.35,
                        }}
                      />
                    )}
                  </button>
                );
              })}
            </div>
            <button
              type="button"
              aria-label={`Speed ${rate}×, switch to ${rate === 1 ? 2 : 1}×`}
              title="Playback speed"
              style={{
                ...iconBtn,
                width: 'auto',
                padding: '0 10px',
                font: 'inherit',
                fontFamily: MONO,
                fontSize: 12,
                color: rate === 2 ? v('accent') : v('muted'),
                borderColor: rate === 2 ? v('accent') : v('border'),
              }}
              onClick={() => setRate((r) => (r === 1 ? 2 : 1))}
            >
              {rate}×
            </button>
          </div>
          {said != null && (
            <div
              key={`${active}-${saidAt}`}
              style={{
                marginTop: 12,
                fontSize: 13.5,
                lineHeight: 1.5,
                color: v('muted'),
                minHeight: '3em',
                maxWidth: 640,
                marginInline: 'auto',
                animation: 'interfig-in .35s ease-out',
              }}
            >
              {said}
            </div>
          )}
        </figcaption>
      )}
    </figure>
  );
}

/** A tiny node-link drawing for a content card: entities and the links between them. `lit` names glow. */
export function MiniGraph({ nodes, links, lit = [] }: { nodes: string[]; links: [string, string][]; lit?: string[] }) {
  const W = 140,
    H = 62;
  // ponytail: nodes sit on an ellipse; fine for the handful a card can hold.
  const at = (i: number) => {
    const a = (i / nodes.length) * 2 * Math.PI - Math.PI / 2;
    return { x: W / 2 + (W / 2 - 26) * Math.cos(a), y: H / 2 + (H / 2 - 12) * Math.sin(a) };
  };
  const pos = Object.fromEntries(nodes.map((n, i) => [n, at(i)]));
  const on = new Set(lit);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ display: 'block', overflow: 'visible', width: '100%', maxWidth: W, height: 'auto' }}>
      {links.map(([a, b]) => (
        <line
          key={a + b}
          x1={pos[a].x}
          y1={pos[a].y}
          x2={pos[b].x}
          y2={pos[b].y}
          stroke={on.has(a) && on.has(b) ? v('accent') : v('muted')}
          strokeWidth={on.has(a) && on.has(b) ? 2 : 1}
        />
      ))}
      {nodes.map((n) => (
        <g key={n}>
          <circle
            cx={pos[n].x}
            cy={pos[n].y}
            r={4}
            fill={on.has(n) ? v('accent') : v('bg')}
            stroke={on.has(n) ? v('accent') : v('muted')}
          />
          <text x={pos[n].x} y={pos[n].y - 7} textAnchor="middle" fontSize={9.5} fill={v('fg')}>
            {n}
          </text>
        </g>
      ))}
    </svg>
  );
}

/** A 16px stroke icon drawn from one SVG path. */
function Icon({ d, fill = false }: { d: string; fill?: boolean }) {
  return (
    <svg
      width={14}
      height={14}
      viewBox="0 0 16 16"
      fill={fill ? 'currentColor' : 'none'}
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d={d} />
    </svg>
  );
}

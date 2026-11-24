"use client";

/**
 * The night sky of Hindsight Wrapped: a twinkling starfield behind the story,
 * and the constellations each slide draws from its own numbers (one star per
 * bank, per month, per hour of the day, per name). Seeded, so a year always
 * draws the same sky.
 */
import { useEffect, useId, useRef } from "react";

const BLUE = "#5BB0FF";
const TEAL = "#2DD4BF";

/** A small seeded random (mulberry32): the same seed draws the same sky. */
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

export interface Star {
  x: number;
  y: number;
  r: number;
  tone?: "blue" | "teal" | "white" | "dim";
  halo?: boolean;
  label?: string;
  sub?: string;
}

const TONES = { blue: BLUE, teal: TEAL, white: "#fff", dim: "rgba(255,255,255,0.35)" };

/** The year as a constellation: one star per month, higher and bigger the busier it was. */
export function yearStars(bars: number[], w = 100, h = 60): Star[] {
  const max = Math.max(1, ...bars);
  const peak = bars.indexOf(max);
  const rand = seeded(2026);
  return bars.map((v, i) => {
    const k = v / max;
    return {
      x: w * (0.06 + (i / 11) * 0.88),
      y: h * (0.82 - k * 0.62) + (rand() - 0.5) * h * 0.06,
      r: v ? 0.8 + 2.4 * Math.sqrt(k) : 0.45,
      tone: i === peak ? "blue" : v ? "white" : "dim",
      halo: i === peak,
    };
  });
}

/** Links each star to the nearest one before it: a tree, which reads as a constellation. */
export function nearestLinks(stars: Star[]): [number, number][] {
  const links: [number, number][] = [];
  for (let i = 1; i < stars.length; i++) {
    let best = 0;
    let bestD = Infinity;
    for (let j = 0; j < i; j++) {
      const d = (stars[i].x - stars[j].x) ** 2 + (stars[i].y - stars[j].y) ** 2;
      if (d < bestD) [best, bestD] = [j, d];
    }
    links.push([best, i]);
  }
  return links;
}

/** A constellation: lines draw in, stars pop in and keep twinkling. */
export function Constellation({
  stars,
  links = [],
  w = 100,
  h = 100,
  delay = 200,
  labelSize = 3.4,
  className = "",
  children,
}: {
  stars: Star[];
  links?: [number, number][];
  w?: number;
  h?: number;
  delay?: number;
  labelSize?: number;
  className?: string;
  children?: React.ReactNode;
}) {
  const id = useId().replace(/:/g, "");
  const rand = seeded(stars.length * 31 + 7);
  const step = Math.min(90, 900 / Math.max(1, stars.length));
  const linesAt = delay + Math.min(500, stars.length * step);
  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className={`w-full h-auto overflow-visible ${className}`}
      aria-hidden
    >
      <defs>
        <filter id={`glow-${id}`} x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="0.8" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <radialGradient id={`halo-${id}`}>
          <stop offset="0%" stopColor={BLUE} stopOpacity="0.55" />
          <stop offset="100%" stopColor={BLUE} stopOpacity="0" />
        </radialGradient>
      </defs>
      {children}
      {links.map(([a, b], i) => (
        <line
          key={`${a}-${b}`}
          x1={stars[a].x}
          y1={stars[a].y}
          x2={stars[b].x}
          y2={stars[b].y}
          pathLength={1}
          stroke="rgba(160,200,255,0.35)"
          strokeWidth={0.25}
          className="wrapped-line"
          style={{ animationDelay: `${linesAt + i * 70}ms` }}
        />
      ))}
      <g filter={`url(#glow-${id})`}>
        {stars.map((s, i) => {
          const d = delay + i * step;
          return (
            <g key={i}>
              {s.halo && (
                <circle
                  cx={s.x}
                  cy={s.y}
                  r={s.r * 5}
                  fill={`url(#halo-${id})`}
                  className="wrapped-star"
                  style={{ animationDelay: `${d}ms, ${d + 700}ms` }}
                />
              )}
              <circle
                cx={s.x}
                cy={s.y}
                r={s.r}
                fill={TONES[s.tone ?? "white"]}
                className="wrapped-star"
                style={{
                  animationDelay: `${d}ms, ${d + 700 + rand() * 2000}ms`,
                  animationDuration: `700ms, ${2.4 + rand() * 2.4}s`,
                }}
              />
            </g>
          );
        })}
      </g>
      {stars.map((s, i) =>
        s.label ? (
          <g
            key={`l${i}`}
            className="wrapped-fade"
            style={{ animationDelay: `${linesAt + 200 + i * 90}ms` }}
          >
            <text
              x={s.x}
              y={s.y + s.r + labelSize * 1.5}
              textAnchor="middle"
              fontSize={labelSize}
              fontWeight={600}
              fill="#fff"
            >
              {s.label}
            </text>
            {s.sub && (
              <text
                x={s.x}
                y={s.y + s.r + labelSize * 2.8}
                textAnchor="middle"
                fontSize={labelSize * 0.75}
                fill="rgba(220,232,255,0.6)"
              >
                {s.sub}
              </text>
            )}
          </g>
        ) : null
      )}
    </svg>
  );
}

/** Memories as a spiral galaxy: facts blue, observations teal, in their real proportion. */
export function Galaxy({ facts, observations }: { facts: number; observations: number }) {
  const total = facts + observations;
  // ponytail: capped at 520 dots; past that the galaxy only gets denser, not truer.
  const n = Math.min(520, Math.max(60, Math.round(Math.sqrt(total) * 6)));
  const teal = total ? Math.round((observations / total) * n) : 0;
  const rand = seeded(total || 1);
  const stars: Star[] = Array.from({ length: n }, (_, k) => {
    const t = Math.sqrt((k + 1) / n);
    const arm = k % 2 ? Math.PI : 0;
    const a = t * 9 + arm + (rand() - 0.5) * 0.7;
    const rr = t * 46 + (rand() - 0.5) * 4;
    return {
      x: 50 + Math.cos(a) * rr,
      y: 50 + Math.sin(a) * rr,
      r: 0.25 + rand() ** 3 * 0.9,
      // Observations are spread through the galaxy, not bunched in one arm.
      tone: (k * 7919) % n < teal ? "teal" : "blue",
    };
  });
  stars.unshift({ x: 50, y: 50, r: 2.2, tone: "white", halo: true });
  return (
    <div className="wrapped-spin-slow">
      <Constellation stars={stars} h={100} delay={100} />
    </div>
  );
}

/** The day as a dial: 24 stars round a circle, sized by the memories written in that hour. */
export function HourDial({ hourly, peak }: { hourly: number[]; peak: number | null }) {
  const max = Math.max(1, ...hourly);
  const stars: Star[] = hourly.map((v, h) => {
    const a = (h / 24) * Math.PI * 2 - Math.PI / 2;
    return {
      x: 50 + Math.cos(a) * 38,
      y: 50 + Math.sin(a) * 38,
      r: v ? 0.7 + 2.6 * Math.sqrt(v / max) : 0.4,
      tone: h === peak ? "blue" : v ? "white" : "dim",
      halo: h === peak,
    };
  });
  const links: [number, number][] = [];
  for (let h = 0; h < 24; h++) if (hourly[h] && hourly[(h + 1) % 24]) links.push([h, (h + 1) % 24]);
  return (
    <Constellation stars={stars} links={links}>
      <circle
        cx={50}
        cy={50}
        r={38}
        fill="none"
        stroke="rgba(255,255,255,0.06)"
        strokeWidth={0.3}
      />
      {[0, 6, 12, 18].map((h) => {
        const a = (h / 24) * Math.PI * 2 - Math.PI / 2;
        return (
          <text
            key={h}
            x={50 + Math.cos(a) * 28}
            y={50 + Math.sin(a) * 28 + 1.2}
            textAnchor="middle"
            fontSize={3.2}
            fill="rgba(220,232,255,0.45)"
          >
            {String(h).padStart(2, "0")}
          </text>
        );
      })}
    </Constellation>
  );
}

/** A logo (or a lone star) with two rings of stars turning around it. */
export function Orbit({ src }: { src: string | null }) {
  const ring = (n: number, r: number, seed: number): Star[] => {
    const rand = seeded(seed);
    return Array.from({ length: n }, (_, i) => {
      const a = (i / n) * Math.PI * 2 + rand() * 0.3;
      return { x: 50 + Math.cos(a) * r, y: 50 + Math.sin(a) * r, r: 0.4 + rand() * 0.9 };
    });
  };
  return (
    <div className="relative">
      <div className="wrapped-spin">
        <Constellation stars={ring(14, 30, 1)} />
      </div>
      <div className="wrapped-spin-reverse absolute inset-0">
        <Constellation stars={ring(22, 45, 2)} delay={500} />
      </div>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="wrapped-rise inline-flex size-[22%] items-center justify-center rounded-xl bg-white shadow-[0_0_80px_rgba(91,176,255,0.55)]">
          {src ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={src} alt="" className="size-[62%] object-contain" />
          ) : (
            <span className="size-1/3 rounded-full bg-[#5BB0FF]" />
          )}
        </span>
      </div>
    </div>
  );
}

/** A slowly drifting, twinkling starfield with the odd shooting star, behind every slide. */
export function Starfield({ count }: { count: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const rand = seeded(count);
    const stars = Array.from({ length: count }, () => ({
      x: rand(),
      y: rand(),
      r: 0.3 + rand() ** 3 * 1.5,
      p: rand() * Math.PI * 2,
      s: 0.4 + rand() * 1.4,
    }));
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0;
    const frame = (now: number) => {
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (canvas.width !== Math.round(w * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const t = now / 1000;
      ctx.fillStyle = "#fff";
      for (const st of stars) {
        const y = (((st.y - t * 0.003 * st.s) % 1) + 1) % 1;
        ctx.globalAlpha = 0.15 + 0.55 * (0.5 + 0.5 * Math.sin(t * st.s + st.p));
        ctx.beginPath();
        ctx.arc(st.x * w, y * h, st.r, 0, Math.PI * 2);
        ctx.fill();
      }
      // A shooting star for the first second of every eight.
      const k = (t % 8) / 1.1;
      if (!still && k < 1) {
        const r2 = seeded(Math.floor(t / 8) + 1);
        const x0 = r2() * w * 0.7;
        const y0 = r2() * h * 0.4;
        const x = x0 + k * w * 0.3;
        const y = y0 + k * w * 0.12;
        const g = ctx.createLinearGradient(x, y, x - 160, y - 64);
        g.addColorStop(0, "rgba(255,255,255,0.9)");
        g.addColorStop(1, "rgba(91,176,255,0)");
        ctx.globalAlpha = 1 - k;
        ctx.strokeStyle = g;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x - 160, y - 64);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      if (!still) raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [count]);
  return (
    <canvas ref={ref} aria-hidden className="pointer-events-none absolute inset-0 h-full w-full" />
  );
}

/** Faint stars over the whole share card. */
export function drawCardStars(ctx: CanvasRenderingContext2D, W: number, H: number) {
  const rand = seeded(1350);
  ctx.fillStyle = "#fff";
  for (let i = 0; i < 220; i++) {
    ctx.globalAlpha = 0.1 + rand() * 0.45;
    ctx.beginPath();
    ctx.arc(rand() * W, rand() * H, 0.6 + rand() ** 3 * 2.4, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

/** The year's constellation on the share card, in a box (counts per month only, no names). */
export function drawCardYear(
  ctx: CanvasRenderingContext2D,
  bars: number[],
  box: { x: number; y: number; w: number; h: number }
) {
  const stars = yearStars(bars, box.w, box.h).map((s) => ({
    ...s,
    x: s.x + box.x,
    y: s.y + box.y,
  }));
  const unit = box.w / 100;
  ctx.strokeStyle = "rgba(160,200,255,0.35)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  stars.forEach((s, i) => (i ? ctx.lineTo(s.x, s.y) : ctx.moveTo(s.x, s.y)));
  ctx.stroke();
  for (const s of stars) {
    if (s.halo) {
      const g = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, s.r * unit * 6);
      g.addColorStop(0, "rgba(91,176,255,0.6)");
      g.addColorStop(1, "rgba(91,176,255,0)");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r * unit * 6, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.shadowColor = s.tone === "blue" ? BLUE : "rgba(255,255,255,0.8)";
    ctx.shadowBlur = 18;
    ctx.fillStyle = TONES[s.tone ?? "white"];
    ctx.beginPath();
    ctx.arc(s.x, s.y, s.r * unit, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.shadowBlur = 0;
}

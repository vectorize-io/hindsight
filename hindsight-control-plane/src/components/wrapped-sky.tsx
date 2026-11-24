"use client";

/**
 * The constellations of Hindsight Wrapped: the year's real memories lighting up
 * in the order they were written, and the small star charts other slides draw
 * from their own numbers (per month, per hour of the day, per name).
 */
import { useEffect, useId, useRef } from "react";
import { paintMemoryMap, seeded, type MemoryMap } from "@/lib/memory-map";

const BLUE = "#5BB0FF";
const TEAL = "#2DD4BF";

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

/** The year's memories lighting up in the order they were written, then twinkling. */
export function MemoryMapCanvas({ map }: { map: MemoryMap }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const UNFOLD_MS = 4500;
    let raf = 0;
    let t0 = 0;
    const frame = (now: number) => {
      t0 ||= now;
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      if (canvas.width !== Math.round(w * dpr)) canvas.width = canvas.height = Math.round(w * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, w);
      const k = still ? 1 : Math.min(1, (now - t0) / UNFOLD_MS);
      // Eased, so the year starts slowly and the last months rush in.
      paintMemoryMap(ctx, map, { x: 0, y: 0, w, h: w }, 1 - (1 - k) ** 2, still ? 0 : now / 1000);
      if (!still) raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [map]);
  return <canvas ref={ref} aria-hidden className="aspect-square w-full" />;
}

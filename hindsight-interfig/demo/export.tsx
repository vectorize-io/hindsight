import { useEffect, useState } from 'react';
import { Flow, type Figure } from '../src';

/** The page `scripts/export.mjs` records: one figure, alone on the page, one step at a time.
 *  It stays blank until the recorder calls `startExport`, so no frames are wasted before the step begins. */
const DARK = { accent: '#3396e8', fg: '#e3e3e3', muted: '#9aa0a6', bg: '#1b1b1d', surface: '#242526', border: '#3a3b3c' };
/** A clip is watched small in a feed: darker text and borders than the docs site uses… */
const BOLD = { fg: '#0b1220', muted: '#374151', border: '#94a3b8' };
/** …every box and edge at full strength, and a bigger caption and card text. */
const BOLD_CSS = `
.interfig [data-fig]{border-width:1.5px !important;opacity:1 !important}
.interfig svg path[stroke-opacity="0.35"]{stroke-opacity:.85}
.interfig figcaption > div:last-child{font-size:19px !important;line-height:1.4 !important;color:var(--fig-fg) !important;max-width:820px !important}
.interfig [data-fig] > div[style*="dashed"]{font-size:12.5px !important;line-height:17px !important}
`;
const ZOOM = 1.3;

/** A camera that follows the moving packets in close, and pulls back out between hops.
 *  It moves the figure's canvas only, so the step label and narration under it stay in view. */
function useCamera() {
  useEffect(() => {
    const cam = { x: 0, y: 0, s: 1 };
    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const canvas = document.querySelector<HTMLElement>('.interfig > div');
      const inner = canvas?.firstElementChild as HTMLElement | null;
      if (!canvas || !inner) return;
      canvas.style.overflow = 'hidden';
      inner.style.transformOrigin = '0 0';
      const W = inner.offsetWidth,
        H = inner.offsetHeight;
      if (!cam.x) Object.assign(cam, { x: W / 2, y: H / 2 });
      // The packets are the dots the player moves: whichever are visible now are where the action is.
      // Read back into unzoomed coordinates, so the camera never chases its own zoom.
      const box = inner.getBoundingClientRect();
      const pts = [...inner.querySelectorAll<SVGGElement>('svg g')]
        .filter((g) => g.style.opacity === '1')
        .map((g) => {
          const r = g.getBoundingClientRect();
          return { x: (r.left + r.width / 2 - box.left) / cam.s, y: (r.top + r.height / 2 - box.top) / cam.s };
        });
      let target = { x: W / 2, y: H / 2, s: 1 };
      if (pts.length) {
        const xs = pts.map((p) => p.x),
          ys = pts.map((p) => p.y);
        const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
        // Fit every packet with room around it, never closer than ZOOM.
        const s = Math.max(1, Math.min(ZOOM, W / (x1 - x0 + 1000), H / (y1 - y0 + 700)));
        target = { x: (x0 + x1) / 2, y: (y0 + y1) / 2, s };
      }
      const k = 0.045; // ease: the camera drifts after the packet instead of snapping to it
      cam.x += (target.x - cam.x) * k;
      cam.y += (target.y - cam.y) * k;
      cam.s += (target.s - cam.s) * k;
      // Keep the view inside the figure: no empty margin at the edges.
      const tx = Math.min(0, Math.max(W - W * cam.s, W / 2 - cam.x * cam.s));
      const ty = Math.min(0, Math.max(H - H * cam.s, H / 2 - cam.y * cam.s));
      inner.style.transform = `translate(${tx}px, ${ty}px) scale(${cam.s})`;
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);
}

declare global {
  interface Window {
    /** Play `step` of the figure. The recorder calls this once it is recording. */
    startExport: (step: number) => void;
    /** The step labels, so the recorder can name the files and know how many clips to make. */
    exportSteps: string[];
    /** The figure's title and, per step, the narration it speaks — the recorder writes it beside the clip. */
    exportNarration: { title: string; steps: { label: string; says: string[] }[] };
  }
}

export function ExportPage({ figure, dark }: { figure: Figure; dark: boolean }) {
  const [step, setStep] = useState<number | null>(null);
  useCamera();

  useEffect(() => {
    window.exportSteps = (figure.props.steps ?? []).map((s, i) => (typeof s.label === 'string' ? s.label : `step-${i + 1}`));
    window.startExport = setStep;
    window.exportNarration = {
      title: figure.title,
      steps: (figure.props.steps ?? []).map((s, i) => ({
        label: window.exportSteps[i],
        // A beat can be a bare hop or an array of them; only the object form carries a `say`.
        says: s.flow.flatMap((b) => (typeof b === 'object' && !Array.isArray(b) && 'say' in b && typeof b.say === 'string' ? [b.say] : [])),
      })),
    };
  }, [figure]);

  const c = dark ? DARK.bg : '#fff';
  return (
    <div style={{ display: 'inline-block', padding: 20, background: c }}>
      {/* A clip keeps the step label and the narration, but not the buttons a reader would click. */}
      <style>
        {
          '.interfig button[aria-label="Full screen"], .interfig button[title="Playback speed"], .interfig button[aria-label="Pause"], .interfig button[aria-label="Play"] { display: none !important }'
        }
      </style>
      <style>{BOLD_CSS}</style>
      {/* `key` restarts the figure on the wanted step: Flow plays the active step from its first beat. */}
      {step != null && (
        <Flow
          key={step}
          {...figure.props}
          steps={(figure.props.steps ?? []).slice(step, step + 1)}
          theme={dark ? DARK : { ...figure.props.theme, ...BOLD }}
        />
      )}
    </div>
  );
}

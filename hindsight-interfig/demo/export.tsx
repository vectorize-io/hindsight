import { useEffect, useState } from 'react';
import { Flow, type Figure } from '../src';

/** The page `scripts/export.mjs` records: one figure, alone on the page, one step at a time.
 *  It stays blank until the recorder calls `startExport`, so no frames are wasted before the step begins. */
const DARK = { accent: '#3396e8', fg: '#e3e3e3', muted: '#9aa0a6', bg: '#1b1b1d', surface: '#242526', border: '#3a3b3c' };

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
      {/* `key` restarts the figure on the wanted step: Flow plays the active step from its first beat. */}
      {step != null && (
        <Flow key={step} {...figure.props} steps={(figure.props.steps ?? []).slice(step, step + 1)} theme={dark ? DARK : undefined} />
      )}
    </div>
  );
}

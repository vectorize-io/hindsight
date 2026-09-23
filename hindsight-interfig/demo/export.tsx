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
  }
}

export function ExportPage({ figure, dark }: { figure: Figure; dark: boolean }) {
  const [step, setStep] = useState<number | null>(null);

  useEffect(() => {
    window.exportSteps = (figure.props.steps ?? []).map((s, i) => (typeof s.label === 'string' ? s.label : `step-${i + 1}`));
    window.startExport = setStep;
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

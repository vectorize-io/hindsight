#!/usr/bin/env node
/**
 * Record each figure's steps as video clips, one clip per step — for social posts and release notes.
 *
 *   npm run export                       # every figure, every step, MP4
 *   npm run export -- what-hindsight-does recall   # only these figures
 *   npm run export -- --gif --dark       # also write GIFs, and record the dark theme
 *   npm run export -- --2x               # play at 2x: half as long, same frames
 *   npm run export -- --square           # pad to a square, for feeds that crop to 1:1
 *
 * Clips land in ~/Downloads/interfig-clips as <figure>-<step>.mp4 (e.g. what-hindsight-does-retain.mp4),
 * each beside a <figure>-<step>.md holding the narration the clip speaks, as bullets to paste into a post;
 * pass --out <dir> for somewhere else. They stay out of the repo: they are throwaway social assets.
 *
 * How: vite serves demo/export.tsx (the figure alone on the page), Playwright records the tab
 * while one step plays, and ffmpeg turns the raw webm into an MP4 (H.264, the format the social
 * platforms want). The page stays blank until the recording is running, so the clip starts on the
 * step's first frame.
 *
 * Needs: `npx playwright install chromium` once, and ffmpeg on PATH.
 */
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { execFile } from 'node:child_process';
import { homedir } from 'node:os';
import { mkdir, readdir, rm, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const run = promisify(execFile);
const root = dirname(dirname(fileURLToPath(import.meta.url)));
const args = process.argv.slice(2);
const outAt = args.indexOf('--out');
const OUT = outAt === -1 ? join(homedir(), 'Downloads', 'interfig-clips') : args[outAt + 1];
const RAW = join(OUT, '.raw');
const rest = outAt === -1 ? args : [...args.slice(0, outAt), ...args.slice(outAt + 2)];
const flags = new Set(rest.filter((a) => a.startsWith('--')));
const only = rest.filter((a) => !a.startsWith('--'));
const dark = flags.has('--dark');
const slug = (s) =>
  s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');

/** Play one step and wait for it to finish: the active tab's progress line fills up. */
const playStep = (page, step, fast) =>
  page.evaluate(
    async ({ i, fast }) => {
      window.startExport(i);
      if (fast) {
        // A clip hides the controls, so press the 2x button through the DOM instead of clicking it.
        // It only exists once the figure has rendered, and the press has to land: check it took.
        for (let tries = 0; tries < 50; tries++) {
          const button = document.querySelector('button[title="Playback speed"]');
          if (button?.innerText.startsWith('2')) break;
          button?.click();
          await new Promise((r) => setTimeout(r, 20));
        }
        if (!document.querySelector('button[title="Playback speed"]')?.innerText.startsWith('2')) throw new Error('2x never took');
      }
      const width = () => {
        const bar = document.querySelector('[role=tab][aria-selected=true] div');
        return bar ? parseFloat(bar.style.transform.slice(7)) || 0 : 0;
      };
      // The step is over when the line is full — or when it has dropped back, which means the
      // figure already looped: at 2x the full moment can fall between two polls.
      let prev = 0;
      for (let waited = 0; waited < 120_000; waited += 50) {
        const now = width();
        if (now > 0.995 || (prev > 0.5 && now < prev - 0.2)) return;
        prev = now;
        await new Promise((r) => setTimeout(r, 50));
      }
      throw new Error(`step ${i} never finished`);
    },
    { i: step, fast },
  );

const server = await createServer({ root, server: { port: 0 }, logLevel: 'warn' });
await server.listen();
const base = server.resolvedUrls.local[0].replace(/\/$/, '');
const browser = await chromium.launch();
await mkdir(OUT, { recursive: true });
await rm(RAW, { recursive: true, force: true });

try {
  const figures = (await readdir(join(root, 'figures'))).filter((f) => f.endsWith('.ts')).map((f) => f.replace(/\.ts$/, ''));
  for (const figure of only.length ? only : figures) {
    const url = `${base}/?export${dark ? '&dark' : ''}#${figure}`;

    // First pass: the step names, and how big the figure is, so each clip is exactly the figure.
    const probe = await browser.newPage({ viewport: { width: 1800, height: 1200 } });
    await probe.goto(url);
    const steps = await probe.evaluate(() => window.exportSteps);
    const narration = await probe.evaluate(() => window.exportNarration);
    await probe.evaluate(() => window.startExport(0));
    await probe.waitForSelector('.interfig');
    const box = await probe.evaluate(() => {
      const el = document.querySelector('#root > div'); // the padded box around the figure
      const r = el.getBoundingClientRect();
      // H.264 wants even dimensions.
      return { width: 2 * Math.ceil(r.width / 2), height: 2 * Math.ceil(r.height / 2) };
    });
    await probe.close();

    for (const [i, label] of steps.entries()) {
      const name = `${figure}-${slug(label)}`;
      const context = await browser.newContext({ viewport: box, recordVideo: { dir: RAW, size: box } });
      const startedAt = Date.now(); // recording starts with the context; the page is blank until playStep
      const page = await context.newPage();
      await page.goto(url);
      const blankLead = (Date.now() - startedAt) / 1000;
      await playStep(page, i, flags.has('--2x'));
      const video = page.video();
      await context.close(); // the video file is only complete once the context is gone
      const webm = await video.path();

      const base = `${name}${flags.has('--square') ? '-square' : ''}`; // the clip and its narration share a name
      const mp4 = join(OUT, `${base}.mp4`);
      // A square fits the feeds that crop to 1:1. Pad rather than crop: the figure stays whole,
      // centred on its own background.
      const side = Math.max(box.width, box.height);
      const square = flags.has('--square') ? ['-vf', `pad=${side}:${side}:(ow-iw)/2:(oh-ih)/2:color=${dark ? '0x1b1b1d' : 'white'}`] : [];
      // -ss drops the blank frames before the step started, so the poster frame shows the figure.
      // +faststart moves the index to the front: without it a browser downloads the whole file
      // before it can show anything, which is what GitHub's player looked like it was doing.
      await run('ffmpeg', [
        // prettier-ignore
        '-y',
        '-ss',
        String(blankLead),
        '-i',
        webm,
        '-an',
        '-c:v',
        'libx264',
        '-crf',
        '20',
        '-preset',
        'slow',
        ...square,
        '-pix_fmt',
        'yuv420p',
        '-movflags',
        '+faststart',
        mp4,
      ]);
      if (flags.has('--gif')) {
        const palette = join(RAW, `${name}.png`);
        const filters = 'fps=15,scale=900:-2:flags=lanczos';
        await run('ffmpeg', ['-y', '-ss', String(blankLead), '-i', webm, '-vf', `${filters},palettegen=stats_mode=diff`, palette]);
        await run('ffmpeg', [
          '-y',
          '-ss',
          String(blankLead),
          '-i',
          webm,
          '-i',
          palette,
          '-lavfi',
          `${filters}[x];[x][1:v]paletteuse=dither=bayer`,
          join(OUT, `${name}.gif`),
        ]);
      }
      // The narration beside the clip: a clip is watched, but the words are what gets pasted into a post.
      const { label: heading, says } = narration.steps[i];
      const md = [`# ${narration.title} — ${heading}`, '', ...says.map((s) => `- ${s}`), ''].join('\n');
      await writeFile(join(OUT, `${base}.md`), md);
      console.log(`${join(OUT, base)}.mp4  ${box.width}x${box.height}  (+ .md, ${says.length} lines)`);
    }
  }
} finally {
  await browser.close();
  await server.close();
  await rm(RAW, { recursive: true, force: true });
}

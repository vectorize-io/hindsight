#!/usr/bin/env node
// Headed web-app capture driver — boundary-safe subprocess entrypoint.
//
// oma-video NEVER imports Playwright. The TypeScript provider
// (`providers/capture-playwright.ts` + `internal/playwright-project.ts`) spawns
// *this* script with the node interpreter of a resolved Playwright install
// (its node_modules reachable via cwd / NODE_PATH). The driver drives a real
// headed Chromium, records a live human-driven flow, and muxes the result to a
// single mp4 — then prints ONE authoritative JSON result line on stdout.
//
// MECHANISM ONLY. The driver assumes NOTHING about the flow or its purpose:
//   * any URL (local / staging / prod),
//   * any number of pages (popups / new tabs / cross-origin redirects are
//     recorded generically — never tied to a specific auth or app shape),
//   * NO credential handling or automation of any kind — a human drives the
//     on-screen flow and logs in themselves if the flow needs it.
//
// SECURITY:
//   * the URL + any query tokens are MASKED in every stderr log line,
//   * the recording + all outputs are confined to --out's directory,
//   * credentials are never read, stored, or printed; on-screen sensitive input
//     is captured as-is (the human controls what is on screen).
//
// Contract — flags in, ONE JSON result line out (stdout's LAST line):
//   IN  --url <url>            target (required)
//       --out <abs mp4 path>   output, inside a run dir (required)
//       --size <WxH>           recording frame size (required; derived upstream)
//       --headless <0|1>       headed (0, default, real use) | headless (1, CI)
//       --ready-selector <css> await before the meaningful capture (optional)
//       --show-cursor          overlay a visible cursor (optional)
//       --timeout <ms>         hard ceiling for the whole capture (optional)
//       --stop <mode>          NON-interactive stop for CI/tests:
//                                duration:<sec>  stop after N seconds
//                                selector:<css>  stop when the selector appears
//                              omitted -> interactive ENTER prompt (real path)
//   OUT {"ok":true, "output":"<abs mp4>", "pages":<n>, "durationSec":<float>}
//    |  {"ok":false,"error":"<message>","code":"<reason>"}
//
// Exit code is 0 on success, 1 on failure; the JSON result line is authoritative.
import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import {
  existsSync,
  mkdtempSync,
  readdirSync,
  rmSync,
  statSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

// All mutable state + scratch handle are declared FIRST, so the early-`fail`
// path (and its `cleanup()`) never touches a binding in its temporal dead zone.
let scratch = null;
let browser;
let hardTimer;
let aborted = false;
// Pages recorded, in chronological open order. Each entry: { video, openedAt }.
const recorded = [];
// Stop coordinators — resolved by ENTER / duration / selector / timeout / SIGINT.
const stopWaiters = [];

const args = parseArgs(process.argv.slice(2));
const url = args.url;
const out = args.out;
const size = parseSize(args.size);
const headless = args.headless === "1";
const readySelector = args["ready-selector"];
const showCursor = args["show-cursor"] === true || args["show-cursor"] === "1";
const timeoutMs = Number.parseInt(args.timeout ?? "", 10);
const stopMode = parseStop(args.stop);

if (!url) fail("missing --url", "bad-args");
if (!out) fail("missing --out", "bad-args");
if (!size) fail("missing or invalid --size (expected WxH)", "bad-args");

const outDir = path.dirname(path.resolve(out));
if (!existsSync(outDir)) fail(`--out directory does not exist: ${mask(outDir)}`, "bad-out");

// Resolve Playwright from the install the provider passes via --playwright-dir
// (its node_modules), falling back to a bare resolution from cwd. ESM ignores
// NODE_PATH, so we resolve the package's absolute entry explicitly and import
// it by file URL — boundary-safe (this script is never imported into the CLI).
const chromium = await resolveChromium(args["playwright-dir"]);

// Recordings land in a private temp dir we own, then get muxed into --out.
// Confinement: nothing is written outside outDir (final mp4) or this scratch
// dir (intermediate webms, cleaned up at the end).
scratch = mkdtempSync(path.join(tmpdir(), "oma-video-pwcap-"));

log(`opening ${mask(url)} (${size.width}x${size.height}, ${headless ? "headless" : "headed"})`);

try {
  browser = await chromium.launch({ headless });
  const context = await browser.newContext({
    viewport: { width: size.width, height: size.height },
    recordVideo: { dir: scratch, size: { width: size.width, height: size.height } },
  });

  // Record EVERY page generically — popups, new tabs, cross-origin redirects.
  // We never inspect or interpret the page; we only track open order so the
  // final concat is chronological.
  context.on("page", (page) => {
    recorded.push({ page, openedAt: Date.now() });
    if (showCursor) injectCursor(page).catch(() => undefined);
  });

  const firstPage = await context.newPage();
  if (recorded.length === 0) {
    recorded.push({ page: firstPage, openedAt: Date.now() });
  }
  if (showCursor) await injectCursor(firstPage).catch(() => undefined);

  // Hard ceiling: abort the whole capture cleanly if it runs past --timeout.
  if (Number.isFinite(timeoutMs) && timeoutMs > 0) {
    hardTimer = setTimeout(() => {
      aborted = true;
      log(`capture timeout (${timeoutMs}ms) reached — stopping`);
      stopWaiters.forEach((resolve) => {
        resolve("timeout");
      });
    }, timeoutMs);
    hardTimer.unref?.();
  }

  await firstPage.goto(url, { waitUntil: "load", timeout: 60_000 }).catch((err) => {
    log(`navigation warning: ${stringifyError(err)}`);
  });
  // Hydration: settle the network, then await an optional readiness selector.
  await firstPage
    .waitForLoadState("networkidle", { timeout: 30_000 })
    .catch(() => log("networkidle not reached within 30s — continuing"));
  if (readySelector) {
    await firstPage
      .waitForSelector(readySelector, { timeout: 30_000 })
      .catch(() => log(`ready-selector not found within 30s — continuing`));
  }

  // STOP: interactive ENTER (real human path) OR a non-interactive mode (CI).
  const reason = await waitForStop(firstPage, stopMode);
  log(`stopping (${reason})`);

  // Close the context to flush every page's video to disk.
  await context.close();
  await browser.close();
  browser = undefined;

  // Collect the produced webms in chronological page order. Playwright names
  // them unpredictably, so we sort scratch entries by mtime to approximate the
  // open order (the first page is always first).
  const webms = collectWebms(scratch);
  if (webms.length === 0) {
    fail("no video produced (0-frame recording)", "empty-recording");
  }

  const durationSec = await muxToMp4(webms, path.resolve(out));
  if (durationSec === null || !existsSync(path.resolve(out))) {
    fail("ffmpeg produced no output mp4", "mux-failed");
  }

  emit({
    ok: true,
    output: path.resolve(out),
    pages: webms.length,
    durationSec,
  });
  cleanup();
  process.exit(0);
} catch (err) {
  cleanup();
  // Ctrl-C / SIGINT surfaces here as an abort; report it without partial output.
  emit({
    ok: false,
    code: aborted ? "aborted" : "capture-error",
    error: stringifyError(err),
  });
  if (browser) await browser.close().catch(() => undefined);
  process.exit(1);
}

// ---------------------------------------------------------------------------
// Stop coordination
// ---------------------------------------------------------------------------

/**
 * Resolve when the capture should stop. Interactive: the first ENTER on stdin.
 * Non-interactive: `duration:<sec>` after N seconds, or `selector:<css>` when
 * the selector appears on the first page. The hard --timeout also resolves it.
 */
function waitForStop(firstPage, stop) {
  return new Promise((resolve) => {
    let settled = false;
    const done = (reason) => {
      if (settled) return;
      settled = true;
      resolve(reason);
    };
    stopWaiters.push(done);

    if (stop?.kind === "duration") {
      log(`non-interactive stop: after ${stop.seconds}s`);
      const t = setTimeout(() => done("duration"), stop.seconds * 1000);
      t.unref?.();
      return;
    }
    if (stop?.kind === "selector") {
      log(`non-interactive stop: when selector appears`);
      firstPage
        .waitForSelector(stop.selector, { timeout: 0 })
        .then(() => done("selector"))
        .catch(() => undefined);
      return;
    }

    // Interactive ENTER. The prompt goes to stderr so stdout stays pure JSON.
    process.stderr.write(
      "\n[oma-video] Browser is open. Perform your flow, then press ENTER here to stop recording.\n",
    );
    process.stdin.resume();
    const onData = () => {
      process.stdin.off("data", onData);
      process.stdin.pause();
      done("enter");
    };
    process.stdin.on("data", onData);
    // Ctrl-C: clean stop, no partial output (caller sees a non-ok result).
    process.once("SIGINT", () => {
      aborted = true;
      done("sigint");
    });
  });
}

// ---------------------------------------------------------------------------
// Media helpers
// ---------------------------------------------------------------------------

/** Sorted list of produced webm paths (chronological by mtime). */
function collectWebms(dir) {
  let entries;
  try {
    entries = readdirSync(dir);
  } catch {
    return [];
  }
  return entries
    .filter((name) => name.toLowerCase().endsWith(".webm"))
    .map((name) => path.join(dir, name))
    .sort((a, b) => mtime(a) - mtime(b));
}

function mtime(file) {
  try {
    return statSync(file).mtimeMs;
  } catch {
    return 0;
  }
}

/**
 * Mux the recorded webms into a single mp4 at `outPath`. One page -> a straight
 * transcode; multiple pages -> a chronological concat (filtergraph, so differing
 * codecs/sizes are normalized to the first page's frame). Returns the muxed
 * duration (seconds) or null on failure.
 */
async function muxToMp4(webms, outPath) {
  if (webms.length === 1) {
    const code = await runFfmpeg([
      "-y",
      "-i",
      webms[0],
      "-c:v",
      "libx264",
      "-pix_fmt",
      "yuv420p",
      "-movflags",
      "+faststart",
      outPath,
    ]);
    if (code !== 0) return null;
    return probeDuration(outPath);
  }

  // Multi-page: concat via the concat filter so heterogeneous inputs are safe.
  const inputs = [];
  for (const webm of webms) {
    inputs.push("-i", webm);
  }
  const labels = webms.map((_, i) => `[${i}:v:0]`).join("");
  const filter = `${labels}concat=n=${webms.length}:v=1:a=0[outv]`;
  const code = await runFfmpeg([
    "-y",
    ...inputs,
    "-filter_complex",
    filter,
    "-map",
    "[outv]",
    "-c:v",
    "libx264",
    "-pix_fmt",
    "yuv420p",
    "-movflags",
    "+faststart",
    outPath,
  ]);
  if (code !== 0) return null;
  return probeDuration(outPath);
}

function runFfmpeg(ffArgs) {
  return new Promise((resolve) => {
    const ff = spawn(ffmpegBin(), ffArgs, { stdio: ["ignore", "ignore", "pipe"] });
    let stderr = "";
    ff.stderr?.on("data", (c) => {
      stderr += c.toString();
    });
    ff.on("error", () => resolve(1));
    ff.on("close", (code) => {
      if (code !== 0) log(`ffmpeg exit ${code}: ${stderr.trim().split("\n").slice(-1)[0] ?? ""}`);
      resolve(code ?? 1);
    });
  });
}

function probeDuration(file) {
  return new Promise((resolve) => {
    const ff = spawn(
      ffprobeBin(),
      [
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        file,
      ],
      { stdio: ["ignore", "pipe", "ignore"] },
    );
    let stdout = "";
    ff.stdout?.on("data", (c) => {
      stdout += c.toString();
    });
    ff.on("error", () => resolve(null));
    ff.on("close", () => {
      const seconds = Number.parseFloat(stdout.trim());
      resolve(Number.isFinite(seconds) && seconds > 0 ? seconds : null);
    });
  });
}

function ffmpegBin() {
  return process.env.OMA_FFMPEG?.trim() || "ffmpeg";
}
function ffprobeBin() {
  return process.env.OMA_FFPROBE?.trim() || "ffprobe";
}

/** Inject a small visible-cursor overlay that follows the mouse, for clarity. */
async function injectCursor(page) {
  await page.addInitScript(() => {
    const dot = document.createElement("div");
    dot.style.cssText =
      "position:fixed;z-index:2147483647;width:18px;height:18px;margin:-9px 0 0 -9px;border-radius:50%;background:rgba(255,80,80,.65);box-shadow:0 0 0 2px rgba(255,255,255,.9);pointer-events:none;transition:transform .03s linear;";
    const mount = () => {
      if (document.body) document.body.appendChild(dot);
    };
    if (document.body) mount();
    else document.addEventListener("DOMContentLoaded", mount);
    document.addEventListener("mousemove", (e) => {
      dot.style.left = `${e.clientX}px`;
      dot.style.top = `${e.clientY}px`;
    });
  });
}

// ---------------------------------------------------------------------------
// Arg parsing + masking + result emission
// ---------------------------------------------------------------------------

/**
 * Resolve `chromium` from a Playwright install. `dir` is the directory whose
 * `node_modules` holds `playwright` (or `@playwright/test`); when omitted we try
 * a bare resolution from the current working directory. Imports the package's
 * absolute entry by file URL because ESM does not honor NODE_PATH. Emits a
 * masked `playwright-unresolved` result and exits 1 on failure.
 */
async function resolveChromium(dir) {
  const anchor = dir
    ? path.join(path.resolve(dir), "node_modules", "__anchor__.js")
    : path.join(process.cwd(), "__anchor__.js");
  const require = createRequire(anchor);
  for (const pkg of ["playwright", "@playwright/test"]) {
    // Playwright is CommonJS; `require` yields the real module.exports whose
    // named `chromium` is the launcher. (An ESM `import` of the CJS entry only
    // interops `default`, dropping the named exports — so we require here.)
    try {
      const mod = require(pkg);
      if (mod?.chromium) return mod.chromium;
    } catch {
      // try the next package specifier
    }
    // Fallback: resolve the absolute entry and import by file URL (covers
    // installs whose package "exports" map only exposes an ESM entry).
    try {
      const entry = require.resolve(pkg);
      const mod = await import(pathToFileURL(entry).href);
      const chromium = mod?.chromium ?? mod?.default?.chromium;
      if (chromium) return chromium;
    } catch {
      // try the next package specifier
    }
  }
  emit({
    ok: false,
    code: "playwright-unresolved",
    error: `could not load Playwright from ${mask(dir ?? process.cwd())}`,
  });
  cleanup();
  process.exit(1);
  return undefined; // unreachable; keeps the type obvious
}

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (!token.startsWith("--")) continue;
    const key = token.slice(2);
    const next = argv[i + 1];
    if (next === undefined || next.startsWith("--")) {
      out[key] = true; // boolean flag
    } else {
      out[key] = next;
      i++;
    }
  }
  return out;
}

function parseSize(value) {
  if (!value || typeof value !== "string") return null;
  const m = /^(\d+)x(\d+)$/.exec(value.trim());
  if (!m) return null;
  const width = Number.parseInt(m[1], 10);
  const height = Number.parseInt(m[2], 10);
  if (!(width > 0 && height > 0)) return null;
  return { width, height };
}

function parseStop(value) {
  if (!value || typeof value !== "string") return null;
  const [kind, ...rest] = value.split(":");
  const arg = rest.join(":");
  if (kind === "duration") {
    const seconds = Number.parseFloat(arg);
    if (Number.isFinite(seconds) && seconds > 0) return { kind: "duration", seconds };
    return null;
  }
  if (kind === "selector" && arg.length > 0) {
    return { kind: "selector", selector: arg };
  }
  return null;
}

/**
 * Mask a URL for logging: keep scheme + host + path shape, strip the query/hash
 * entirely (it may carry tokens), and redact userinfo. Non-URL strings get a
 * coarse redaction of anything that looks like a token/query.
 */
function mask(value) {
  if (typeof value !== "string") return String(value);
  try {
    const u = new URL(value);
    const auth = u.username ? "***@" : "";
    const query = u.search ? "?<redacted>" : "";
    const hash = u.hash ? "#<redacted>" : "";
    return `${u.protocol}//${auth}${u.host}${u.pathname}${query}${hash}`;
  } catch {
    // Not a URL — redact any "?...="/"&...=" token-ish tails and long hex/jwt.
    return value
      .replace(/([?&][^=\s]+=)[^&\s]+/g, "$1<redacted>")
      .replace(/\b[A-Za-z0-9_-]{24,}\b/g, "<redacted>");
  }
}

function log(message) {
  process.stderr.write(`[oma-video] ${mask(message)}\n`);
}

function stringifyError(err) {
  const msg = err instanceof Error ? err.message : String(err);
  return mask(msg);
}

function emit(result) {
  process.stdout.write(`${JSON.stringify(result)}\n`);
}

function fail(message, code) {
  cleanup();
  emit({ ok: false, code: code ?? "error", error: mask(message) });
  process.exit(1);
}

function cleanup() {
  if (hardTimer) clearTimeout(hardTimer);
  if (!scratch) return;
  try {
    rmSync(scratch, { recursive: true, force: true });
  } catch {
    // best-effort scratch cleanup
  }
}

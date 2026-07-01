// load-fonts.ts — embed Pretendard for cross-machine identical renders.
//
// Pretendard is a CJK-ready variable font (design rule 2: CJK services
// prioritize Pretendard Variable). Embedding it locally — rather than relying
// on a system font or a network fetch — is what makes a render byte-stable
// across machines (design 013 §5). The .woff2 is NOT committed to git here to
// keep the skill tree light; `oma video doctor` fetches it once into
// `public/fonts/` (see public/fonts/README.md), then this loader picks it up.
//
// loadFont() blocks the render until the font is ready, so captions/on-screen
// text never flash a fallback face mid-render.
import { loadFont } from "@remotion/fonts";
import { staticFile } from "remotion";

export const PRETENDARD_FAMILY = "Pretendard";

// staticFile() resolves against public/. doctor writes the woff2 here.
const PRETENDARD_URL = staticFile("fonts/PretendardVariable.woff2");

let fontPromise: Promise<void> | null = null;

/**
 * Idempotently load Pretendard. Compositions call this at module scope so the
 * render is delayed until the font is ready. If the embedded woff2 is missing
 * (doctor not run), the browser falls back to the system stack — the render
 * still succeeds, but is not guaranteed byte-identical across machines.
 *
 * `@remotion/fonts` `loadFont()` calls Remotion's `cancelRender()` internally
 * when the font URL fails to load (e.g. a 404 because `oma video doctor` has not
 * fetched the woff2 yet). `cancelRender()` aborts the WHOLE render, so a plain
 * `.catch()` on the returned promise is not enough — the render is already
 * cancelled. We therefore probe the URL with `fetch` first and only call
 * `loadFont()` when the asset is actually present. A missing font then degrades
 * gracefully to the system `FONT_STACK` instead of hard-failing the render.
 */
export function ensurePretendard(): Promise<void> {
  if (!fontPromise) {
    fontPromise = (async () => {
      try {
        const probe = await fetch(PRETENDARD_URL, { method: "HEAD" });
        if (!probe.ok) return;
      } catch {
        // Network/probe error -> system fallback.
        return;
      }
      await loadFont({
        family: PRETENDARD_FAMILY,
        url: PRETENDARD_URL,
        format: "woff2",
        weight: "100 900",
        display: "block",
      }).catch(() => undefined);
    })();
  }
  return fontPromise;
}

// System fallback stack (design rule 1 + 2): CJK-ready first, then system-ui.
export const FONT_STACK =
  `"${PRETENDARD_FAMILY}", "Noto Sans CJK KR", system-ui, ` +
  `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;

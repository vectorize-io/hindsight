# oma-video Remotion compositor (vendored)

The universal compositor for `oma-video`. It registers three compositions —
**Shorts** (9:16), **Explainer** (16:9 / 9:16), and **Demo** (16:9) — that each
consume `render-spec.json` as input props and emit an `.mp4`.

This project is **lockfile-pinned and installed once via `oma video doctor`** —
never installed during a render (design 013 §5, §7 Tier-1).

## Layout

```
resources/remotion/
├── package.json            # pinned remotion + @remotion/cli + @remotion/captions + @remotion/fonts
├── tsconfig.json           # strict, noEmit (typecheck-shaped)
├── remotion.config.ts      # deterministic encoder defaults (h264 mp4)
├── public/fonts/           # Pretendard embed (provisioned by doctor; see README)
└── src/
    ├── index.ts            # registerRoot(RemotionRoot)
    ├── Root.tsx            # <Composition> Shorts / Explainer / Demo + calculateMetadata
    ├── render-spec.ts      # Zod mirror of the CLI RenderSpec (schemaVersion "1.0")
    ├── load-fonts.ts       # Pretendard via @remotion/fonts loadFont()
    ├── compositions/       # Shorts.tsx · Explainer.tsx · Demo.tsx (thin wrappers)
    └── components/
        ├── VideoBase.tsx   # shared timeline: scenes + audio + captions
        ├── Scene.tsx       # one render-spec scene (image/video/slide/capture) + Ken Burns
        └── Captions.tsx    # @remotion/captions: parseSrt -> createTikTokStyleCaptions
```

## render-spec.json as input props

`render-spec.json` is the **deterministic compute boundary** (design 013 §4.1).
The CLI writes it into the run dir; the compositions read it via `--props`.
`Root.tsx` validates the props with `RenderSpecSchema` (Zod) and derives the real
`width` / `height` / `fps` / `durationInFrames` from the spec via
`calculateMetadata`. An invalid render-spec fails fast (maps to the CLI's
`SchemaValidationError`, exit 4) instead of rendering garbage.

The Zod schema here mirrors `cli/commands/video/types.ts` `RenderSpecSchema`.
**The CLI is the source of truth** — keep the two in sync.

## Rendering

Install once (via doctor), then render a run:

```bash
# Shorts (9:16). <entry> = src/index.ts, <CompId> = Shorts
npx remotion render src/index.ts Shorts out.mp4 --props=render-spec.json

# Explainer (16:9)
npx remotion render src/index.ts Explainer out.mp4 --props=render-spec.json

# Demo (16:9)
npx remotion render src/index.ts Demo out.mp4 --props=render-spec.json
```

Where `render-spec.json` is the run dir's spec (the CLI passes an absolute path).
Preview interactively with `npm run studio`.

## Live render is deferred (key-optional, backend rule 11)

Actual rendering needs **Node + Chromium + FFmpeg**, bootstrapped by
`oma video doctor`. The CLI adapter
(`cli/commands/video/providers/compositor.ts`) gates the real invocation on
FFmpeg availability and, until the toolchain is wired, falls back to a
deterministic placeholder mp4 derived from the render-spec so the run dir +
manifest are always well-formed:

```
// real     : when FFmpeg + this vendored project are present, invoke
//            `npx remotion render src/index.ts <CompId> out.mp4 --props=render-spec.json`
// fallback : deterministic placeholder mp4 from the render-spec (zero toolchain)
//
// TODO(oma-deferred): remotion render (F3) — wire the CLI adapter to spawn the
//   `npx remotion render` subprocess against this project once doctor guarantees
//   Node/Chromium/FFmpeg. The placeholder branch stays as the key-free fallback.
```

The fallback is itself a pure function of the render-spec, so it is reproducible
from the same spec.

## Determinism

- `render-spec.json` + asset files + `seed` + the embedded **Pretendard** font
  are the determinism boundary. The same inputs render byte-stable across
  machines.
- Ken Burns and all motion are driven purely by the frame (no randomness).
- `loadFont(..., { display: "block" })` blocks the render until Pretendard is
  ready, so text never flashes a fallback face mid-render.

## Dependency notes (lockfile)

- `remotion`, `@remotion/cli`, `@remotion/captions`, `@remotion/fonts` are pinned
  to the **same exact version** (Remotion requires lockstep versions across its
  packages). Bump them together.
- Generate and commit a lockfile (`package-lock.json` / `bun.lock`) when this
  project is first installed so doctor installs the exact pinned tree. Do not run
  `install` during a render.

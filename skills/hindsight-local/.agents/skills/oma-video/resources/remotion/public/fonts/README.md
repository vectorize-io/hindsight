# Embedded fonts (Pretendard)

`src/load-fonts.ts` embeds **Pretendard Variable** via `@remotion/fonts`
`loadFont()`. Embedding the font locally — instead of relying on a system font
or a network fetch — is what makes a render **byte-identical across machines**
(design 013 §5; design rule 2: CJK-ready font priority).

## What goes here

```
public/fonts/PretendardVariable.woff2
```

`staticFile("fonts/PretendardVariable.woff2")` resolves to this path at render
time. The `.woff2` is **not committed** to keep the skill tree light.

## How it is provisioned

`oma video doctor` fetches the font once into this directory (lockfile-pinned,
no fetch during a render). Until then, `ensurePretendard()` swallows the missing
file and the browser falls back to the system stack defined in `FONT_STACK`
(`system-ui, -apple-system, …`). The render still succeeds, but is not
guaranteed byte-identical across machines.

Source: Pretendard (OFL-1.1) — https://github.com/orioncactus/pretendard
Mirror the exact release pinned by `oma video doctor` so renders stay
reproducible.

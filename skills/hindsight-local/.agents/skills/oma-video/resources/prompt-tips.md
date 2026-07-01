# Prompt Tips

A good video brief specifies the **mode**, the **topic/source**, and the
**arc** (what the viewer should feel/learn). The brief drives the script; the
script's per-scene `visual.prompt` then drives oma-image (English prompts —
image models are trained predominantly on English captions).

## Brief structure

```
Mode + audience → Topic/source → Arc (hook → body → payoff) → Tone/pacing
```

Example (shorts): `30s vertical short for a dev audience: how oma-video turns a topic into a finished clip; hook with the pain, show the 3-step flow, end on the one-command CTA; upbeat, fast cuts`

## Per-mode guidance

### shorts (9:16)
- **Hook in the first 1.5s** — the first scene's on-screen text must earn the swipe.
- Keep scenes 2–4s; favor 6–10 scenes for a 30s clip (≤ `max_scenes` 40).
- oma-image stills get Ken Burns motion — write prompts that frame a clear subject with headroom for the pan.
- TikTok captions are centered and animated; keep narration lines short (≤ ~8 words/segment) so caption pages switch cleanly.

### explainer (16:9 / 9:16)
- Source is a README / code / data — point the brief at the file and the *one* thing to teach.
- Mix oma-slide frames (structure, bullet beats) with oma-image diagrams (concepts) and code frames.
- Code frames use one fixed deterministic theme (v1). Keep snippets short and legible at 1920×1080.
- Narration should explain *why*, not read the code line-by-line.

### demo (16:9)
- The visual is a human-recorded capture (`--capture`). The brief drives the intro card, zoom/callout beats, and outro.
- Call out *where to look* on screen — Remotion adds zoom + callouts over the capture.
- Keep the intro ≤ 3s; viewers came for the product, not the title card.

## Per-scene image prompts (oma-image)

The script's `scenes[].visual.prompt` is forwarded to oma-image. Write them with
the same structure oma-image expects:

```
Scene/backdrop → Subject → Details → Constraints
```

| Mode | Example per-scene prompt |
|------|--------------------------|
| shorts | `Aerial drone shot of Jeju coastline, turquoise water meeting volcanic rock, golden hour, vertical composition with headroom for a slow zoom` |
| shorts | `Close-up of a hand pouring espresso into a glass over ice, warm cafe light, shallow depth of field` |
| explainer | `Clean isometric diagram of a 3-stage pipeline (script → assets → render), flat vector, labeled boxes, neutral background` |
| demo | `Minimal title card: product name centered, dark UI background, subtle accent gradient, 16:9` |

## Do's

- Anchor the **arc**: hook → body → payoff. A short without a hook gets swiped past.
- Match **aspect to mode** (9:16 shorts, 16:9 explainer/demo) or use `auto`.
- Keep narration **per-scene and short** so caption pages and scene boundaries align.
- Pick **music** that matches pacing (`upbeat` for shorts, `calm` for explainer).

## Don'ts

- Don't write a brief with no mode and no topic — the agent will have to clarify.
- Don't request paid stock/AIGC visuals without the key — the chain silently falls back to oma-image stills (annotated in `warnings`), which may surprise you.
- Don't exceed `max_duration_sec` (180) or `max_scenes` (40).
- Don't put secrets in the brief unless `--no-brief-in-manifest` is set.

## Localization

- Narration + on-screen text are authored in `--locale`. For a non-source locale, oma-translator translates the script text before TTS/captions (key-free). If the translator is absent, the source text is kept and a warning is recorded.
- oma-image prompts are sent in **English** regardless of `--locale`; translate the user's subject and show the translated prompt during amplification.

## Determinism

- `--seed <n>` makes the render reproducible. The same `render-spec.json` + assets + seed + embedded Pretendard font produce a byte-stable render.
- Re-render an existing run with `oma video render <runDir>` — it consumes `render-spec.json` only, so script/voice/visual generation is not re-run.
- `OMA_VIDEO_MOCK=1` replays golden fixtures for deterministic tests.

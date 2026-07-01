# Vendor Matrix

oma-video is a **key-optional** router (backend rule 11). Each capability has a
provider `order` (a fallback chain). The orchestrator probes availability and
walks the chain: the first available provider wins; only chain exhaustion is a
stage failure. Paid providers auto-enable **only** when their env key is set;
otherwise the chain falls through to a key-free default.

## Capabilities -> providers

| Capability | order (config) | real (key/resource) | key-free fallback | marker |
|------------|----------------|---------------------|-------------------|--------|
| script | `[agent-script]` | agent-authored script (agent-as-key) | — | — |
| voice | `[oma-voice]` | Voicebox MCP TTS + STT timing | estimated timing (no wav) | — |
| visual | `[oma-image, pexels, pixelle]` | Pexels stock · Pixelle AIGC | oma-image stills + Ken Burns | `TODO(oma-deferred): pexels` / `pixelle` |
| caption | `[oma-captions]` | oma-translator for non-source locale | source-locale text from timing | `TODO(oma-deferred): oma-translator` |
| capture | `[cap]` | Cap CLI trigger | guided protocol + `--capture <path>` | `TODO(oma-deferred): cap` |
| compositor | `[remotion, mpt]` | Remotion render · MPT custom-script | deterministic placeholder mp4 | `TODO(oma-deferred): remotion render` |

## Tier model

| Tier | Surface | Providers | Notes |
|:---:|---------|-----------|-------|
| 1 | CLI-first (subprocess) | Remotion, MPT, oma-image, oma-slide, oma-voice (REST) | deterministic; preferred whenever a CLI can drive the work |
| 2 | MCP | Voicebox MCP, Pixelle-MCP | localhost MCP; Pixelle off by default, community-MCP consent + key |
| 3 | Guided (human) | Cap, openscreen | `demo` capture is performed by a human |

## oma-voice (VoiceProvider + timing)

| Field | Value |
|-------|-------|
| Surface | Voicebox MCP at `127.0.0.1:17493` |
| Synthesize | `voicebox_speak{text, profile, language}` -> `generation_id` |
| Retrieve wav | REST `GET /audio/{generation_id}` (MCP has no save-to-disk) |
| Timing (real) | `voicebox_transcribe{audio_path}` on the wav -> `source: voicebox-stt` |
| Timing (fallback) | whisper.cpp -> `estimated` (no wav written, `audio` field empty) |
| Side effect | Narration plays on the speakers during synthesis |
| Health | exit 5 if MCP down; exit 3 if the named profile is missing |

## oma-image (VisualProvider: generate) — default key-free visual

| Field | Value |
|-------|-------|
| Transport | `oma image generate "<prompt>" --vendor auto --size <16-multiple> --format json --out <runDir>/visuals` |
| Aspect -> size | snapped to nearest 16-multiple: 9:16 -> 1088×1920, 16:9 -> 1920×1088, 1:1 -> 1088×1088 |
| Crop | Remotion crops the still to the exact frame; Ken Burns adds motion |
| Cost | free defaults (pollinations / antigravity); codex per-image per oma-image config |

## oma-slide (VisualProvider: slide, explainer) — key-free

| Field | Value |
|-------|-------|
| Transport | `oma slide` generate deck -> `oma slide export --format png` -> 1920×1080 frames |
| Layering | oma-slide internally calls oma-image (same key-free chain) |
| Use | explainer code/diagram frames |

## Pexels (VisualProvider: stock) — paid, opt-in

| Field | Value |
|-------|-------|
| Auth | `PEXELS_API_KEY` env var |
| Enabled | only when the key is present (`providers.pexels.enabled` gate) |
| Fallback | absent key -> skip; chain falls to oma-image stills + Ken Burns |
| Marker | `TODO(oma-deferred): pexels` on the real-call branch until the key is provisioned |

## Pixelle-MCP + RunningHub (VisualProvider: aigc) — paid, off by default

| Field | Value |
|-------|-------|
| Surface | MCP `http://localhost:9004/pixelle/mcp`; RunningHub cloud or local ComfyUI |
| Setup | `uvx pixelle@latest` + wizard; one-time **explicit consent + source review** |
| Auth | `RUNNINGHUB_API_KEY` env var; cost gates on RunningHub credits via `--max-usd` |
| Enabled | off by default (`providers.pixelle.enabled`); never auto-connects |
| Fallback | absent/declined -> oma-image stills |
| Marker | `TODO(oma-deferred): pixelle` on the real-call branch |

## Cap (CaptureProvider, demo) — Tier 3 guided

| Field | Value |
|-------|-------|
| Real | Cap CLI trigger when installed |
| Fallback | guided protocol: instruct the human to record, then pass `--capture <path>` |
| Path safety | `--capture` is absolutized, `$PWD`-guarded, existence + format validated |
| Marker | `TODO(oma-deferred): cap` on the CLI-trigger branch |

## Compositor: Remotion (default) / MPT (alt)

| Field | Value |
|-------|-------|
| Real | vendored `resources/remotion/` -> `npx remotion render <entry> <CompId> out.mp4 --props=render-spec.json` |
| Requires | Node + Chromium + FFmpeg (bootstrapped once via `oma video doctor`) |
| Fallback | deterministic placeholder mp4 derived from the render-spec (well-formed run dir + manifest, zero toolchain) |
| Determinism | render-spec + assets + seed + embedded Pretendard; re-render is byte-stable |
| Marker | `TODO(oma-deferred): remotion render` on the live-render branch (CLI adapter) |
| MPT alt | inject the agent-written script (custom-script mode); keys env-only + log masking; `--compositor mpt` |

## Error Classification

| Error kind | Retry policy | Exit code when solo |
|------------|-------------|---------------------|
| `provider-unavailable` | try next provider in `order`; chain-exhaustion fails | 5 |
| `auth-required` | fail; hint tells the user how to authenticate | 5 |
| `compositor-bootstrap` | fail; point to `oma video doctor` (+ MPT fallback) | 1 |
| `cost-guardrail` | confirm; decline -> stop | 1 |
| `capture-required` | guided protocol; not a hard error | (guided) |
| `schema-validation` | fail; identify the offending field | 4 |
| `safety-refused` | short-circuit (no fallback) | 2 |
| `not-found` | fail; missing profile / asset | 3 |
| `timeout` | record; fail | 6 |

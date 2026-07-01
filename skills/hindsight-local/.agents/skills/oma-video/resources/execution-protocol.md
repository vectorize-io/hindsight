# Video Agent - Execution Protocol

## Step -1: Clarify / Infer Mode (agent-side, before `oma video generate`)

Run the **Clarification Protocol** in `SKILL.md` before shelling out. Infer the
mode from keywords (shorts/reels/쇼츠/릴스 -> shorts; README/code/data/explain/설명
-> explainer; demo/walkthrough/capture/데모 -> demo) and show the user the inferred
plan when the brief is a one-liner.

## Step 0: Parse Request

1. Extract the brief and flags from the invocation.
2. Resolve defaults from `config/video-config.yaml` -> env vars -> CLI flags (lowest to highest precedence).
3. Validate:
   - `mode` ∈ {`shorts`, `explainer`, `demo`}.
   - `aspect` ∈ {`9:16`, `16:9`, `1:1`, `auto`} (`auto` snaps to the mode default: shorts -> 9:16, explainer/demo -> 16:9).
   - `captions` ∈ {`tiktok`, `lower-third`, `none`}; `visual` ∈ {`auto`, `generate`, `stock`, `aigc`, `slide`}.
   - `music` ∈ {`upbeat`, `calm`, `none`}; `compositor` ∈ {`remotion`, `mpt`}.
   - `duration` ≤ `limits.max_duration_sec` (180); resulting `scenes` ≤ `limits.max_scenes` (40).
   - `out` is inside `$PWD` unless `--allow-external-out`.
   - For `demo`: `--capture` (if given) exists, is absolute + `$PWD`-guarded, and is a valid video format.
4. If invalid: exit code 4 with a message identifying the offending field.

## Step 0.5: Mode Routing

- **shorts** (9:16): synthetic from the topic. Default visual = oma-image stills + Ken Burns.
- **explainer** (16:9 / 9:16): from README / code / data. Default visual = oma-slide frames + oma-image diagrams.
- **demo** (16:9): from a screen capture. If no `--capture` and Cap is unavailable -> emit the **guided capture protocol** (Step 4b) and stop.

## Step 1: Provider Availability + Selection

1. Call `available()` on every registered provider in parallel (`oma video list-providers`).
2. For each capability, walk `providers.<capability>.order`:
   - The first available provider wins.
   - Paid providers (`pexels`, `pixelle`) are skipped unless their env key is present (`enabled` gate).
   - On chain exhaustion for a required capability -> exit 5 naming the capability.
3. Log `using: <provider>` per capability to stderr before generation.

## Step 2: Cost Guardrail

1. Estimate cost = sum of each selected provider's `estimateCost()` (most are `$0`; Pixelle/RunningHub credits are non-zero).
2. If `--dry-run`: emit `script.json` / `render-spec.json` / `manifest.json`, skip rendering, exit 0.
3. If estimate ≥ `cost.guardrail_usd` (or `--max-usd`) and not `--yes` / `OMA_VIDEO_YES=1`:
   - Prompt on stderr: `Estimated cost $X.XX. Proceed? (y/N)`. Decline -> exit 1.

## Step 3: Cancellation Setup

1. Install `SIGINT` / `SIGTERM` handlers that call `AbortController.abort()`.
2. Thread the signal into every provider call and into the render subprocess.

## Step 4: Pipeline (asset bus)

```
brief ─► [ScriptProvider] ─► script.json {scenes[], narration[], onScreenText[]}
script.json ─► parallel:
   ├► [VoiceProvider]   ─► audio/narration-*.wav + timing.json
   ├► [VisualProvider]  ─► visuals/scene-NN.*
   └► [CaptionProvider] ─► captions.srt / captions.vtt
all assets ─► render-spec.json ─► [Compositor: Remotion] ─► <mode>-<slug>.mp4
```

1. **Script**: AgentScriptProvider writes `script.json` (start of the determinism boundary).
2. **Voice**: oma-voice synthesizes narration -> `audio/narration-NN.wav` + `timing.json`. Fallback: estimated timing (no wav).
3. **Visuals**: walk the visual chain. oma-image stills (key-free default) / oma-slide frames (explainer) / Pexels (key) / Pixelle (key). Aspect -> 16-multiple size; Remotion crops to the exact frame.
4. **Captions**: oma-captions builds `captions.srt` + `captions.vtt` from `timing.json`. For a non-source locale, translate via oma-translator (key-free); absent -> warn + keep source.
5. **render-spec**: compose `render-spec.json` (the deterministic compute boundary) from the assets + seed.
6. **Render**: the compositor consumes `render-spec.json`.

### Step 4b: Guided Capture Protocol (demo, no capture)

State plainly to the user: **"Demo capture is performed by a human."** Then:
1. Instruct: record the walkthrough with Cap (or any recorder) at 16:9.
2. Ask the user to re-run with `--capture <absolute-path>.mp4`.
3. Stop without rendering (capture-required is a guided stop, not a hard error).

## Step 5: Compositor Render

- **Remotion (default)**: when the vendored toolchain (Node + Chromium + FFmpeg) is bootstrapped, run `npx remotion render <entry> <CompId> <mode>-<slug>.mp4 --props=render-spec.json` from `resources/remotion/`. The live invocation is deferred at the CLI adapter (`TODO(oma-deferred): remotion render`).
- **Fallback**: write a deterministic placeholder mp4 derived from the render-spec so the run dir + manifest are still well-formed with zero toolchain.
- **MPT (`--compositor mpt`)**: inject the agent-written script (custom-script mode); keys env-only + log masking.

## Step 6: Write Artifacts

1. Copy every external asset into the run dir and hash it (`sha256`); no URL refs.
2. Validate each asset-bus schema (`script` / `timing` / `render-spec` / `manifest`) — `schemaVersion` must be `"1.0"`.
3. Build `manifest.json`: `runId`, `mode`, `providers{...}`, `assets[{path,sha256,bytes,seed}]`, `outputs{video,durationSec,sha256}`, `cost{usd,breakdown}`, `warnings[]`, `exitCode`.
4. If `--no-brief-in-manifest`: replace `prompt` with `promptSha256`.

## Step 7: Report

1. Print a one-line status per capability to stderr:
   - `[oma video] <capability> <provider> ok (Xs)`
   - `[oma video] <capability> <provider> fallback -> <fallback>`
2. Print the run-dir path + the mp4 path.
3. For `--format json`: write `{exitCode, runDir, manifestPath, outputs}` to stdout as one JSON object.

## Step 8: Exit Code Aggregation (aligned with `oma search fetch`)

- Success (mp4 + valid manifest) -> exit 0 (fallbacks recorded in `warnings`).
- Otherwise pick the most specific code:
  - `safety-refused` -> 2
  - `not-found` (profile/asset) -> 3
  - `schema-validation` / invalid input -> 4
  - `provider-unavailable` / `auth-required` -> 5
  - `timeout` -> 6
  - otherwise -> 1

## On Error

| Situation | Action |
|-----------|--------|
| No provider for a required capability | Exit 5, print `Run: oma video doctor` |
| Remotion toolchain not bootstrapped | Exit 1 (CompositorBootstrapError) + doctor remediation; MPT fallback where applicable |
| Voicebox MCP down | Fall back voicebox-stt -> whisper.cpp -> estimated timing; still emit captions |
| Pexels / Pixelle key absent | Skip provider; fall through to oma-image stills; annotate coverage in `warnings` |
| `demo` with no capture + no Cap | Guided protocol (Step 4b); stop without rendering |
| `--capture` outside `$PWD` or wrong format | Exit 4 with the path/format problem |
| Cost over guardrail, declined | Exit 1 |
| Timeout | Exit 6, manifest records `after_ms` |
| Cancelled (Ctrl+C) | Exit 130 (signal); no manifest if abort was pre-write |

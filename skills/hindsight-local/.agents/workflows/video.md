---
name: video
description: Agent-native, key-optional video generation workflow that turns a brief into a finished MP4 — script → parallel asset generation (voice/visual/caption) → render-spec → Remotion compositor (MPT fallback) → QA loop → output + manifest
disable-model-invocation: true
---

# MANDATORY RULES: VIOLATION IS FORBIDDEN

- **Response language follows `language` setting in `.agents/oma-config.yaml` if configured.**
- **NEVER skip steps.** Execute from Step 1 in order. Report completion of each step before proceeding.
- **Key-optional by default (backend rule 11).** The baseline path uses **zero external API keys**: the agent writes the script, oma-voice does TTS, oma-image does visuals, captions are key-free, Remotion composites. Every paid upgrade (Pexels stock, Pixelle AIGC) sits behind a key-free fallback and is **off by default**. Never disable the fallback to force a real call, and never silently drop a requested real path.
- **Determinism boundary = `render-spec.json` + asset files (+ seed + embedded Pretendard).** "Reproducible from script/assets, not from brief." Never edit assets or render-spec by hand after generation; re-run the stage that produced them. (Live web capture is **outside** this boundary; its manifest carries `nondeterministic: true`.)
- **Web capture: human-driven, no credential automation, masked.** For `demo --source web`, the tool only opens a headed browser and records — a **human** performs the entire on-screen flow and any login. **NEVER** script, type, or automate credentials of any kind. The `--url` and any query tokens are **masked** in logs and the manifest; credentials are never stored or printed; all outputs stay in the run dir. The skill provides only the mechanism — assume and prescribe **nothing** about what the flow is or what the recording is for, and bake in **no** platform- or policy-specific guidance.
- **The `oma video` CLI owns the pipeline. This workflow owns the brief, the agent-authored script, the QA loop, and decision checkpoints.** Do NOT reimplement orchestration, provider selection, or rendering in the workflow.
- **You MUST use MCP tools where the project provides them.**
  - Use memory tools (read/write/edit) for run tracking. Memory path: configurable via `memoryConfig.basePath` (default: `.serena/memories`).
  - Tool names: configurable via `memoryConfig.tools` in `.agents/mcp.json`.
- **Read the oma-video skill BEFORE starting.** Read `.agents/skills/oma-video/SKILL.md` and follow its Core Rules and execution protocol, including `resources/execution-protocol.md`. If the skill is not installed, stop and ask the user to run `oma install` first.

---

> **Vendor note:** This workflow executes inline. The script-authoring step (Step 3) is performed by the running agent itself — **the agent is the LLM key** (agent-as-key). A subagent may be spawned only for broad multi-scene research; the asset pipeline (Steps 4-7) runs through the `oma video` CLI, not through subagents.

---

## L1 Decision Events

Emit required L1 decisions by calling `oma state:emit` directly, as documented in `.agents/skills/_shared/runtime/event-spec.md`.

This workflow has two required checkpoints: **mode-selection** (Step 2) and **cost-confirmation** (Step 5). Do not skip either emit/verify pair.

---

## Mode Routing

Resolve the mode first — it determines aspect, source, visual track, and compositor. If the user did not name a mode, infer from intent and confirm at Step 2.

| mode | aspect | source | visual track (default → opt) | compositor | output |
|------|:---:|------|------|------|------|
| `shorts` | 9:16 | synthetic (topic → clip) | oma-image stills · Pexels (opt) · Pixelle AIGC (opt) | Remotion · MPT alt | `shorts-<slug>.mp4` |
| `explainer` | 16:9 / 9:16 | README · code · data | oma-slide frames + oma-image diagrams + code | Remotion (deterministic) | `explainer-<slug>.mp4` |
| `demo` | 16:9 | `--source file` (Cap / capture file / guided) · `--source web` (headed browser at `--url`, human-driven) | raw footage (default) · Remotion intro · zoom · callouts (`--polish`) | Remotion polish | `demo-<slug>.mp4` |

Intent heuristics: "reel / TikTok / short / hook" → `shorts`; "walkthrough / how it works / from the README / explain the architecture" → `explainer`; "record / screen / show the app running / product demo" → `demo`.

For `demo`, also resolve the **source**: a recorded file or Cap → `--source file`; a live web app at a URL → `--source web --url <url>`. The web path opens a headed browser and records while a **human** drives the on-screen flow (whatever it is) and presses ENTER to stop; the tool **never automates a login or prescribes the flow**. Example categories are equal and illustrative only — demo, walkthrough, onboarding, repro, app-review screencast. Raw footage is the default output; `--polish` overlays the Remotion `Demo` composition.

---

## Cost Guardrail & Key-Optional Notes (read before Step 4)

- **Guardrail**: default `cost.guardrail_usd: 0.20` in `.agents/skills/oma-video/config/video-config.yaml` (reused from oma-image). Any provider whose estimated cost meets or exceeds the guardrail requires explicit confirmation (`-y` / `--yes` or the Step 5 checkpoint). `--max-usd <n>` overrides the threshold.
- **Key-optional pairs** (real path is gated; fallback is always wired):

  | capability | real (key/resource) | key-free fallback | deferred marker |
  |------|------|------|------|
  | stock video | Pexels (`PEXELS_API_KEY`) | oma-image stills + Ken Burns | `TODO(oma-deferred): pexels` |
  | AIGC video | Pixelle-MCP + RunningHub (`RUNNINGHUB_API_KEY`) | oma-image stills | `TODO(oma-deferred): pixelle` |
  | caption timing | TTS-native timestamps | voicebox-stt → whisper.cpp → estimate | — |
  | premium TTS | (not needed — oma-voice is local) | — | — |

- **Pixelle AIGC is a community MCP**: off by default, requires one-time explicit user consent plus a source review before connecting, and is always cost-gated on RunningHub credits.
- **Fallbacks are not failures.** A run that used the key-free path is a successful run — the manifest records `pathTaken: fallback` and a warning, not an error.

---

## Step 1: Resolve Brief & Preflight

1. Capture the brief from the user's request. If absent, ask:
   ```
   What is the video about? Give me a one-line brief, and a mode if you have one (shorts / explainer / demo).
   ```
2. // turbo
   Run the readiness check and surface gaps before spending any time on assets:
   ```bash
   oma video doctor --format json
   ```
   This reports Node / Chromium / FFmpeg, Voicebox MCP (oma-voice), oma-image vendors, optional Pixelle-MCP, Cap, and (for `demo --source web`) Playwright web-capture readiness. If Remotion is not yet installed, doctor performs the **install-once** bootstrap — do not install during a run. For web capture, `oma video doctor --install-playwright` is the one-time install (`npm i playwright` + chromium); it reuses the project's Playwright when present.
3. If doctor reports a hard blocker for the chosen mode (e.g. no compositor for `shorts`/`explainer`), report the remediation and stop. If only an optional provider is missing (Pexels, Pixelle, Cap), note it and continue on the fallback.
4. Record run start with the memory write tool: brief summary, requested mode, doctor result.

---

## Step 2: Confirm Mode & Plan

1. State the resolved **mode**, **aspect**, **locale**, **caption style**, **visual track**, and **compositor** you intend to use, and the expected output name.
2. For `demo` mode, state up front: **"Capture is performed by a human."** Resolve the **source**:
   - `--source file`: if no `--capture <path>` is available and Cap CLI is not present, ask the user to record and provide the file path before proceeding.
   - `--source web --url <url>` (any URL — local/staging/prod): state that the tool opens a **headed browser** and records while the **human drives the entire on-screen flow** and presses ENTER to stop; **no login is ever automated**, and the `--url` plus any query tokens are masked in logs and the manifest. The capture size is derived from `--aspect`/`--device` (no hardcoded size). If Playwright is unresolvable, or the session has no interactive TTY (CI / `-y` / no stdin), say so and fall back to the guided protocol — never hang. (`--capture-stop duration:<sec>|selector:<css>` supplies a non-interactive stop for CI.)
3. **You MUST get user confirmation on the mode/plan before Step 3.**
4. After the user confirms, emit and verify the mode-selection decision:
   ```bash
   oma state:emit "decision.made" '{"subject":"video.mode-selection","decision":"Proceed with the confirmed mode and pipeline plan.","rationale":"The user confirmed mode, aspect, visual track, and compositor before asset generation."}'
   oma state:verify --workflow video --checkpoint mode-selection
   ```

---

## Step 3: Author `script.json` (agent-as-key)

The agent writes the script — this is the start of the determinism boundary. Do NOT call an external LLM; you are the script provider.

1. Produce a script honoring the `script.json` schema (`mode, aspect, locale, title, scenes[{id, durationSec, narration, onScreenText, visual{kind,prompt,ref,source}, transition}], music, brand`).
2. Respect limits from `.agents/skills/oma-video/config/video-config.yaml` (`max_duration_sec: 180`, `max_scenes: 40`). Keep narration tight and per-scene so scene boundaries map cleanly to TTS timing.
3. Mode-specific sourcing:
   - `shorts`: a hook-first synthetic script from the topic; each scene gets a `visual.prompt` for oma-image.
   - `explainer`: ground scenes in the README / code / data the user pointed to; mark scenes that should become oma-slide frames vs oma-image diagrams.
   - `demo`: narration + on-screen callouts over the captured footage; visual refs point at the ingested capture segments.
4. Translate narration / on-screen text via oma-translator when `locale` differs from the source language (key-free). If oma-translator is absent, keep the source text and let the run warn.
5. // turbo
   Hand the agent-authored script to the CLI as a custom script and let it validate against the schema. Use `--dry-run` for the first pass so the pipeline emits `script.json` + `render-spec.json` + `manifest.json` **without rendering**:
   ```bash
   oma video generate "<brief>" --mode <mode> --aspect <aspect> --locale <lang> \
     --captions <tiktok|lower-third|none> --visual <auto|generate|stock|aigc|slide> \
     --voice <profile|none> --music <upbeat|calm|none> --duration <sec|auto> \
     --compositor <remotion|mpt> --seed <n> --dry-run --format json
   ```
6. Review the emitted `script.json` for scene count, durations, and narration quality. Iterate here — fixing the script is cheap; fixing a render is not.

---

## Step 4: Parallel Asset Generation (voice / visual / caption)

The CLI orchestrator fans out the asset tracks per the asset bus. Trigger the full (non-dry) run; the orchestrator runs the tracks and writes them into the run directory. **Do not author assets by hand.**

```bash
oma video generate "<brief>" --mode <mode> [same flags as Step 3, without --dry-run] --format json
```

The three tracks (per `.agents/skills/oma-video/SKILL.md` and its execution protocol):

- **Voice** (oma-voice / Voicebox MCP) → `audio/narration-*.wav` + `timing.json`. Timing source preference: TTS-native → `voicebox-stt` (transcribe the generated wav) → `whisper.cpp` → `estimated`. If oma-voice is down, the run falls back to silent + estimated timing and warns — it does not hard-fail.
- **Visual** (per-scene, fallback chain `oma-image → pexels → pixelle`) → `visuals/scene-NN.*`. Default is key-free oma-image stills (aspect snapped to the nearest 16-multiple; Remotion crops to exact frame). `--visual stock` engages Pexels only when `PEXELS_API_KEY` is set; `--visual aigc` engages Pixelle only after consent + cost gate. Each scene that falls back is recorded with `pathTaken: fallback`.
- **Caption** (key-free) → `captions.srt` / `.vtt`, aligned to `timing.json`, styled `tiktok` or `lower-third`, with platform safe-area presets. Non-source locales translate via oma-translator; if absent, captions keep the source locale and warn.

Report which path each track took (real vs fallback) and surface any warnings.

### Demo capture track (`--mode demo`)

For `demo`, the orchestrator produces the footage in place of synthetic visuals, dispatched on `--source`:

- **`--source file`** — ingest the `--capture` path (absolutized, `$PWD`-guarded, format-validated). No `--capture` and no Cap → guided protocol.
- **`--source web --url <url>`** — the orchestrator runs the **headed web-capture** path:
  1. Opens a real browser at `--url`, waits for load/hydration (`networkidle`, optional `--ready-selector <css>`), at a size derived from `--aspect`/`--device` (no hardcoded size).
  2. Prompts on the terminal: the **human performs the entire on-screen flow** (whatever it is — multi-page popups / new tabs / redirects are all recorded generically) and presses **ENTER** to stop. The tool **never automates a login**; if the flow needs one, the human does it.
  3. Records to a real `capture.mp4` in the run dir, validated with ffprobe. The `--url` and any query tokens are **masked** in logs and `manifest.json`; outputs stay in the run dir.
  4. **Fallback (key-optional, non-blocking):** if Playwright is unresolvable, or there is no interactive TTY (CI / `-y` / no stdin), the orchestrator falls back to the **guided protocol** and warns — it never hangs. `--capture-stop duration:<sec>|selector:<css>` gives CI a non-interactive stop instead of the ENTER prompt.
  5. Live capture is **outside** the determinism boundary, so the manifest records `nondeterministic: true`.

> **Optional fast-path (agent sessions only):** when an agent session has a Playwright/Chrome MCP available, it may drive the headed flow through that MCP as a complement. This is **not** the primary path — the CLI web-capture subprocess remains canonical, and the same rules hold (human drives any login, URL/tokens masked, run-dir-only).

---

## Step 5: Cost Gate & `render-spec.json`

1. Inspect the cost estimate the orchestrator computed across providers (`cost.usd` + breakdown in the manifest/JSON output).
2. **If the estimate meets or exceeds the guardrail** (default $0.20, or `--max-usd`), pause and present the breakdown. **You MUST get user confirmation before the paid render proceeds.** Then emit and verify:
   ```bash
   oma state:emit "decision.made" '{"subject":"video.cost-confirmation","decision":"Proceed with the estimated paid cost or fall back to the key-free path.","rationale":"Estimated cost crossed the guardrail; the user confirmed spend or chose the fallback."}'
   oma state:verify --workflow video --checkpoint cost-confirmation
   ```
   If the user declines, re-run with the key-free providers (drop `--visual stock|aigc`) — the fallback chain keeps the run alive.
3. If the estimate is under the guardrail, note "cost under guardrail ($X.XX < $0.20)" and continue without a confirmation prompt.
4. Confirm `render-spec.json` was written. This is the **deterministic compute boundary**: `compositor, composition, fps, dimensions, durationInFrames, audio, scenes[], captions, background, seed`. The seed is embedded so re-renders are byte-identical.

---

## Step 6: Composite (Remotion → MPT fallback)

1. The orchestrator renders via the selected compositor:
   - **Remotion** (default, all modes): renders the vendored `Shorts` / `Explainer` / `Demo` composition from `render-spec.json` props, with embedded Pretendard for cross-machine identical output. Long renders are SIGINT-abortable.
   - **MoneyPrinterTurbo** (`--compositor mpt`, shorts e2e alt): the agent-written script is injected in custom-script mode; provider keys are env-only and masked in logs.
   - **Demo raw vs `--polish`**: for `demo`, the **default** is the raw captured footage copied through as the output (no compositor over-processing). `--polish` overlays the Remotion `Demo` composition (intro / captions / zoom) with the captured `capture.mp4` as the full-frame background.
2. If Remotion bootstrap fails (`CompositorBootstrapError`), the doctor remediation is the fix path — re-run `oma video doctor` to install once, then re-render. Do not attempt an ad-hoc install mid-run.
3. // turbo
   To reproduce or re-render an existing run without regenerating assets (deterministic from the spec):
   ```bash
   oma video render <runDir> --format json
   ```
4. Confirm the output MP4 exists in the run directory and matches the expected `<mode>-<slug>.mp4` name.

---

## Step 7: QA Loop

Review the finished video against the brief and the quality bars. Iterate by re-running the **smallest** upstream stage that owns the defect — never patch the artifact.

1. **Checklist** (priority order: correctness → sync → readability → polish):
   - Output plays; duration matches the script total within tolerance.
   - Narration audio is present (or intentionally silent) and aligns to scenes.
   - Captions are synced to `timing.json`, within the safe area, and legible (greedy-wrap, Pretendard, design rule 2).
   - Visuals match each scene's intent; no placeholder leakage unless the run intentionally used the fallback.
   - Aspect / dimensions are correct for the mode; branding/music applied as requested.
2. **Route each defect to its stage:**
   - script/narration/scene-count → **Step 3** (re-author script).
   - audio/timing → **Step 4** voice track (check oma-voice, re-synthesize).
   - wrong/placeholder visual → **Step 4** visual track (adjust prompt or `--visual` mode).
   - missing/incomplete demo capture → **Step 4** demo capture track (re-run the web capture; adjust `--ready-selector`/`--capture-timeout`, or fall back to `--source file`).
   - caption sync/wrap/locale → **Step 4** caption track (or oma-translator).
   - layout/transition/crop → **Step 6** render-spec → re-render (for `demo`, toggle `--polish`).
3. **Determinism guard:** when validating reproducibility, run the golden harness — render-spec and assets must be byte-identical:
   ```bash
   OMA_VIDEO_MOCK=1 oma video generate "<brief>" --mode <mode> --seed <n> --dry-run --format json
   ```
4. **If the same defect persists after 2 fix attempts**, stop iterating blindly: present 2-3 alternative approaches (different visual track, different mode framing, different compositor) and get the user to choose before the next attempt. Record discarded attempts.
5. Repeat until the checklist passes or the user accepts the result.

---

## Step 8: Output & Manifest

1. Confirm the run directory is complete (mirrors `.agents/results/videos/<runId>-<mode>/`):
   ```
   script.json · timing.json · render-spec.json
   audio/narration-*.wav
   visuals/scene-*.{jpg,png,mp4}        # synthetic modes
   capture.mp4                           # demo: ingested or web-captured footage
   captions.srt (+ .vtt)
   <mode>-<slug>.mp4
   manifest.json
   ```
2. Verify `manifest.json` is the reproducibility record: `runId, mode, providers{...}, assets[{path,sha256,bytes,seed}], outputs{video,durationSec,sha256}, cost{usd,breakdown}, warnings[], exitCode`. All external assets are copied into the run dir and hashed — no URL refs. For `demo --source web`, the manifest carries `nondeterministic: true`, the capture provider id (`playwright-web`), and a **masked** `--url` (query tokens stripped) — never the raw URL or any credential.
3. Report to the user:
   - Output MP4 path (absolute) and duration.
   - Providers used per track, and which tracks took the **fallback** path.
   - Final cost (`$0.00` on the all-key-free path).
   - Any warnings (silent audio, source-locale captions, placeholder visuals).
   - Reproduce command: `oma video render <runDir>`.
4. Record run completion with the memory write tool: run dir, output path, providers, cost, warnings.

---

## Exit Codes (aligned with `oma search fetch`)

`0` ok · `1` generic · `2` safety · `3` not-found · `4` invalid-input · `5` auth-required · `6` timeout.

Common error → action map:

| error | exit | action |
|------|:---:|------|
| `ProviderUnavailableError` | 5 | a required provider is down → run `oma video doctor`, fix or fall back |
| `CompositorBootstrapError` | 1 | Remotion not installed → `oma video doctor` install-once, then re-render |
| `CostGuardrailError` | confirm | estimate crossed guardrail → Step 5 confirmation or drop paid providers |
| `CaptureRequiredError` | guided | demo needs footage → `--source file`: ask user to record + pass `--capture <path>`. `--source web`: Playwright unresolvable / no TTY / empty recording → fall back to the guided protocol (no hang) |
| `SchemaValidationError` | 4 | script/render-spec invalid, or `--source web` without `--url` → fix in Step 3 (or supply `--url`), re-validate with `--dry-run` |

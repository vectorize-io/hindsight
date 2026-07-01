---
name: oma-video
description: Short-form, explainer, and demo video generation via a key-optional 3-tier router. Composes scripts, oma-voice narration, oma-image/oma-slide/stock visuals, key-free captions, and a vendored Remotion compositor into reproducible run directories. Routes three modes — shorts/reels (9:16), explainer (16:9 README/code/data), and demo/walkthrough (screen capture, incl. supervised headed web-app capture of any URL). Use for video, shorts, reels, short-form, demo, explainer, walkthrough, screencast, web capture, video generation, 영상, 숏폼, 쇼츠, 릴스, 데모, 설명 영상.
---

# Video Agent - Short-form, Explainer & Demo Router

## Scheduling

### Goal
Generate finished `.mp4` videos through a key-optional, 3-tier (CLI-first / MCP / guided) provider router while preserving deterministic asset buses (script -> timing -> render-spec), reproducible manifests, cost controls, and capture-path safety.

### Intent signature
- User asks for a short-form video, shorts/reels clip, TikTok/YouTube Short, explainer, demo, walkthrough, or screencast.
- User wants a topic, README, code, or data turned into a narrated, captioned video.
- Another skill needs shared video-generation infrastructure (script -> assets -> render).

### When to use

- Generating short-form video (shorts / reels) from a topic or brief (`--mode shorts`, 9:16)
- Generating an explainer from a README, code, or data set (`--mode explainer`, 16:9 / 9:16)
- Producing a demo / walkthrough from a screen capture file (`--mode demo --source file`, 16:9)
- Supervised headed web-app capture of any URL (`--mode demo --source web --url <url>`) — a human drives the on-screen flow; the tool only opens a headed browser and records. Example categories are equal and illustrative only: demo, walkthrough, onboarding clip, bug repro, app-review screencast.
- Re-rendering an existing run deterministically from `render-spec.json`
- Other skills needing video-generation infrastructure (shared invocation via `--format json`)

### When NOT to use

- Generating a single still image -> use `oma-image`
- Generating a slide deck / presentation -> use `oma-slide` (this skill calls it internally for explainer frames)
- Generating speech audio only (no video) -> use `oma-voice`
- Non-linear video editing of an existing finished mp4 -> out of scope (OpenCut-MCP deferred)
- Supervised headed web capture is in-scope (`--source web`); live streaming is out of scope

### Expected inputs
- A brief (topic / README path / data) plus optional mode, aspect, locale, captions, visual, voice, music, duration, compositor, capture path, seed
- For `demo` `--source file`: a screen-capture file path (`--capture`) or Cap availability
- For `demo` `--source web`: a target `--url` (any URL — local/staging/prod), optional `--device`/`--ready-selector`/`--show-cursor`/`--polish`/`--capture-timeout`; capture size is derived from `--aspect`/`--device` (no hardcoded size); a resolvable Playwright + an interactive TTY (else the run falls back to the guided protocol)
- Authentication/environment state for oma-voice (Voicebox MCP), oma-image vendors, and optional Pexels / Pixelle keys

### Expected outputs
- A run directory under `.agents/results/videos/<timestamp>-<shortid>-<mode>/`
- Deterministic asset bus: `script.json`, `timing.json`, `render-spec.json`
- `audio/`, `visuals/`, `captions.srt` / `captions.vtt`, the rendered `<mode>-<slug>.mp4`
- `manifest.json` with providers, asset hashes, cost breakdown, and exit code

### Dependencies
- `oma video generate` CLI + central error module (exit codes aligned with `oma search fetch`)
- oma-voice (Voicebox MCP), oma-image, oma-slide as key-free fallback providers
- Vendored Remotion project at `resources/remotion/` (compositor)
- `resources/vendor-matrix.md`, `resources/execution-protocol.md`, `resources/prompt-tips.md`, `config/video-config.yaml`

### Control-flow features
- Branches by mode (shorts / explainer / demo), aspect, visual strategy, provider availability, cost threshold, capture requirement, and path safety
- Runs a per-capability fallback chain (real key/resource -> key-free fallback) per backend rule 11
- Reads briefs/captures and writes assets, render-spec, and manifests
- Calls external resources: Voicebox MCP, oma-image vendors, Remotion toolchain, optional Pexels / Pixelle

## Structural Flow

### Entry
1. Validate that the brief carries enough mode/topic signal (or infer the mode from keywords).
2. For `demo`, confirm a capture path exists (or Cap is available); otherwise enter the guided protocol.
3. Resolve defaults from `config/video-config.yaml` -> env vars -> CLI flags; check output path safety and limits.

### Scenes
1. **PREPARE**: Resolve mode/aspect/locale, clarify or amplify the brief, choose the visual + compositor strategy.
2. **ACQUIRE**: Probe provider availability (voice / visual / caption / compositor), validate capture path, check cost.
3. **ACT**: Run the mode pipeline — script -> (voice ∥ visuals ∥ captions) -> render-spec -> compositor render.
4. **VERIFY**: Validate every asset-bus schema, manifest hashes, exit code, and the output mp4.
5. **FINALIZE**: Return the run-dir path, the mp4 path, and any provider/coverage warnings.

### Transitions
- If the brief lacks a clear mode, infer from keywords (shorts/reels -> shorts; README/code -> explainer; capture -> demo) and show the user the inferred plan before generating.
- If the selected visual provider key is absent (Pexels / Pixelle), fall through the chain to the key-free oma-image stills + Ken Burns fallback and annotate coverage.
- If `demo` `--source web` has a `--url`, dispatch the headed web-capture path (human-driven flow, ENTER to stop); if Playwright is unresolvable OR there is no interactive TTY, fall back to the guided protocol (no hang).
- If `demo` `--source file` has no capture and Cap is unavailable, emit the guided capture protocol and stop (exit code maps to capture-required).
- If estimated cost (Pixelle / RunningHub credits) exceeds the guardrail, require confirmation unless bypassed.

### Failure and recovery
- If a provider is unavailable, try the next provider in the capability's `order`; only chain exhaustion is a stage failure.
- If the Remotion toolchain is not bootstrapped, point the user to `oma video doctor`; fall back to the MPT compositor where applicable.
- If Voicebox MCP is down, fall back through voicebox-stt -> whisper.cpp -> estimated timing (still produces captions).
- If the brief locale is non-source, translate via oma-translator (key-free); if absent, warn and keep source text.

### Exit
- Success: `<mode>-<slug>.mp4` and `manifest.json` exist in the run directory; all schemas validate.
- Partial success: video renders with a key-free fallback in place of a paid provider; coverage is annotated in warnings.
- Failure: no video is produced and the route/cost/capture/auth/safety blocker is explicit in the exit code + manifest.

## Logical Operations

### Actions
| Action | SSL primitive | Evidence |
|--------|---------------|----------|
| Validate brief + mode | `VALIDATE` | Clarification protocol, mode inference |
| Select provider strategy | `SELECT` | Vendor matrix, `providers.*.order`, availability |
| Read brief / capture | `READ` | Brief text, `--capture` path |
| Generate script | `CALL_TOOL` | AgentScriptProvider -> `script.json` |
| Synthesize narration + timing | `CALL_TOOL` | oma-voice -> `audio/*.wav` + `timing.json` |
| Produce visuals | `CALL_TOOL` | oma-image / oma-slide / stock -> `visuals/*` |
| Build captions | `WRITE` | key-free `captions.srt` / `.vtt` from timing |
| Compose render-spec | `WRITE` | `render-spec.json` (determinism boundary) |
| Render video | `CALL_TOOL` | Remotion / MPT compositor -> `<mode>-<slug>.mp4` |
| Validate result | `VALIDATE` | Schema parse, manifest hashes, exit code |
| Report output | `NOTIFY` | Run-dir + mp4 path summary |

### Tools and instruments
- `oma video generate`, `oma video doctor`, `oma video list-providers`, `oma video render`
- Provider adapters: AgentScript, oma-voice, oma-image, oma-slide, Pexels, Pixelle, oma-captions, Cap, Remotion, MPT
- Vendored Remotion project (`resources/remotion/`), prompt tips, vendor matrix, video config

### Canonical command path
```bash
oma video doctor
oma video generate "<brief>" --mode shorts --aspect auto --captions tiktok --format json
```

Explainer from a README, with a deterministic seed:
```bash
oma video generate "explain this project" --mode explainer --aspect 16:9 --seed 42 --out ./out
```

Demo from a screen capture:
```bash
oma video generate "feature walkthrough" --mode demo --capture <absolute-path>.mp4
```

Deterministic re-render from an existing run:
```bash
oma video render .agents/results/videos/20260603-143052-ab12cd-shorts
```

### Resource scope
| Scope | Resource target |
|-------|-----------------|
| `LOCAL_FS` | Briefs, captures, assets, render-spec, run dir, manifests |
| `PROCESS` | oma-image / oma-slide CLIs, Remotion / MPT, Cap CLI, Playwright web-capture driver (subprocess) |
| `NETWORK` | Voicebox MCP (localhost), oma-image vendor APIs, the user-supplied `--url` for web capture (masked in logs/manifest), optional Pexels / Pixelle / RunningHub |
| `CREDENTIALS` | oma-image vendor auth, optional `PEXELS_API_KEY` / `RUNNINGHUB_API_KEY`. Web capture handles NO credentials — a human logs in if the flow needs it; nothing is stored or printed. |

### Preconditions
- Brief carries enough signal for the mode, or the user approves the inferred/amplified plan.
- Output path is inside `$PWD` (or `--allow-external-out` is set).
- For `demo` `--source file`: the capture path exists, is absolute/$PWD-guarded, and is a valid format.
- For `demo` `--source web`: a `--url` is supplied (else `SchemaValidationError`); a resolvable Playwright + an interactive TTY exist (else the run falls back to the guided protocol).
- Required provider availability holds for the chosen (non-fallback) path, or the fallback is acceptable.

### Effects and side effects
- Creates a run directory with assets, render-spec, captions, mp4, and manifest.
- oma-voice plays narration on the speakers as a side effect of synthesis.
- May call paid or rate-limited providers (Pexels / Pixelle / RunningHub) only when keys are present.

### Guardrails

1. **Clarify or infer before invoking**: if the mode/topic is ambiguous, infer the mode from keywords and show the user the plan, or ask. Do NOT silently render from a vague brief. See `Clarification Protocol` below.
2. **Key-optional dispatch (backend rule 11)**: every external capability has a real (key/resource) path AND a key-free fallback. Paid providers (Pexels, Pixelle) auto-enable only when their env key is present; otherwise the chain falls through to oma-image stills + Ken Burns. Default providers (oma-voice local, oma-image, oma-slide, Remotion) are key-free, so auto-triggering on keywords is safe.
3. **Cost guardrail**: confirm before runs whose estimated cost is >= `cost.guardrail_usd` (`$0.20`, configurable) or `--max-usd`. `--yes` / `OMA_VIDEO_YES=1` bypass. Local/free paths carry zero cost.
4. **Path safety**: output paths outside `$PWD` require `--allow-external-out`. `--capture` is absolutized, `$PWD`-guarded, and format-validated; external assets are copied into the run dir and hashed (no URL refs).
5. **Cancellable**: SIGINT/SIGTERM aborts in-flight provider calls, the render, and the orchestrator.
6. **Deterministic outputs**: `render-spec.json` + asset files (+ seed + embedded Pretendard font) are the determinism boundary. Re-rendering the same render-spec is byte-stable; `OMA_VIDEO_MOCK=1` replays golden fixtures.
7. **Limits**: `limits.max_duration_sec` = 180, `limits.max_scenes` = 40 (wall-time + memory bound).
8. **Community-MCP consent**: Pixelle-MCP is off by default and requires one-time explicit consent + source review before connecting; RunningHub credits gate on `--max-usd`.
9. **Demo is human-in-the-loop**: capture is performed by a human. For `--source file` the skill guides but does not screen-record autonomously; for `--source web` the tool only opens a headed browser and records while the human drives the entire on-screen flow (interactive ENTER to stop). The mechanism prescribes nothing about what the flow is or what the recording is for.
10. **Web-capture security**: NO credential automation of any kind — if a flow needs a login, a human performs it. The driver runs as a subprocess under the resolved Playwright (never imported into the CLI). The `--url` and any query tokens are masked in logs and in `manifest.json`; credentials are never stored or printed. Recording and all outputs are confined to the run dir. On-screen sensitive input is captured as-is — the user controls the flow. Multi-page navigation (popup / new tab / redirect) is recorded generically, with no assumption about the flow's shape.
11. **Web capture is key-optional + non-blocking**: web capture is the real branch; the guided protocol is the fallback when Playwright is unresolvable OR there is no interactive TTY (CI / `-y` / no stdin) — the run falls back to guided and never hangs. Live capture is outside the determinism boundary, so the manifest records `nondeterministic: true`.
12. **Exit codes align with `oma search fetch`** (0 ok, 1 generic, 2 safety, 3 not-found, 4 invalid-input, 5 auth-required, 6 timeout).

### Clarification Protocol

Before invoking `oma video generate`, the calling agent runs this checklist. **If any answer is "no / unknown", clarify or infer-and-confirm with the user first.**

**Required signal (must be present or inferable):**
- [ ] **Mode**: shorts / explainer / demo? Infer from keywords (shorts/reels/쇼츠/릴스 -> shorts; README/code/data/explain/설명 -> explainer; demo/walkthrough/capture/데모 -> demo).
- [ ] **Topic / source**: what is the video about? (a topic, a README/code path, a capture file, or — for `demo` `--source web` — a `--url`)
- [ ] **For `demo`**: `--source file` (a `--capture` path) or `--source web` (a `--url`)? For `--source web`, state up front that a **human drives the on-screen flow** and that the tool **never automates any login**.

**Strongly recommended (ask if absent AND not inferable):**
- [ ] **Aspect**: `9:16` (shorts/reels), `16:9` (explainer/demo), `1:1`, or `auto` (snaps to the mode default).
- [ ] **Locale**: narration + caption language (default from config; translated via oma-translator when non-source).
- [ ] **Captions**: `tiktok` (centered, animated), `lower-third`, or `none`.
- [ ] **Duration**: target seconds (<= 180) or `auto` (derived from the script).
- [ ] **Voice / music**: voice profile or `none`; music `upbeat` / `calm` / `none`.

**Amplification shortcut.** For a one-line brief (e.g. "shorts about Jeju coffee"), do not pop a questionnaire if the request is genuinely simple. Instead **amplify inline and show the user** the inferred plan before invoking:

> User: "make a short about Jeju coffee"
> Agent: "I'll generate this as: *mode `shorts`, 9:16, ~30s, oma-image stills with Ken Burns, TikTok captions, locale `en`, calm music*. Proceed, or adjust mode/aspect/voice?"

Skip clarification when the user authored a full brief (mode + topic + aspect + captions). Respect their flags verbatim.

**Output language.** Narration and on-screen text are authored in the requested locale. Image-generation prompts passed to oma-image are sent in English (image models are trained predominantly on English captions); translate the user's request and show the translated version during amplification.

### Modes

| Mode | Aspect | Source | Default visual | Compositor | Output |
|------|:---:|--------|----------------|------------|--------|
| `shorts` | 9:16 | synthetic (topic) | oma-image stills + Ken Burns; Pexels (key) · Pixelle AIGC (key) opt | Remotion · MPT alt | `shorts-<slug>.mp4` |
| `explainer` | 16:9 / 9:16 | README · code · data | oma-slide frames + oma-image diagrams + code | Remotion (deterministic) | `explainer-<slug>.mp4` |
| `demo` | 16:9 | `--source file` (Cap / capture file) · `--source web` (headed browser at `--url`) | raw footage (default) · Remotion intro · zoom · callouts (`--polish`) | Remotion polish | `demo-<slug>.mp4` |

### 3-Tier Integration

| Tier | Surface | Providers | Trigger |
|:---:|---------|-----------|---------|
| 1 | CLI-first (subprocess, deterministic) | Remotion render, MPT, oma-image, oma-slide, oma-voice | always available (key-free defaults) |
| 2 | MCP | Voicebox MCP (voice/timing), Pixelle-MCP (AIGC, off by default) | MCP server reachable; Pixelle needs explicit consent + key |
| 3 | Guided (human-in-the-loop) | Playwright headed web capture (`--source web`, human drives the flow), Cap (capture), openscreen fallback | `demo` mode; web capture needs a resolvable Playwright + a TTY (else guided protocol) |

### Invocation

#### Standalone
```
/oma-video make a 30s short about Jeju coffee
/oma-video --mode explainer --aspect 16:9 explain this project from the README
/oma-video --mode demo --source file --capture ~/recordings/walkthrough.mp4 feature demo
/oma-video --mode demo --source web --url http://localhost:3000 record my app flow
/oma-video --mode demo --source web --url <url> --ready-selector "#app" --polish onboarding clip
```

#### Shell CLI
```
oma video generate "<brief>" [--mode shorts|explainer|demo] \
                             [--aspect 9:16|16:9|1:1|auto] [--locale <lang>] \
                             [--captions tiktok|lower-third|none] \
                             [--visual auto|generate|stock|aigc|slide] \
                             [--voice <profile>|none] [--music upbeat|calm|none] \
                             [--duration <sec>|auto] [--compositor remotion|mpt] \
                             [--capture <path>] \
                             [--source file|web] [--url <url>] [--device <name>] \
                             [--ready-selector <css>] [--show-cursor] [--polish] \
                             [--capture-timeout <sec>] [--capture-stop duration:<sec>|selector:<css>] \
                             [--out <dir>] [--allow-external-out] \
                             [--max-usd <n>] [--seed <n>] [--timeout 600] [-y] \
                             [--dry-run] [--format text|json] [--no-brief-in-manifest]
# --source web: headed browser at --url; capture size derived from --aspect/--device (no hardcoded size).
# A human drives the on-screen flow; press ENTER to stop. NO credential automation. --url/tokens masked.
# Non-interactive (CI / -y / no TTY) or unresolvable Playwright -> falls back to the guided protocol (no hang).
# --capture-stop gives CI a non-interactive stop (duration / selector) in place of the ENTER prompt.
oma video doctor          # Node/Chromium/FFmpeg · Voicebox MCP · oma-image vendors · Pixelle-MCP · Cap · Playwright (web capture)
oma video doctor --install-playwright  # one-time: npm i playwright + chromium (web capture)
oma video list-providers  # availability + key/fallback status
oma video render <runDir>  # re-render from render-spec.json (deterministic)
```

#### Shared Infrastructure (from other skills)
Other skills call `oma video generate --format json` and parse the JSON envelope (`{exitCode, runDir, manifestPath, outputs}`) from stdout. The deterministic boundary is `render-spec.json` + assets, so a downstream consumer can re-render via `oma video render <runDir>` without re-running script/voice/visual generation.

### Output Layout

```
.agents/results/videos/
└── 20260603-143052-ab12cd-shorts/        # {timestamp}-{shortid}-{mode}
    ├── script.json                        # determinism boundary start
    ├── timing.json
    ├── render-spec.json                   # deterministic compute boundary
    ├── audio/
    │   └── narration-01.wav …
    ├── visuals/
    │   └── scene-01.jpg …
    ├── captions.srt
    ├── captions.vtt
    ├── shorts-<slug>.mp4
    └── manifest.json                      # reproducibility record
```

## References

Follow `resources/execution-protocol.md` step by step.
See `resources/vendor-matrix.md` for provider precheck + fallback-chain rules.
Use `resources/prompt-tips.md` for writing effective briefs per mode.
Before submitting, run `resources/checklist.md`.
The vendored Remotion compositor lives at `resources/remotion/` (see its `README.md`).
The web-capture driver lives at `resources/playwright/record.mjs` (runs as a subprocess under the resolved Playwright; never imported into the CLI).
The MPT fallback compositor driver lives at `resources/mpt/driver.py` (consumed by the CLI's mpt-project internals).

### Configuration

Project-specific settings: `config/video-config.yaml`.
Env vars: `OMA_VIDEO_DEFAULT_MODE`, `OMA_VIDEO_DEFAULT_OUT`, `OMA_VIDEO_YES`, `PEXELS_API_KEY`, `RUNNINGHUB_API_KEY` (+ `POLLINATIONS_API_KEY` via oma-image), `OMA_VIDEO_MOCK`, `OMA_VIDEO_PLAYWRIGHT_DIR` (web-capture Playwright override), `OMA_VIDEO_PWTEST` (opt-in web-capture e2e).

- Execution steps: `resources/execution-protocol.md`
- Vendor matrix: `resources/vendor-matrix.md`
- Prompt tips: `resources/prompt-tips.md`
- Checklist: `resources/checklist.md`
- Remotion compositor: `resources/remotion/README.md`
- Context loading: `../_shared/core/context-loading.md`

---
name: oma-image
description: Multi-vendor AI image generation with authentication-aware parallel dispatch. Routes to Codex (gpt-image-2 via ChatGPT OAuth), Antigravity (gemini-2.5-flash-image aka nano-banana via `agy` CLI + Gemini Code Assist), and Pollinations (flux/zimage, free with signup). Use for image generation, image creation, visual asset generation, and AI art.
---

# Image Agent - Multi-Vendor Image Router

## Scheduling

### Goal
Generate images and visual assets through authenticated multi-vendor routing while preserving prompt clarity, reference-image handling, cost controls, and reproducible output manifests.

### Intent signature
- User asks to generate images, visual assets, illustrations, product photos, concept art, mockups, or AI art.
- Another skill needs shared image-generation infrastructure.
- User provides reference images or asks for vendor comparison.

### When to use

- Generating images, visual assets, illustrations, product photos, concept art
- Comparing output between multiple image models for the same prompt
- Producing images from prompts within editor workflows (Claude Code, Codex, Gemini CLI)
- Other skills needing image generation infrastructure (shared invocation)

### When NOT to use

- Editing an existing image or photo manipulation -> out of scope
- Generating videos or audio -> out of scope
- Inline vector art / SVG composition from structured data -> use a templating skill
- Simple asset resizing or format conversion -> use a dedicated image library

### Expected inputs
- Image prompt or creative brief
- Optional vendor, size, quality, count, output directory, and reference images
- Authentication/environment state for Codex, Pollinations, or Gemini

### Expected outputs
- Generated image files under `.agents/results/images/` or requested output directory
- `manifest.json` with prompt, vendor, model, and reproducibility metadata
- Vendor comparison outputs when `--vendor all` is used

### Dependencies
- `oma image generate` CLI and vendor authentication
- Codex image generation, Pollinations API, or Gemini API/CLI strategy
- `resources/vendor-matrix.md`, `resources/prompt-tips.md`, and `config/image-config.yaml`

### Control-flow features
- Branches by prompt ambiguity, vendor auth, cost threshold, reference-image support, path safety, and safety/timeout exit codes
- Calls external vendor APIs/CLIs
- Reads reference images and writes generated images plus manifests

## Structural Flow

### Entry
1. Validate that the request contains enough subject, setting, style, usage, and aspect-ratio signal.
2. Detect attached/reference images and vendor support.
3. Check authentication, cost guardrails, output path, and count limits.

### Scenes
1. **PREPARE**: Clarify or amplify prompt and choose vendor strategy.
2. **ACQUIRE**: Validate auth, references, output path, and provider availability.
3. **ACT**: Invoke `oma image generate` with selected vendor(s), prompt, references, and options.
4. **VERIFY**: Check manifest, output files, exit code, and provider result.
5. **FINALIZE**: Return output paths and relevant warnings.

### Transitions
- If prompt lacks required signal, clarify or show amplified prompt before generation.
- If `--vendor all` is requested, require every requested vendor to be available.
- If reference path is supported by selected vendor, pass it automatically.
- If estimated cost exceeds guardrail, require confirmation unless bypassed.

### Failure and recovery
- If auth is missing, report vendor-specific authentication requirement.
- If reference support is unavailable for the selected vendor, reject with actionable guidance.
- If local CLI is outdated, ask user to run `oma update`.
- If generation times out or is blocked, surface exit code and provider status.

### Exit
- Success: images and manifest exist in the output directory.
- Partial success: some vendors fail in comparison mode and failures are reported.
- Failure: no image is produced and the route/cost/auth/safety blocker is explicit.

## Logical Operations

### Actions
| Action | SSL primitive | Evidence |
|--------|---------------|----------|
| Validate prompt completeness | `VALIDATE` | Clarification protocol |
| Select vendor strategy | `SELECT` | Vendor matrix and auth state |
| Read reference images | `READ` | `--reference` paths |
| Call generation CLI/API | `CALL_TOOL` | `oma image generate` |
| Write image outputs | `WRITE` | Image files and manifest |
| Validate result | `VALIDATE` | Exit code, manifest, files |
| Report output | `NOTIFY` | Final path summary |

### Tools and instruments
- `oma image generate`, `oma image doctor`, `oma image list-vendors`
- Codex, Pollinations, and Gemini provider paths
- Prompt tips, vendor matrix, and image config

### Canonical command path
```bash
oma image doctor
oma image generate "<prompt>" --vendor auto --size auto --quality auto --format json
```

With reference images:
```bash
oma image generate --reference "<absolute-path>" --vendor codex "<prompt>"
```

### Resource scope
| Scope | Resource target |
|-------|-----------------|
| `LOCAL_FS` | Reference images, generated images, manifests |
| `PROCESS` | Provider CLIs and image router commands |
| `NETWORK` | Pollinations/Gemini or provider APIs |
| `CREDENTIALS` | Provider auth and API keys |

### Preconditions
- Prompt is sufficiently specified or user approves amplification.
- Required vendor auth and output permissions exist.
- Reference paths are accessible when used.

### Effects and side effects
- Creates image files and manifests.
- May call paid or rate-limited provider APIs.
- May read attached/reference images.

### Guardrails

1. **Clarify before invoking**: if the user's request is ambiguous about subject, style, composition, or usage context, **ask the user first** or **amplify the prompt explicitly** (showing the user the expanded version for approval). Do NOT silently generate from a vague prompt. See `Clarification Protocol` below.
2. **Authentication-aware dispatch**: detect which vendor CLIs are authenticated and run only those; with `--vendor all`, every requested vendor must be available (strict).
3. **Cost guardrail**: confirm before executing runs whose estimated cost is ≥ `$0.20` (configurable). `--yes` / `OMA_IMAGE_YES=1` bypass. Default vendors `pollinations` (flux/zimage) and `antigravity` (nano-banana via Gemini Code Assist) are free, so auto-triggering on keywords is safe.
4. **Path safety**: output paths outside `$PWD` require `--allow-external-out`.
5. **Cancellable**: SIGINT/SIGTERM aborts in-flight provider calls and the orchestrator.
6. **Deterministic outputs**: every run writes `manifest.json` next to the images for reproducibility.
7. **Max `n` = 5**: wall-time bound.
8. **Exit codes align with `oma search fetch`** (0, 1, 2=safety, 3=not-found, 4=invalid-input, 5=auth-required, 6=timeout).

### Clarification Protocol

Before invoking `oma image generate`, the calling agent runs this checklist against the user's request. **If any answer is "no / unknown", clarify with the user first.**

**Required signal (must be present or inferable):**
- [ ] **Subject**: what is the primary thing in the image? (object, person, scene)
- [ ] **Setting / backdrop**: where is it? (context, environment)

**Strongly recommended (ask if absent AND not inferable from context):**
- [ ] **Style**: photorealistic, illustration, 3D render, oil painting, concept art, flat vector, …?
- [ ] **Mood / lighting**: bright vs moody, warm vs cool, dramatic vs minimal
- [ ] **Usage context**: hero image, icon, thumbnail, product shot, poster? (dictates aspect ratio + composition)
- [ ] **Aspect ratio / resolution**: any `WxH` where each edge is a multiple of 16 between 16 and 3840 and aspect ∈ [1:3, 3:1] (e.g. `1024x1024` square, `2048x1152` 16:9, `3840x2160` 4K UHD, `1024x1536` portrait), or `auto`.

**Amplification shortcut.** For brief prompts (e.g. "a red apple"), do not pop clarifying questions if the request is genuinely that simple. Instead **amplify inline and show the user** the expanded version before invoking:

> User: "a red apple"
> Agent: "I'll generate this as: *a single glossy red apple centered on a clean white background, soft studio lighting, photorealistic, shallow depth of field, 1024×1024*. Shall I proceed, or would you like a different style/composition?"

Skip both clarification and amplification when the user has clearly authored a full creative brief (≥ 2 of: subject + style + lighting + composition). Respect their prompt verbatim.

**Category-specific briefs** (app mockup, poster, thumbnail, infographic, comic panel, avatar): consult `resources/prompt-tips.md` → *External Prompt Libraries*.

**Output language.** Generation prompts are sent to the provider in English (image models are trained predominantly on English captions). Translate the user's request if they wrote in another language, and show them the translated version during amplification so they can correct misreadings.

### Vendors

This skill follows oh-my-agent's CLI-first concept: whenever a vendor's native CLI can drive generation (and return raw bytes), the subprocess path is preferred over direct API keys. Direct API is only used as a fallback for vendors whose CLI can't yet emit raw image bytes.

| Vendor | Strategy | Models | Trigger |
|--------|----------|--------|---------|
| `codex` | CLI-first via `codex exec` over ChatGPT OAuth (`codex login`), built-in `image_gen` | `gpt-image-2` | Logged in via Codex CLI (no API key) |
| `pollinations` | Direct HTTP via `gen.pollinations.ai/v1/images/generations` (free signup for key) | Free: `flux`, `zimage`. Credit-gated: `qwen-image`, `wan-image`, `gpt-image-2`, `klein`, `kontext`, `gptimage`, `gptimage-large` | `POLLINATIONS_API_KEY` set (free at https://enter.pollinations.ai). No native CLI exists. |
| `antigravity` | `agy -p --dangerously-skip-permissions --add-dir <outDir>` — Antigravity's agentic CLI runs over the user's Gemini Code Assist subscription. agy writes raw bytes to absolute target paths we embed in the prompt; the provider sniffs format via magic bytes and renames the file extension to match. Model selection is opaque — agy picks internally, we never name a model. | (opaque — chosen by agy) | `agy` CLI installed + signed in. No API key, no per-image charge. |

> The direct Gemini path (`gemini -p` stream, `generativelanguage.googleapis.com` API) is deprecated. `agy` is the supported Gemini image route — it's free with Gemini Code Assist and doesn't require billing on AI Studio.

### Invocation

#### Standalone

```
/oma-image a red apple on white background
/oma-image --vendor all --size 1536x1024 jeju coastline at sunset
/oma-image -n 3 --quality high --out ./hero "minimalist dashboard hero illustration"
```

#### Shell CLI

```
oma image generate "<prompt>" [--vendor auto|codex|pollinations|antigravity|all] [-n 1..5] \
                             [--size WxH|auto] \
                             [--quality low|medium|high|auto] \
                             [--out <dir>] [--allow-external-out] \
                             [-r <path>]... \
                             [--timeout 180] [-y] [--no-prompt-in-manifest] \
                             [--dry-run] [--format text|json]
oma image doctor
oma image list-vendors
```

#### Reference Images (`-r`, `--reference`)

Attach up to 10 reference images (PNG/JPEG/GIF/WebP, ≤ 5MB each) to guide style, subject identity, or composition. Repeatable or comma-separated.

```
oma image generate -r ~/Downloads/otter.jpeg "same otter in dramatic lighting"
oma image generate -r a.png -r b.png "blend these two styles"
```

Supported vendors:

| Vendor | Support | How |
|--------|---------|-----|
| `codex` (gpt-image-2) | PASS | Passes `-i <path>` to `codex exec` |
| `antigravity` | PASS | Refs copied to a per-run temp dir, `agy --add-dir <tmpdir>` grants access, paths inlined into the prompt |
| `pollinations` | N/A | Rejected with exit code 4 (requires URL hosting; see PR #2 roadmap) |

**Paths**: absolute or relative to `$CWD`. Host CLIs usually expose attached images via:
- **Claude Code**: `~/.claude/image-cache/<session>/N.png` (surfaced in system messages as `[Image: source: <path>]`)
- **Antigravity**: workspace upload directory (exact path shown in IDE)
- **Codex CLI as host**: user must pass the filesystem path explicitly; in-conversation attachments are not forwarded

#### Agent Behavior: Auto-forward Attached References (MANDATORY)

When ALL of the following are true, the calling agent MUST pass the attached image via `--reference <path>` automatically. Never describe the image in prose as a workaround.

1. The user asks to generate or edit an image (referencing the attached one by phrases like "이거", "this image", "same style as this", "이 수달", etc.).
2. A host-surfaced attached image is visible to the agent (e.g. a Claude Code system message with `[Image: source: <path>]`, or an Antigravity workspace upload path, or an explicit filesystem path in the user's message).
3. The selected vendor supports references (`codex` or `antigravity`).

**Required action**: invoke `oma image generate --reference <absolute-path> --vendor <codex|antigravity> "<prompt>"`. If the user didn't specify a vendor, default to `codex` (CLI-first, widest availability). Do NOT:

- Fall back to prose description ("I'll describe the otter's appearance...").
- Ask the user to re-type or re-attach the path.
- Claim the CLI doesn't support references without first running `oma image generate --help` to verify.

**If the local CLI is outdated** (`--reference` is missing from `--help`): tell the user to run `oma update` once, then retry. Do not silently degrade to prose.

**If the reference path is from Claude Code's `image-cache`**: note to the user that the path is session-scoped and suggest copying the file to a durable location if they want to reuse it later. Still proceed with the generation.

#### Shared Infrastructure (from other skills)

Other skills call `oma image generate --format json` and parse the JSON manifest from stdout.

### Output Layout

```
.agents/results/images/
├── 20260424-143052-ab12cd/                    # single-vendor run
│   └── pollinations-flux.jpg
│       (or codex-gpt-image-2.png)
│       manifest.json
└── 20260424-143122-7z9kqw-compare/            # --vendor all run
    ├── codex-gpt-image-2.png
    ├── pollinations-flux.jpg
    └── manifest.json
```

## References

Follow `resources/execution-protocol.md` step by step.
See `resources/vendor-matrix.md` for strategy precheck rules.
Use `resources/prompt-tips.md` for writing effective prompts.
Before submitting, run `resources/checklist.md`.

### Configuration

Project-specific settings: `config/image-config.yaml`.
Env vars: `OMA_IMAGE_DEFAULT_VENDOR`, `OMA_IMAGE_DEFAULT_OUT`, `OMA_IMAGE_YES`, `POLLINATIONS_API_KEY`.

- Execution steps: `resources/execution-protocol.md`
- Vendor matrix: `resources/vendor-matrix.md`
- Prompt tips: `resources/prompt-tips.md`
- Checklist: `resources/checklist.md`
- Context loading: `../_shared/core/context-loading.md`

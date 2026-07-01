# Vendor Matrix

## Reference Image Support (`--reference` / `-r`)

| Vendor | Reference input | Transport | Notes |
|--------|-----------------|-----------|-------|
| `codex` | PASS | `codex exec -i <path>` (repeatable) | Local file path; 5MB-per-file cap enforced by Codex CLI |
| `antigravity` | PASS | Per-run temp dir staged via `agy --add-dir`; paths referenced inline in prompt | Up to 10 refs (skill cap); copied so agy sandboxes don't need access to the originals' parent dir |
| `pollinations` | N/A | (none) | Requires URL hosting; rejected with exit 4. Planned for PR #2. |

All paths are validated in `reference-guard.ts` (magic-byte MIME check + size + count + duplicate rejection) before dispatch. The magic-byte-detected MIME is threaded through `GenerateInput.referenceImages` and used verbatim at the vendor API boundary; file extension is never trusted for MIME type.

## Codex

| Field | Value |
|-------|-------|
| Binary | `codex` (npm: `@openai/codex`) |
| Auth | OAuth via `codex login` |
| Health check | `codex login status` output contains "Logged in" |
| Model | `gpt-image-2` |
| Transport | `codex exec "<instruction>"` (internal bridge invokes `image_gen` tool) |
| Image location | `~/.codex/generated_images/<session>/ig_*.png` → copied to `outDir` |
| Sizes | Any `WxH` passing `size-guard.ts` (each edge ∈ [16, 3840], multiples of 16, aspect 1:3..3:1) or `auto`. Codex CLI clamps to its own gpt-image-2 limits internally. |
| Qualities | `low`, `medium`, `high`, `auto` |

Codex requires `--skip-git-repo-check` for invocation inside a git worktree; this is inherited from the upstream `codex-image` skill and is a known dependency of the Codex CLI image path.

## Antigravity

The Antigravity CLI (`agy`) is an agentic CLI that runs against the user's Gemini Code Assist subscription (no separate API key, no per-image charge). It exposes an internal image generation tool that drives Gemini-family image models — including the one currently called "nano-banana" — but does **not** expose a model selector to callers. We deliberately do not pretend to choose: the prompt has no model hint, the manifest records `"model": "agy-internal"`, and the output filename is `antigravity-<runShortid>.<ext>` with no model segment.

| Field | Value |
|-------|-------|
| Binary | `agy` (Antigravity CLI) |
| Auth | Sign-in via Gemini Code Assist account during `agy install` |
| Health check | `agy --version` exits 0 |
| Model selection | Opaque — chosen by agy's internal agent loop. Not exposed via flags, not recorded as a vendor-side promise. |
| Transport | `agy -p --dangerously-skip-permissions --add-dir <outDir> --print-timeout <s> "<instruction>"` (spawn `cwd` is forced to `<outDir>` to prevent agy from inheriting a stale workspace context) |
| Output bytes | Saved directly by agy to absolute paths embedded in the prompt; detected via PNG/JPEG/WebP/GIF magic bytes and renamed accordingly |
| Sizes | Any `WxH` allowed by `size-guard.ts` (each edge ∈ [16, 3840], multiples of 16, aspect 1:3..3:1) or `auto`. The dimension is passed as an advisory hint in the prompt — agy/Gemini may pick its own internal aspect. |

### Why `agy` instead of the Gemini CLI / direct API

- `gemini -p` runs the full agent loop and does not emit raw `inlineData` bytes on stdout (as of Gemini CLI 0.38) — it tries to invoke image-generation tools itself, often recursing back into `oma-image`.
- The direct `generativelanguage.googleapis.com` API requires `GEMINI_API_KEY` plus billing on AI Studio. Image models are not in the free tier.
- `agy -p` already wraps Gemini Code Assist credentials in the user's Antigravity session and exposes a working `image_gen` tool. It writes raw bytes to disk for us, so we don't have to capture them from stdout.

### Output format gotcha

Gemini image surfaces currently return JPEG bytes regardless of the requested filename extension. The provider writes to a `.img` placeholder, sniffs the actual format from the first 12 bytes, then renames to `.png` / `.jpg` / `.webp` / `.gif`. The result's `mime` field reflects the sniffed format, not the user's requested extension.

## Strategy Attempt Record

Manifest field `strategy_attempts` is always an array of objects:

```
{ "strategy": "codex-exec-oauth" | "agy-print" | "pollinations-http",
  "status": "ok" | "skipped" | "failed",
  "reason"?: string,
  "duration_ms"?: number }
```

The last successful strategy also appears on the run as `strategy`.

## Error Classification

| Error kind | Retry policy | Exit code when solo |
|------------|-------------|---------------------|
| `not-installed` | fail (the vendor health check catches this first) | 5 |
| `auth-required` | fail; printed hint tells user how to authenticate | 5 |
| `invalid-input` | fail; surfaces validation problems from the provider | 4 |
| `safety-refused` | short-circuit (no fallback) | 2 |
| `rate-limit` | record attempt as failed | 1 (if no vendor succeeded) |
| `timeout` | record attempt as failed | 6 |
| `network` | `retryable=true` → record; `false` → record | 1 |
| `other` | record | 1 |

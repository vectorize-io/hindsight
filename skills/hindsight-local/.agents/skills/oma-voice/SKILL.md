---
name: oma-voice
description: >
  Local-first text-to-speech and speech-to-text via the Voicebox MCP server.
  Generates speech from cloned or preset voice profiles for agent notifications,
  content voiceovers, and audio asset creation, and transcribes audio files for
  meeting notes or memos. Runs entirely on-device with no cloud, no API keys,
  no per-call cost. Use for voice generation, TTS, STT, transcription,
  voiceover, narration, dictation, audio asset work.
---

# Voice Skill - Local TTS and STT via Voicebox

## Scheduling

### Goal
Drive the Voicebox local app through its MCP server so any MCP-aware agent can speak (TTS) or listen (STT) without invoking cloud vendors. The skill standardizes intent routing, voice profile resolution, output layout, and guardrails while voicebox itself owns the engines, voice cloning UI, captures archive, and stories editor.

### Intent signature
- User asks to generate speech, narrate text, produce a voiceover, create an mp3 or wav from text.
- User wants an audio file transcribed into text, meeting notes, or a transcript.
- User asks for a voice notification when a long task completes or a workflow step is blocked.
- Another skill needs local audio generation infrastructure.

### When to use
- Generating short notification audio for agent task completion or blockers.
- Producing voiceover, narration, or audio assets (mp3 or wav) for apps and content.
- Transcribing local audio files (mp3, wav, m4a, webm, flac) to Markdown.
- Comparing voice profiles by re-running the same text against different profile ids.

### When NOT to use
- Cloud TTS or high-fidelity multilingual cloud voices -> out of scope; future multi-vendor extension.
- Real-time microphone dictation loop in the terminal -> use Voicebox app's built-in hotkey dictation.
- Voice cloning sample upload and profile creation -> done in the Voicebox desktop app UI.
- Video synthesis, music, sound design -> out of scope.
- Stories Editor multi-voice timeline composition -> use the Voicebox app UI.

### Expected inputs
- TTS: text (<= 5000 chars per call), optional profile id, optional engine, optional language, optional output path.
- STT: audio file path (absolute or relative to `$CWD`), optional language hint.
- Notification: short message (<= 240 chars), profile id resolved from config.

### Expected outputs
- TTS: audio file (`mp3` default, `wav` optional) at `.agents/results/voice/{timestamp}-{shortid}/output.{mp3|wav}` plus `manifest.json`.
- STT: `transcript.md` at `.agents/results/voice/transcripts/{timestamp}-{shortid}/` plus `manifest.json`.
- Notification: ephemeral playback through Voicebox; no disk write by default.

### Dependencies
- Voicebox desktop app installed and running locally.
- Voicebox MCP registered (`claude mcp add --transport http voicebox http://127.0.0.1:17493/mcp`).
- At least one voice profile created in the Voicebox app UI.
- Optionally pre-downloaded engine models for the selected profile.

### Control-flow features
- Branches by mode (notify, asset, transcribe), language, and profile availability.
- Calls voicebox via MCP tools, with REST `GET /health` as the handshake probe.
- Reads input audio files and writes generated audio plus manifests.
- Caches discovered MCP tool names after the first successful `tools/list`.

## Structural Flow

### Entry
1. Detect the requested mode: notification, asset TTS, or transcription.
2. Verify Voicebox is reachable via MCP handshake or `GET /health`.
3. On the first run only, call MCP `tools/list` and cache the resolved tool names.
4. Resolve the target voice profile id (notification, asset, or explicit user choice).

### Scenes
1. **PREPARE**: Validate text length, audio duration, language, output path, and profile id.
2. **ACQUIRE**: If a required signal is missing, run the clarification protocol once.
3. **ACT**: Invoke the appropriate MCP tool (TTS or STT) with the resolved parameters.
4. **VERIFY**: Confirm the response carries audio output or transcript content. Validate manifest fields.
5. **FINALIZE**: Write `manifest.json` alongside the output. Report the path or transcript to the user.

### Transitions
- If voicebox is unreachable, surface the install or launch hint and exit. Do not attempt auto-relaunch.
- If `voicebox_list_profiles` is empty, point the user at the Voicebox app UI to create a profile, then exit.
- If a TTS request exceeds 5000 chars, ask whether to truncate or split. Do not auto-chunk in v1.
- If an STT input exceeds 30 minutes, ask whether to proceed. Do not auto-split.
- If the selected engine model is not loaded, ask the user before triggering a download.

### Failure and recovery
| Failure | Recovery |
|---------|----------|
| Voicebox app not running | Print install/launch hint, exit code 5 |
| No voice profile | Print "create a profile in Voicebox" hint, exit code 3 |
| Engine model missing | Ask before triggering download |
| Output path outside `$PWD` | Warn the user, require explicit confirmation |
| TTS over 5000 chars | Ask the user to split or truncate |
| STT over 30 minutes | Ask the user to confirm |
| MCP tool name drift | Re-run `tools/list` and update the cache |
| SIGINT | Abort the MCP call, write no partial output |

### Exit
- Success: audio file or transcript exists with a complete manifest, and the path is reported.
- Partial success: output exists but a guardrail warning is surfaced (length, disk, model fallback).
- Failure: no output, the blocker (auth, profile, engine, network) is explicit.

## Logical Operations

### Actions
| Action | SSL primitive | Evidence |
|--------|---------------|----------|
| Validate mode and inputs | `VALIDATE` | Clarification protocol in execution-protocol.md |
| Resolve voice profile | `SELECT` | `voicebox_list_profiles` + config defaults |
| Health check | `READ` | MCP handshake or `GET /health` |
| Generate speech | `CALL_TOOL` | MCP `voicebox_speak` |
| Transcribe audio | `CALL_TOOL` | MCP `voicebox_transcribe` |
| Write output and manifest | `WRITE` | Audio or transcript plus `manifest.json` |
| Inspect result | `VALIDATE` | Output presence, duration, manifest fields |
| Report result | `NOTIFY` | Final user-facing summary |

### Tools and instruments
- Voicebox MCP server at `http://127.0.0.1:17493/mcp`.
- REST surface for health and audio retrieval (`GET /health`, `GET /audio/{generation_id}`).
- Resource references: voice matrix, prompt tips, execution protocol, checklist.

### Canonical command path
```text
# 1. MCP handshake or REST health
GET http://127.0.0.1:17493/health  ->  200 OK

# 2. Discover tool names on first run
MCP tools/list                      ->  cache real names

# 3. Resolve profile
MCP voicebox_list_profiles          ->  pick profile by name or config default

# 4. Generate or transcribe
MCP voicebox_speak     { text, profile, language?, engine?, personality? }
MCP voicebox_transcribe { audio_path | audio_base64, language?, model? }

# 5. Persist output + manifest
.agents/results/voice/<timestamp>-<shortid>/output.mp3 + manifest.json
.agents/results/voice/transcripts/<timestamp>-<shortid>/transcript.md + manifest.json
```

### MCP tool mapping (verified against Voicebox 0.5.0)

| Use case | MCP tool | REST backing |
|---|---|---|
| TTS generation | `voicebox_speak` | `POST /generate` |
| STT transcription | `voicebox_transcribe` | `POST /transcribe` |
| Profile listing | `voicebox_list_profiles` | `GET /profiles` |
| Captures listing | `voicebox_list_captures` | `GET /history` (captures view) |

Tools not exposed via MCP (REST only): model status (`GET /models/status`), audio file serving (`GET /audio/{generation_id}`). The skill calls those over loopback HTTP when needed.

**Notes on `voicebox_speak`:**
- Required: `text`. Optional: `profile`, `engine`, `language`, `personality` (bool).
- Audio plays on the user speakers and is saved to the Captures / History panel automatically. There is no `save_to_disk` toggle on the MCP tool itself.
- Without a default profile set in Voicebox Settings, `profile=` is required.

**Notes on `voicebox_transcribe`:**
- Accepts exactly one of `audio_base64` or `audio_path` (loopback only). Optional `language`, `model`.

### Resource scope
| Scope | Resource target |
|-------|-----------------|
| `LOCAL_FS` | Input audio, generated audio, transcripts, manifests |
| `PROCESS` | Local Voicebox app subprocess (managed by the user) |
| `NETWORK` | Loopback HTTP to `127.0.0.1:17493` only |
| `MEMORY` | Cached MCP tool names, resolved profile metadata |
| `CREDENTIALS` | None. Voicebox is local and key-free. |

### Preconditions
- Voicebox app is running and the MCP handshake succeeds.
- At least one voice profile exists.
- The selected engine model is loaded or the user approves a download.
- Output directory is inside `$PWD` unless explicitly allowed.

### Effects and side effects
- Creates audio files, transcripts, and manifests under `.agents/results/voice/`.
- Triggers local Voicebox generation, which consumes CPU or GPU.
- May trigger an engine model download when the user approves.
- Does not call any cloud service. No external network traffic.

### Guardrails

1. **Voicebox required**: if the MCP handshake or `GET /health` fails, exit with a one-shot install or launch hint. Do not retry, do not auto-relaunch.
2. **Profile required**: if `voicebox_list_profiles` returns empty, instruct the user to create a profile in the Voicebox app (Profiles tab → + New Profile → pick Kokoro preset for the fastest path), then exit.
3. **Tool-name discovery**: on first invocation, call MCP `tools/list` and cache the resolved names. Reuse the cache for subsequent calls in the same session.
4. **Length limits**: TTS calls cap at 5000 chars per call; warn at 2000. STT inputs cap at 30 minutes. v1 does not auto-chunk or auto-split.
5. **Auto-invocation transparency**: notifications fire automatically only when the active task exceeds `auto_notify_after_sec` (default 60s). Always announce intent in one short line before generating audio.
6. **Path safety**: when the user requests an output path outside `$PWD`, warn once and require explicit confirmation.
7. **Cancellation**: SIGINT aborts the MCP call and writes no partial output.
8. **Manifest required**: every generation writes `manifest.json` with at minimum: `skill`, `mode`, `voicebox_generation_id`, `text` (or `transcript_preview`), `profile`, `engine`, `language`, `format` (TTS only), `created_at`.
9. **Out of scope**: voice cloning UI, captures archive, stories editor, microphone dictation loop, and cloud vendors are intentionally not exposed.
10. **No cost guard**: Voicebox is free. The cost guardrail from `oma-image` does not apply.

### Clarification protocol

Before invoking a TTS or STT call, the agent checks the following. If any required signal is missing, clarify with the user first.

**TTS (asset mode) required:**
- [ ] Text content provided?
- [ ] Voice profile id or tone description provided?

**TTS strongly recommended:**
- [ ] Language explicit or detectable from the text?
- [ ] Output format (mp3 default, wav optional)?

**STT required:**
- [ ] Audio path provided and the file exists?
- [ ] Duration within 30 minutes, or user approves splitting?

**Notification mode skips clarification.** It uses `notification_profile` from config and language is auto-detected from the message.

### Invocation

#### Standalone
```text
/oma-voice "build succeeded, 4 minor warnings"
/oma-voice transcribe ~/Downloads/standup.m4a
/oma-voice --profile prof_warm_korean "다음 단계 진행 준비됐어요"
```

#### Shared infrastructure (other skills)

Other skills can request audio output by calling the same MCP tools directly, or by invoking `/oma-voice` with their text. There is no separate CLI; the skill is MCP-native.

## References

- Voice engine matrix: `resources/voice-matrix.md`
- Prompt writing rules: `resources/prompt-tips.md`
- Execution protocol: `resources/execution-protocol.md`
- Pre-flight checklist: `resources/checklist.md`
- Configuration: `config/voice-config.yaml`
- Context loading: `../_shared/core/context-loading.md`
- Quality principles: `../_shared/core/quality-principles.md`
- Design reference: `../../../docs/plans/designs/012-oma-voice.md` (source repo only; absent in global-mode installs)

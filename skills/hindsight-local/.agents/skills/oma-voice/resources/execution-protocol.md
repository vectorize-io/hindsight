# Execution Protocol

This file lays out the runtime sequence for every oma-voice call. SKILL.md gives the high-level scenes; this file is the operational checklist the agent executes.

## Step 0: Mode detection

Detect one of three modes from the user request:

| Signal | Mode |
|---|---|
| Short message after a long task, "tell me when done", "음성으로 알려줘" | `notify` |
| "Generate a voiceover", "make an mp3", "narrate this text", explicit length over a sentence | `asset` |
| Audio file path provided plus a "transcribe", "회의록", "받아 적어" intent | `transcribe` |

If the signal is ambiguous, ask one question before proceeding.

## Step 1: Health check

1. Probe `GET http://127.0.0.1:17493/health` with a 5-second timeout (configurable).
2. On success, continue to Step 2.
3. On failure, surface this hint and exit with code 5:
   ```
   Voicebox is not reachable. Install or launch the Voicebox desktop app
   from GitHub Releases (https://github.com/jamiepine/voicebox/releases).
   Voicebox is not in Homebrew.
     - Apple Silicon: download Voicebox_<ver>_aarch64.dmg
     - Intel Mac:     download Voicebox_<ver>_x64.dmg
     - Windows:       Voicebox_<ver>_x64-setup.exe
   Then register the MCP server:
     claude mcp add --transport http voicebox http://127.0.0.1:17493/mcp
   ```

## Step 2: Tool discovery cache

Verified Voicebox MCP tool names (as of v0.5.0):

```
tts        -> voicebox_speak
stt        -> voicebox_transcribe
profiles   -> voicebox_list_profiles
captures   -> voicebox_list_captures
```

Model status and audio retrieval are REST-only at the Voicebox app, not exposed as MCP tools. Call them over loopback HTTP when needed:
- `GET http://127.0.0.1:17493/models/status`
- `GET http://127.0.0.1:17493/audio/{generation_id}`

On the first call in this session:

1. Invoke MCP `tools/list` on the `voicebox` server.
2. Confirm all four tool names above are present.
3. If a required tool is missing, exit with code 4 and surface the gap. Voicebox may have renamed a tool in a newer release.

## Step 3: Profile resolution

1. Call the cached `profiles` tool.
2. If the list is empty, surface this hint and exit with code 3:
   ```
   No voice profile found in Voicebox.
   Create one in the app UI (cloning or preset), then retry.
   ```
3. Resolve the target profile in this order:
   1. Explicit `profile` argument from the user.
   2. `notification_profile` from `voice-config.yaml` when mode is `notify`.
   3. `asset_profile` from `voice-config.yaml` when mode is `asset` and the user did not specify one.
   4. Else ask the user which profile to use.

## Step 4: Clarification (asset and transcribe modes only)

Run the checklist from `SKILL.md > Clarification protocol`. Notification mode skips this step.

## Step 5: Length and duration guard

| Mode | Check | Action |
|---|---|---|
| `notify` | `len(text) <= 240` | Truncate with ellipsis; warn user |
| `asset` | `len(text) <= 5000` | Ask user to split or truncate if exceeded |
| `transcribe` | audio duration <= 30 min | Ask user to confirm if exceeded |

## Step 6: Model availability (optional)

For `asset` and `notify` modes:

1. Call the cached `models` tool.
2. If the selected engine reports `loaded: false`, ask the user before triggering a download. Voicebox owns the download flow; this skill only relays the prompt.

## Step 7: Invoke MCP tool

### TTS path
```text
MCP call: voicebox_speak
  text:        <normalized text, required>
  profile:     <resolved profile name from voicebox_list_profiles>
  language:    <auto or explicit; optional>
  engine:      <optional engine override>
  personality: <optional bool>
```
Voicebox plays the audio on the user speakers and saves a capture entry automatically. The tool returns a generation id; the audio file lives inside the Voicebox app's Captures panel and is also reachable over loopback at `GET /audio/{generation_id}` when a local copy is needed.

### STT path
```text
MCP call: voicebox_transcribe
  audio_path:    <absolute local path; loopback only>
  # OR
  audio_base64:  <base64-encoded audio bytes>
  language:      <auto or explicit; optional>
  model:         <optional model override>
```
Expected response: transcript text and detected language. Pass exactly one of `audio_path` or `audio_base64`.

## Step 8: Persist output and manifest

### TTS output
```text
.agents/results/voice/<YYYYMMDD-HHMMSS>-<shortid>/
├── output.mp3
└── manifest.json
```

### STT output
```text
.agents/results/voice/transcripts/<YYYYMMDD-HHMMSS>-<shortid>/
├── transcript.md
└── manifest.json
```

### Manifest shape

TTS:
```json
{
  "skill": "oma-voice",
  "mode": "tts",
  "voicebox_generation_id": "gen_abc123",
  "text": "...",
  "profile": "Nova",
  "engine": "kokoro",
  "language": "en",
  "format": "mp3",
  "duration_sec": 7.4,
  "created_at": "2026-05-15T09:15:33+09:00"
}
```

STT:
```json
{
  "skill": "oma-voice",
  "mode": "stt",
  "voicebox_generation_id": "gen_def456",
  "transcript_preview": "first 200 chars...",
  "source_path": "/abs/path/to/input.mp3",
  "profile": null,
  "engine": "whisper",
  "language": "ko",
  "duration_sec": 312.0,
  "created_at": "2026-05-15T09:18:20+09:00"
}
```

For notification mode no manifest is required: Voicebox auto-saves the clip to its Captures panel and that is the system of record.

## Step 9: Report

Report to the user in one short message:

- Mode (notify / tts / stt)
- Output path (or "played" for ephemeral notifications)
- Duration in seconds
- Any warnings (length truncated, disk warn fired, model fallback used)

## Error code mapping

Mirror `oma-image` exit codes:

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Generic failure |
| 2 | Safety block (currently unused; reserved) |
| 3 | Resource not found (no profile, missing audio file) |
| 4 | Invalid input (length, format, MCP tool missing) |
| 5 | Auth or environment required (voicebox not reachable) |
| 6 | Timeout |

## Cancellation

On SIGINT or SIGTERM:

1. Send an abort signal to the in-flight MCP call when supported.
2. Do not write partial output.
3. Exit with the originating signal's conventional code.

## Idempotency

- Two identical TTS calls produce two separate generations and two separate output folders. The skill does not de-duplicate.
- Voicebox's own History panel tracks all generations; users can prune it from the app UI.

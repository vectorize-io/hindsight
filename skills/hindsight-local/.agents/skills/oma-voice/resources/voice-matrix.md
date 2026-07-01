# Voice Engine Matrix

Voicebox bundles seven engines. All run locally; the user picks one when creating a voice profile in the Voicebox app, and the skill respects that choice. This matrix exists so the agent can recommend a profile-engine pair when the user has not yet decided.

## Engine comparison

| Engine | Languages | Strengths | Best for | Weight | Notes |
|---|---|---|---|---|---|
| Qwen3-TTS | 10, strong CJK | Natural CJK prosody, fast on consumer GPU | Korean, Japanese, Chinese narration | Medium | First choice for mixed CJK content |
| Qwen CustomVoice | 9 presets | Quick start, no sample needed | Demo and prototyping in CJK | Medium | Limited tonal variety |
| LuxTTS | English only | Lightweight, CPU-friendly | Low-resource machines, quick notifications | Light | Less expressive than Kokoro |
| Chatterbox Multilingual | 23 | Broad coverage, decent quality | Multilingual content, mixed sentences | Medium-Heavy | Slower than Kokoro on CPU |
| Chatterbox Turbo | English | Emotion tags, expressive delivery | Character voices, podcast intros | Medium | Best with voice cloning |
| TADA (HumeAI) | 10 | Strong emotional range, narration cadence | Audiobook, story narration | Heavy | Largest models, GPU recommended |
| Kokoro | 8, 50 preset voices | Polished English, dependable defaults | Notifications and short English assets | Light-Medium | Default English fallback |

## Picks by use case

| Use case | First choice | Second choice |
|---|---|---|
| Korean notification | Qwen3-TTS | Chatterbox Multilingual |
| English notification | Kokoro | LuxTTS |
| Japanese or Chinese narration | Qwen3-TTS | Chatterbox Multilingual |
| Mixed-language sentence | Chatterbox Multilingual | Qwen3-TTS |
| CPU-only laptop, short clips | LuxTTS | Kokoro |
| Expressive narration | TADA | Chatterbox Turbo |
| Character voice or cloned voice | Chatterbox Turbo + cloning | Qwen CustomVoice |
| Quick voiceover, English | Kokoro | Chatterbox Turbo |

## Language-first rules

- **Korean text**: prefer Qwen3-TTS. If Korean mixes English technical terms, stay on Qwen3-TTS rather than switching engines mid-sentence.
- **English text**: Kokoro for clean defaults; Chatterbox Turbo when emotion tags or character voices matter.
- **Mixed sentences**: pick a multilingual engine and keep the call as a single request. Splitting hurts prosody.
- **Bottom-tier hardware**: LuxTTS or Kokoro stay snappy without a GPU.
- **Long-form narration (over 5 minutes total in chunks)**: prefer TADA or Chatterbox Turbo for sustained expression; reserve Kokoro for short bursts.

## Profile selection heuristic

1. If the user passes an explicit `profile` argument, use it.
2. Else use `notification_profile` (notify mode) or `asset_profile` (asset mode) from `voice-config.yaml`.
3. Else use the first available profile from `voicebox_list_profiles` whose engine matches the detected language pick in the table above.
4. Else exit with a setup hint pointing the user to the Voicebox app Profiles tab.

## Caveats

- Engine names above are the ones Voicebox surfaces in its UI. If voicebox renames an engine, the skill defers to whatever `get_model_status` returns.
- Voice cloning quality depends on the reference sample voicebox captured. The skill does not judge sample quality; it routes by language only.
- Some engines auto-download large model weights on first use. Trigger the download from the Voicebox app UI to keep the agent loop responsive.

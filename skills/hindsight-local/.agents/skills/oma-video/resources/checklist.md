# Checklist: before you run `oma video generate`

- [ ] Mode is set or inferable (shorts / explainer / demo) and the topic/source is clear (see `prompt-tips.md`).
- [ ] `--aspect` matches the mode (9:16 shorts, 16:9 explainer/demo) or is `auto`.
- [ ] Duration ≤ 180s and the script stays ≤ 40 scenes.
- [ ] `--out` is inside the project, or `--allow-external-out` is set.
- [ ] For `demo`: `--capture <path>` exists, is absolute + inside `$PWD`, and is a valid video format — or you accept the guided protocol.
- [ ] Provider readiness checked with `oma video doctor` (Node/Chromium/FFmpeg · Voicebox MCP · oma-image vendors · Pixelle-MCP · Cap).
- [ ] Paid visuals (Pexels / Pixelle) have their env key set, OR you accept the key-free oma-image fallback.
- [ ] Pixelle-MCP, if used: one-time explicit consent + source review done; `--max-usd` set for RunningHub credits.
- [ ] Estimated cost is acceptable. Run `--dry-run` first for unfamiliar combinations.
- [ ] Secrets are not in the brief, or `--no-brief-in-manifest` is set.

# Checklist: after the run

- [ ] The run directory contains `script.json`, `timing.json`, `render-spec.json`, `captions.srt`/`.vtt`, the `<mode>-<slug>.mp4`, and `manifest.json`.
- [ ] Every asset-bus schema validates (`schemaVersion: "1.0"`).
- [ ] `manifest.json` records each provider, asset `sha256` hashes, cost breakdown, and the exit code.
- [ ] `warnings[]` annotates any fallback taken (e.g. Pexels key absent -> oma-image stills, translator absent -> source locale).
- [ ] External assets were copied into the run dir and hashed (no URL refs).
- [ ] The mp4 plays (or, on the toolchain-free path, the deterministic placeholder is present and reproducible).
- [ ] Re-rendering with `oma video render <runDir>` reproduces the same output from `render-spec.json`.
- [ ] If results are consumed downstream, the consumer parses `--format json` stdout rather than re-reading the manifest file.

# Checklist: before you run `oma image generate`

- [ ] Prompt is specific about scene, subject, and style (see `prompt-tips.md`).
- [ ] `--vendor` matches available authenticated CLIs. Run `oma image doctor` if unsure.
- [ ] `-n` is ≤ 5; wall time scales with count.
- [ ] `--out` is inside the project, or you've set `--allow-external-out`.
- [ ] Estimated cost is acceptable. Run `--dry-run` first for unfamiliar combinations.
- [ ] Secrets are not in the prompt, or `--no-prompt-in-manifest` is set.
- [ ] For `--vendor all`, every enabled vendor is healthy (strict mode exits 5 otherwise).

# Checklist: after the run

- [ ] `manifest.json` was written inside the run folder.
- [ ] Each recorded run has an `ok` status or a classified error.
- [ ] Strategy attempts are objects (not compact strings).
- [ ] Images open without corruption.
- [ ] If results are consumed downstream, the consumer parses `--format json` stdout rather than re-reading the manifest file.

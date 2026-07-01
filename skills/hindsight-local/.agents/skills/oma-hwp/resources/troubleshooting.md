# oma-hwp Troubleshooting

## Diagnosing conversion issues

### 1. "지원하지 않는 파일 형식" / unsupported file format
- Verify the extension matches the content. HWP5 files should start with `D0 CF 11 E0` (OLE compound). HWPX starts with `50 4B 03 04` (ZIP).
- Check with: `xxd "{file}" | head -1`
- If the extension is wrong, rename and retry.
- If the file is genuinely a different format (e.g., Word, Excel), use the appropriate skill.

### 2. Encrypted or password-protected HWP
- kordoc reports `암호화된 문서` / "encrypted document".
- General password-locked HWP: the CLI currently has no inline password flag. Ask the user to decrypt the file first (via Hancom Office) and retry.
- DRM "distribution-only" HWP (배포용): kordoc ports rhwp's AES-128 ECB algorithm and often succeeds. If it still fails, note as best-effort and continue with other files.

### 3. Empty Markdown output
Most likely causes:
- HWP contains only scanned images without a text layer → needs OCR (out of scope).
- kordoc bug on a specific document → try `bunx kordoc@latest`; if still empty, report upstream with a sample.

### 4. Missing or broken tables
- Simple tables: GFM pipe syntax.
- Tables with `colspan` / `rowspan`: kordoc falls back to HTML `<table>`. This is expected and correct.
- Nested tables: large nested tables become separate blocks; small ones get flattened into the parent cell.
- If tables are completely absent, confirm the source file actually contains tables (sometimes they are images).

### 5. Hyperlinks look wrong
kordoc sanitizes links (XSS defense). If a legitimate link is stripped, verify the original URL scheme: only `http://`, `https://`, `mailto:`, and relative paths are preserved by default.

### 6. Image extraction issues
- kordoc extracts images from ZIP entries (HWPX) and HWP5 `BinData` streams.
- Output is typically written as `{input_basename}.assets/` next to the Markdown file.
- If images are missing, check whether the source file truly embeds them (vs linking externally).

### 7. Slow conversion / timeout on huge files
- Use `-p` to narrow to a page/section range.
- Split processing across page ranges and concatenate results.
- If a specific PDF-embedded HWPX is slow, consider oma-pdf for the PDF portion.

### 8. kordoc crash / stack trace
1. Upgrade: `bunx kordoc@latest` (bust the bunx cache if needed)
2. Capture a minimal reproducible fixture
3. File an issue: https://github.com/chrisryugj/kordoc/issues
4. Meanwhile, if only a subset of the document triggers the crash, use `-p` to skip it

### 9. Reproducibility vs freshness tradeoff
- `bunx kordoc@latest` always pulls latest. Fixes land fast, but outputs may drift. A bare `bunx kordoc` reuses the bunx cache and can be arbitrarily old; always include `@latest` or a pinned version.
- For long-running projects, pin a version: edit `config/hwp-config.yaml`:
  ```yaml
  version:
    channel: pinned
    pinned: "2.4.0"
  ```
  Then invoke via `bunx kordoc@2.4.0 ...` in commands.

## kordoc limitations (upstream-owned)
- Inline password entry for encrypted HWP: not yet supported.
- OCR of scanned HWPs: out of kordoc's scope; it does not perform OCR.
- Perfect fidelity of complex shapes / drawings: Markdown cannot represent drawing primitives; expect them to be dropped or replaced by placeholder markers.

## When to route elsewhere
| Symptom | Route to |
|---------|----------|
| Input is `.pdf` | `oma-pdf` |
| Input is `.xlsx` / `.docx` | `bunx kordoc@latest` directly (skill not advertising) |
| Need OCR of scanned documents | Out of scope; use a dedicated OCR pipeline |
| Need to author / fill HWPX | `bunx kordoc@latest fill ...` directly (skill not advertising) |

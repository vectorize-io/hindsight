# HWP Conversion - Execution Protocol

## Step 0: Validate Input

1. Confirm the input file path exists
2. Check extension is one of `.hwp`, `.hwpx`, `.hwpml`
   - If `.pdf` -> hand off to `oma-pdf`
   - If `.xlsx` / `.docx` -> not in scope; advise user to run `bunx kordoc@latest <file>` directly
3. Check file size (`wc -c` or `ls -lh`); warn if >100MB
4. Determine output location:
   - If user specified output path (`-o` or `-d`) → use it
   - Otherwise → same directory as the input file
5. Determine output filename: `{input_basename}.md`

## Step 1: Check Runtime

```bash
bun --version
```

If `bun` is not available, ask the user to install Bun (https://bun.sh). `bunx` is required.

## Step 2: Convert

> **Important**: Without `-o` or `-d`, kordoc prints the Markdown to **stdout** and does not create a file. Always pass an explicit output path to match this skill's "same directory as input" convention.

### Standard conversion (single file, write next to input)
```bash
bunx kordoc@latest "{input_path}" -o "{output_path}"
# {output_path} = "{dirname(input_path)}/{basename_without_ext(input_path)}.md"
```

### Output directory (multiple files)
```bash
bunx kordoc@latest "{input_pattern}" -d "{output_dir}"
```

### Print to stdout (piping / preview)
```bash
bunx kordoc@latest "{input_path}"
# No file written; useful when piping to another tool
```

### Page / section range
```bash
bunx kordoc@latest "{input_path}" -p "{range}"
```

### JSON output (structured intermediate form)
```bash
bunx kordoc@latest "{input_path}" --format json
```

Notes:
- Default format is `markdown`. Pass `--format json` only when you need the structured AST.
- `--silent` suppresses progress output, useful in automation / piping contexts.

## Step 2.5: Post-process kordoc output (default)

`resources/flatten-tables.ts` cleans up two kordoc artifacts that hurt downstream use:

1. **HTML `<table>` blocks**: kordoc emits these when a table has `colspan` / `rowspan` because GFM cannot represent merged cells. Converted to GFM pipe tables via `turndown-plugin-gfm`.
2. **Private Use Area characters**: HWP references Hancom-font-specific glyphs via U+E000-U+F8FF / U+F0000-U+FFFFD / U+100000-U+10FFFD code points. Without the Hancom font these render as blanks or tofu squares. Silently stripped.

```bash
node "{skill_resources}/flatten-tables.ts" "{output_path}"
```

- `{skill_resources}` = `.agents/skills/oma-hwp/resources`
- Ensure `bun install` has been run once inside that directory (it installs `turndown` + `turndown-plugin-gfm` locally)
- Merged cells get flattened during the conversion (accepted trade-off)
- Skip this step only if the caller explicitly needs HTML tables or PUA characters preserved (rare)

## Step 3: Verify

1. Read the generated Markdown file
2. Verify structure:
   - Headings preserved (`#`, `##`, etc.)
   - Tables rendered (GFM pipe syntax, or HTML `<table>` fallback for merged cells)
   - Lists maintained (bullets, numbered)
   - Images referenced (relative paths or inline)
   - Footnotes linked via `[^id]`
3. If the file is empty or clearly garbled, consult `troubleshooting.md`
4. If output directory is temporary, move the file to the target location
5. If the user needs the content in-conversation, read and present it

## Step 4: Report

Tell the user:
- Output file path
- Source format detected (HWP5 / HWPX / HWPML)
- Any issues encountered (encrypted segments, missing tables, best-effort DRM extraction)
- Page/section count if available

## Error Recovery

| Error | Recovery |
|-------|----------|
| `bunx` / `bun` not found | Ask user to install Bun: `curl -fsSL https://bun.sh/install \| bash` |
| `지원하지 않는 파일 형식` / `unsupported file format` | Confirm extension matches actual content; check magic bytes |
| `암호화된 문서` / `encrypted document` | Ask user for password; current kordoc CLI may not accept it inline (document as limitation) |
| Empty Markdown output | Likely a scanned-image-only HWP; suggest OCR pipeline (out of this skill's scope) |
| Broken / missing tables | Complex merged-cell tables fall back to HTML `<table>`; accept as best-effort |
| kordoc crash or stack trace | Ensure `@latest` is passed (bunx cache can be stale); if persistent, capture fixture and file issue upstream at https://github.com/chrisryugj/kordoc/issues |
| Slow conversion on large files | Use `-p` to narrow the page range; process sections separately |

## Pin vs Latest

- **Default**: `bunx kordoc@latest` (always latest). Gets upstream fixes automatically. `@latest` is required because a bare `bunx kordoc` can reuse a stale cached version indefinitely.
- **Reproducibility**: pin with `bunx kordoc@2.4.0 ...` or similar. Record the pinned version in `config/hwp-config.yaml` under `version.pinned` and set `version.channel: pinned`.

## Scope Reminder

If the user hands you a `.pdf` / `.xlsx` / `.docx` with this skill activated, **do not proceed**. Redirect:

- `.pdf` → switch to `oma-pdf`
- `.xlsx` / `.docx` → ask the user to run `bunx kordoc@latest <file>` directly; this skill does not advertise those formats

kordoc technically supports them, but routing keeps skill scopes clean.

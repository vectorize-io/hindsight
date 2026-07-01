---
name: oma-hwp
description: >
  Convert HWP / HWPX / HWPML files to Markdown using kordoc. Extracts text, headings, tables,
  lists, images, footnotes, and hyperlinks. Use for Korean word processor files (Hangul),
  government documents, and AI-ready data preparation.
---

# HWP Skill - HWP / HWPX / HWPML to Markdown Conversion

## Scheduling

### Goal
Convert Korean HWP-family documents into readable Markdown or structured JSON while preserving document structure for LLM context, RAG, government-document review, or enterprise document processing.

### Intent signature
- User asks to convert, parse, read, extract, or transform `.hwp`, `.hwpx`, or `.hwpml`.
- User mentions Korean word processor files, Hangul documents, government forms, or "한글 파일".
- User needs headings, tables, nested tables, lists, images, footnotes, or hyperlinks extracted from HWP-family files.

### When to use
- Converting Korean HWP documents (`.hwp`, `.hwpx`, `.hwpml`) to Markdown
- Preparing Korean government/enterprise documents for LLM context or RAG
- Extracting structured content (tables, headings, lists, images) from HWP
- User says "convert this HWP", "parse hwpx", "HWP to markdown", "한글 파일"

### When NOT to use
- PDF files -> use `oma-pdf` (OCR + Tagged PDF specialization)
- XLSX / DOCX files -> out of scope; run `bunx kordoc` directly if needed (note: `oma-docs` is the documentation-drift skill, not a converter)
- Generating or editing HWP documents -> out of scope
- Already-text files -> use Read tool directly

### Expected inputs
- `input_path`: `.hwp`, `.hwpx`, or `.hwpml` file path
- `output_path` or `output_dir`: optional explicit output target
- `format`: optional output format, default `markdown`
- `page_range`: optional page or section range
- `kordoc_version`: optional pinned kordoc version

### Expected outputs
- Markdown output next to the input file or in the requested directory
- Optional JSON output when requested
- Post-processed Markdown with flattened GFM tables and stripped Private Use Area glyphs by default
- A short report with output path, detected source format, and conversion issues

### Dependencies
- `bun` and `bunx`
- `bunx kordoc@latest` or configured pinned kordoc version
- `resources/flatten-tables.ts` for Markdown cleanup
- Local filesystem access to input and output paths

### Control-flow features
- Branches by file extension, output target, format, page range, encryption/DRM state, and post-processing requirements
- Calls external CLI tools through `bunx` and `bun run`
- Reads local HWP-family files and writes local Markdown or JSON output
- Routes non-HWP inputs to other skills instead of stretching this skill's scope

## Structural Flow

### Entry
1. Confirm the input path exists.
2. Confirm the extension is `.hwp`, `.hwpx`, or `.hwpml`.
3. Resolve output path or directory and default filename.
4. Check that `bun` is available.

### Scenes
1. **PREPARE**: Validate path, extension, size, output target, and requested format.
2. **ACQUIRE**: Detect source format and runtime availability.
3. **ACT**: Run `kordoc` with explicit output target and requested options.
4. **VERIFY**: Post-process Markdown and inspect structure for headings, tables, lists, images, and footnotes.
5. **FINALIZE**: Report output path, source format, and any conversion limitations.

### Transitions
- If the input is `.pdf`, stop and route to `oma-pdf`.
- If the input is `.xlsx` or `.docx`, explain that this skill does not advertise those formats.
- If `bun` is unavailable, stop and ask the user to install Bun.
- If Markdown is produced, run `resources/flatten-tables.ts` unless the caller explicitly needs HTML tables or PUA glyphs preserved.
- If output is empty or garbled, consult `resources/troubleshooting.md`.

### Failure and recovery
| Failure | Recovery |
|---------|----------|
| `bun` or `bunx` unavailable | Ask user to install Bun |
| Unsupported or mismatched format | Check extension and magic bytes, then route or stop |
| Encrypted or DRM-locked document | Report limitation and request an accessible copy when needed |
| Empty Markdown output | Treat as possible scanned-image content and recommend OCR outside this skill |
| Complex merged tables | Accept flattened Markdown or HTML fallback as best effort |
| Stale kordoc cache | Use `bunx kordoc@latest` or configured pinned version |
| `Cannot find module "turndown"` from `flatten-tables.ts` | Run `bun install` in `.agents/skills/oma-hwp/resources/` (its `node_modules` is gitignored and absent on fresh clones) |

### Exit
- Success: output file exists and structure is readable after post-processing.
- Partial success: output exists with explicitly reported table, glyph, encryption, or fidelity limitations.
- Failure: no reliable output is produced and the blocking cause is reported.

## Logical Operations

### Actions
| Action | SSL primitive | Evidence |
|--------|---------------|----------|
| Validate file path and extension | `VALIDATE` | Input preflight in execution protocol |
| Check runtime availability | `VALIDATE` | `bun --version` |
| Select output target and format | `SELECT` | Output behavior and config |
| Run converter | `CALL_TOOL` | `bunx kordoc@latest` |
| Write output artifact | `WRITE` | Markdown or JSON output |
| Flatten tables and strip PUA glyphs | `CALL_TOOL` | `resources/flatten-tables.ts` |
| Inspect extraction quality | `VALIDATE` | Verification step |
| Report result | `NOTIFY` | Final user-facing summary |

### Tools and instruments
- `kordoc`: primary HWP-family conversion CLI
- `flatten-tables.ts`: post-processing for GFM tables and Hancom PUA cleanup
- `bun` / `bunx`: runtime and CLI executor

### Canonical command path
```bash
bunx kordoc@latest "{input_path}" -o "{output_path}"
# fresh clone: run `bun install` in .agents/skills/oma-hwp/resources/ first (node_modules is gitignored)
bun ".agents/skills/oma-hwp/resources/flatten-tables.ts" "{output_path}"
```

For batch conversion, use an explicit output directory:
```bash
bunx kordoc@latest "{input_pattern}" -d "{output_dir}"
```

### Resource scope
| Scope | Resource target |
|-------|-----------------|
| `LOCAL_FS` | Input HWP-family files and generated outputs |
| `PROCESS` | `bunx kordoc` and `bun run` subprocesses |
| `MEMORY` | Format decisions, validation notes, and final report |

### Preconditions
- Input file exists and is readable.
- Output location is writable or can be created.
- `bun` is installed.
- `kordoc` can parse the document or fail with a reportable error.

### Effects and side effects
- Creates Markdown or JSON output files.
- May flatten merged-cell tables, trading cell fidelity for Markdown compatibility.
- Strips Private Use Area characters by default because they render as blanks without Hancom fonts.
- Does not intentionally modify the source HWP-family document.

### Guardrails
1. Always pass `@latest` or an explicit pinned version to avoid stale `bunx` cache.
2. Always pass an explicit output target when the user expects a file.
3. Do not add custom security defenses around kordoc's ZIP, XML, SSRF, or XSS defenses.
4. Report missing tables, garbled text, empty output, encrypted segments, and best-effort DRM extraction.
5. Keep full CLI details in `resources/execution-protocol.md` and troubleshooting branches in `resources/troubleshooting.md`.

### Supported Formats
| Format | Extension | Notes |
|--------|-----------|-------|
| HWP 5.x binary | `.hwp` | Full support (incl. DRM-locked via kordoc's rhwp-algorithm port) |
| HWPX | `.hwpx` | Full support incl. nested tables, merged cells |
| HWPML | `.hwp` (XML variant) | Auto-detected by signature |

> kordoc also parses PDF / XLSX / DOCX. Those are intentionally outside this skill's scope; see "When NOT to use".

## References
- Execution protocol: `resources/execution-protocol.md`
- Troubleshooting: `resources/troubleshooting.md`
- Configuration: `config/hwp-config.yaml`
- Upstream: https://github.com/chrisryugj/kordoc
- Related: `../oma-pdf/SKILL.md` (use for `.pdf` inputs)

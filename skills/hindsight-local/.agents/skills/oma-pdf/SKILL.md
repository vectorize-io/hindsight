---
name: oma-pdf
description: >
  Convert PDF files to Markdown using opendataloader-pdf. Extracts text, tables, headings,
  lists, and images with correct reading order. Use for PDF parsing, PDF to Markdown conversion,
  document extraction, and AI-ready data preparation.
---

# PDF Skill - PDF to Markdown Conversion

## Scheduling

### Goal
Convert PDF files into structured Markdown or another requested extraction format while preserving readable document structure for LLM context, RAG, or downstream review.

### Intent signature
- User asks to convert, parse, read, extract, or transform a PDF.
- User needs PDF text, headings, lists, tables, or images prepared for AI consumption.
- User mentions "PDF to markdown", "parse PDF", "read this PDF", or equivalent wording.

### When to use
- Converting PDF documents to Markdown for LLM context or RAG
- Extracting structured content such as tables, headings, lists, images, footnotes, or hyperlinks
- Preparing PDF data for AI consumption
- Checking whether a PDF has a text layer before choosing OCR

### When NOT to use
- Generating or creating PDFs -> use document-generation tools
- Editing existing PDFs -> out of scope
- Reading an already-text file -> use direct file reading
- Processing HWP, HWPX, DOCX, XLSX, or slide decks -> use the matching document skill

### Expected inputs
- `input_path`: PDF file or folder path
- `output_dir`: optional target directory
- `format`: optional output format, default `markdown`
- `ocr_languages`: optional OCR language list for scanned or image-based PDFs
- `extraction_options`: optional flags for tagged structure, image extraction, or hybrid conversion

### Expected outputs
- Markdown, text, JSON, HTML, or combined extraction output
- Normalized Markdown when Markdown is produced
- A short report with output path, page count, and conversion issues

### Dependencies
- `uvx opendataloader-pdf` for standard conversion
- `uvx --from "opendataloader-pdf[hybrid]" opendataloader-pdf-hybrid` for OCR or hybrid conversion (the hybrid server is a console script of the `[hybrid]` extra, not a standalone package)
- `uvx mdformat` for Markdown normalization
- Local filesystem access to input and output paths
- Optional OCR runtime via the hybrid server

### Control-flow features
- Branches on text-layer quality, tagged PDF availability, scan/OCR needs, and user-requested output format
- Calls external CLI tools through `uvx`
- Reads local files and writes local extraction outputs
- Uses a hybrid server only when OCR or complex extraction needs justify it

## Structural Flow

### Entry
1. Confirm that the input path exists and is a PDF file, PDF folder, or supported batch input.
2. Check file size and warn when the input is large enough to risk slow conversion or memory pressure.
3. Resolve `output_dir` and the expected output filename.

### Scenes
1. **PREPARE**: Validate the input path, output target, and requested extraction options.
2. **ACQUIRE**: Assess whether the PDF has a readable text layer by extracting a text preview.
3. **ACT**: Convert using standard mode, tagged-structure mode, or hybrid OCR mode.
4. **VERIFY**: Run `mdformat` for Markdown output and inspect the result for readable structure.
5. **FINALIZE**: Report output path, page count, format, and any extraction quality issues.

### Transitions
- If the preview text is readable, use standard conversion.
- If the PDF is tagged and standard output is garbled, retry with `--use-struct-tree`.
- If the PDF is scanned or image-based, start or reuse the hybrid OCR server and convert with hybrid mode.
- If conversion fails because the PDF is encrypted, stop and ask for the password or an unlocked copy.
- If conversion hits memory or size limits, process smaller page ranges or batches.

### Failure and recovery
| Failure | Recovery |
|---------|----------|
| `uvx` unavailable | Ask user to install `uv` before conversion |
| `opendataloader-pdf-hybrid` not found | Invoke via `uvx --from "opendataloader-pdf[hybrid]" opendataloader-pdf-hybrid`; the bare package name does not exist on PyPI |
| Password-protected PDF | Ask for password or unlocked PDF |
| Garbled output | Retry with tagged structure or hybrid mode |
| Missing tables | Retry with hybrid mode for complex or borderless tables |
| OCR language mismatch | Retry with explicit OCR languages, for example `ko,en` |
| Large file or memory pressure | Split into page ranges or batch smaller inputs |

### Exit
- Success: output file exists, Markdown is formatted when applicable, and extracted structure is readable.
- Partial success: output exists but quality issues are reported explicitly.
- Failure: no reliable output is produced and the blocking cause is reported.

## Logical Operations

### Actions
| Action | SSL primitive | Evidence |
|--------|---------------|----------|
| Validate path and options | `VALIDATE` | Input preflight in execution protocol |
| Probe text layer | `READ` | Text preview extraction |
| Choose conversion strategy | `SELECT` | Standard, tagged, or hybrid mode decision |
| Run converter | `CALL_TOOL` | `uvx opendataloader-pdf` |
| Start OCR server | `CALL_TOOL` | `uvx --from "opendataloader-pdf[hybrid]" opendataloader-pdf-hybrid` |
| Write output artifact | `WRITE` | Markdown, text, JSON, or HTML output |
| Normalize Markdown | `CALL_TOOL` | `uvx mdformat` |
| Inspect extraction quality | `VALIDATE` | Structure/readability verification |
| Report result | `NOTIFY` | Final user-facing summary |

### Tools and instruments
- `opendataloader-pdf`: primary PDF extraction CLI
- `opendataloader-pdf-hybrid`: hybrid OCR and complex extraction path
- `mdformat`: Markdown normalization
- Filesystem commands such as `file`, `wc`, or `pdfinfo` may be used for preflight when available

### Canonical command path
```bash
uvx opendataloader-pdf "{input_path}" --format markdown --output-dir "{output_dir}"
uvx mdformat "{output_path}"
```

For scanned/image-based PDFs, start OCR first and then convert through hybrid mode:
```bash
uvx --from "opendataloader-pdf[hybrid]" opendataloader-pdf-hybrid --port 5002 --force-ocr --ocr-lang "{languages}"
uvx opendataloader-pdf --hybrid docling-fast "{input_path}" --format markdown --output-dir "{output_dir}"
```

### Resource scope
| Scope | Resource target |
|-------|-----------------|
| `LOCAL_FS` | Input PDFs and generated output files |
| `PROCESS` | `uvx` subprocesses and optional hybrid server |
| `MEMORY` | Extracted previews and validation notes |
| `OTHER` | OCR model/runtime behavior inside hybrid mode |

### Preconditions
- The input PDF path exists and is readable.
- The output location is writable or can be created.
- Required CLIs are available through `uvx`.
- OCR is only attempted when hybrid mode is available or can be started.

### Effects and side effects
- Creates or overwrites extraction output depending on configuration and user intent.
- May start a local hybrid OCR server on the configured port.
- May consume significant CPU, memory, or time for large or scanned PDFs.
- Does not intentionally modify the source PDF.

### Guardrails
1. Do not invent missing content when extraction is incomplete.
2. Always report garbled text, missing tables, OCR uncertainty, or partial extraction.
3. Prefer standard conversion first when the text layer is readable.
4. Use OCR only when the PDF is scanned, image-based, or standard extraction quality is insufficient.
5. Keep detailed command sequences in `resources/execution-protocol.md` rather than duplicating every variant here.

## References
- Execution protocol: `resources/execution-protocol.md`
- Configuration: `config/pdf-config.yaml`
- Context loading: `../_shared/core/context-loading.md`
- Quality principles: `../_shared/core/quality-principles.md`

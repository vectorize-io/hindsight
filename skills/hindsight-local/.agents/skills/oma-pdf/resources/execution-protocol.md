# PDF Conversion - Execution Protocol

## Step 0: Validate Input

1. Confirm the PDF file path exists
2. Check file size (`wc -c` or `ls -lh`); warn if >100MB
3. Determine output location:
   - If user specified output path → use it
   - If not specified → use the same directory as the input PDF
4. Determine output filename: `{input_name}.md` (same base name, `.md` extension)

## Step 1: Assess PDF Type

Quick check to determine conversion strategy:

```bash
# Check if PDF has text layer (vs scanned image)
uvx opendataloader-pdf input.pdf --format text --output-dir /tmp/pdf-check/
```

- If output contains readable text → standard mode
- If output is empty or garbled → needs OCR (hybrid mode)

## Step 2: Convert

### Standard conversion
```bash
uvx opendataloader-pdf "{input_path}" --format markdown --output-dir "{output_dir}"
```

### If Tagged PDF (structured documents, official reports)
```bash
uvx opendataloader-pdf "{input_path}" --format markdown --output-dir "{output_dir}" --use-struct-tree
```

### If scanned/image-based PDF (requires hybrid server)
```bash
# Start hybrid server (if not already running).
# The server is a console script of the [hybrid] extra — bare `uvx opendataloader-pdf-hybrid` fails (no such PyPI package).
# First run downloads a large OCR stack (torch, easyocr, docling); warn the user before starting.
uvx --from "opendataloader-pdf[hybrid]" opendataloader-pdf-hybrid --port 5002 --force-ocr --ocr-lang "{languages}"

# Convert
uvx opendataloader-pdf --hybrid docling-fast "{input_path}" --format markdown --output-dir "{output_dir}"
```

## Step 3: Lint & Format

Run `mdformat` to normalize the converted Markdown:

```bash
uvx mdformat "{output_path}"
```

This auto-fixes:
- Inconsistent heading style
- Missing blank lines around blocks
- Trailing whitespace
- Unordered list marker normalization

## Step 4: Verify

1. Read the generated Markdown file
2. Verify structure:
   - Headings preserved (`#`, `##`, etc.)
   - Tables rendered correctly (pipe syntax)
   - Lists maintained (bullets, numbered)
   - No garbled or missing sections
3. If output directory is temporary, move the file to the target location
4. If the user needs the content in the conversation, read and present it

## Step 5: Report

Tell the user:
- Output file path
- Page count processed
- Any issues encountered (missing tables, OCR quality, etc.)
- Suggest hybrid mode if standard conversion had quality issues

## Error Recovery

| Error | Recovery |
|-------|----------|
| `uvx` not found | Ask user to install `uv`: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| PDF password protected | Ask user for the password, then retry with `-p "{password}"` (`opendataloader-pdf --password`) |
| Hybrid server not running | Guide user to start it, or fall back to standard mode with quality warning |
| Out of memory on large PDF | Process in smaller page ranges |
| Network error (hybrid mode) | Check server port, retry, or fall back to standard mode |

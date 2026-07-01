---
name: convert
description: Convert a file from one format to another, routed by media category. Documents (PDF/HWP/HWPX/HWPML) extract to Markdown via oma-pdf/oma-hwp; images, video, and audio transcode to a target format via ffmpeg.
disable-model-invocation: true
---

# MANDATORY RULES: VIOLATION IS FORBIDDEN

- **Response language follows `language` setting in `.agents/oma-config.yaml` if configured.**
- **NEVER skip steps.** Execute from Step 1 in order.
- **Default output location: same directory as input file.**
- **Route by category, then by extension** — never run a document converter on a media file or vice versa.
- **Never re-encode losslessly-convertible data destructively without saying so** — report quality/codec choices.

---

> **Vendor note:** This workflow executes inline (no subagent spawning). It is a dispatcher over
> three engines: `oma-pdf` (PDF), `oma-hwp` (HWP family), and `ffmpeg` (image/video/audio
> transcode — already provisioned for `oma-video`). Read the matching SKILL.md when a document
> branch needs detail beyond the canonical command path below.

---

## Step 1: Validate Input & Route

1. Identify the input file path from the user's request.
2. Confirm the file exists (`ls -lh "{path}"`).
3. Determine the **target format**:
   - User said it explicitly (`--to webp`, `to png`, `jpg -> webp`) → use it.
   - Not specified → use the **category default** (Document → Markdown; media → ask which target).
4. Determine output location:
   - User specified a path → use it
   - Not specified → same directory as the input file
5. Set output filename: `{input_basename}.{target_ext}`
6. **Route by category** (extension, case-insensitive):

   | Category | Extensions | Branch | Engine |
   |----------|-----------|--------|--------|
   | **Document** | `.pdf` | Step 2A | `oma-pdf` → `uvx opendataloader-pdf` |
   | **Document** | `.hwp`, `.hwpx`, `.hwpml` | Step 2B | `oma-hwp` → `bunx kordoc` |
   | **Image** | `.jpg`, `.jpeg`, `.png`, `.webp`, `.avif`, `.gif`, `.tiff`, `.bmp`, `.heic` | Step 2C | `ffmpeg` |
   | **Video** | `.mp4`, `.webm`, `.mov`, `.mkv`, `.avi`, `.gif` | Step 2D | `ffmpeg` |
   | **Audio** | `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.opus`, `.aac` | Step 2E | `ffmpeg` |
   | other | — | Stop — out of scope | route to the matching skill |

If user provided no file path, ask:
```
Which file should I convert? Provide the path, and the target format if it's an image/video/audio file.
```

For batch input (a glob or directory), apply the same category routing per file; mixed-type
batches run each type through its own branch.

---

## Step 2A: Document — PDF branch

```bash
uvx opendataloader-pdf "{input_path}" --format markdown --output-dir "{output_dir}"
```

Variants: `--use-struct-tree` (Tagged PDF), `--image-output embedded` (inline images),
`--format markdown,json` (multiple formats). Then go to **Step 3 (document normalization)**.

---

## Step 2B: Document — HWP branch

Always pin `@latest` to avoid stale `bunx` cache:

```bash
bunx kordoc@latest "{input_path}" -o "{output_path}"
```

Batch into a directory: `bunx kordoc@latest "{input_pattern}" -d "{output_dir}"`.

> First run on a fresh clone: run `bun install` in `.agents/skills/oma-hwp/resources/` once
> (its `node_modules` is gitignored). Then go to **Step 3 (document normalization)**.

---

## Step 2C: Image → Image

Transcode with ffmpeg. The target codec is inferred from the output extension.

```bash
ffmpeg -i "{input_path}" "{output_path}"          # e.g. photo.jpg -> photo.png
```

Common targets:
```bash
ffmpeg -i in.jpg -quality 80 out.webp             # WebP (lossy, q 0-100)
ffmpeg -i in.jpg -lossless 1 out.webp             # WebP lossless
ffmpeg -i in.png out.avif                          # AVIF (needs libaom/libsvtav1 build)
ffmpeg -i in.png -vf "scale=800:-1" out.webp       # resize while converting
```

> AVIF/HEIC support depends on the local ffmpeg build. If ffmpeg lacks the codec, report it and
> suggest `sharp` (already used by `oma-slide`) or ImageMagick as a fallback. Then go to **Step 4**.

---

## Step 2D: Video → Video

```bash
ffmpeg -i "{input_path}" "{output_path}"          # container/codec from target ext
```

Common targets:
```bash
ffmpeg -i in.mov -c:v libx264 -crf 23 -c:a aac out.mp4        # MOV -> MP4 (H.264)
ffmpeg -i in.mp4 -c:v libvpx-vp9 -crf 32 -b:v 0 out.webm      # MP4 -> WebM (VP9)
ffmpeg -i in.mp4 -vf "fps=12,scale=480:-1" out.gif           # MP4 -> GIF
```

State the CRF/codec used in the report (quality is not lossless). Then go to **Step 4**.

---

## Step 2E: Audio → Audio

```bash
ffmpeg -i "{input_path}" "{output_path}"          # codec from target ext
```

Common targets:
```bash
ffmpeg -i in.wav -b:a 192k out.mp3                 # WAV -> MP3
ffmpeg -i in.m4a out.wav                            # M4A -> WAV (PCM)
ffmpeg -i in.flac -c:a libopus -b:a 96k out.opus   # FLAC -> Opus
```

Then go to **Step 4**.

---

## Step 3: Normalize (Document branches only)

### PDF
```bash
uvx mdformat "{output_path}"          # add --check for dry-run
```

### HWP
Flatten GFM tables and strip Hancom Private Use Area glyphs:
```bash
bun ".agents/skills/oma-hwp/resources/flatten-tables.ts" "{output_path}"
```
Skip flattening only when the caller explicitly needs HTML tables or PUA glyphs preserved.

---

## Step 4: Verify Output

1. Confirm the output file was created (`ls -lh "{output_path}"`).
2. **Document:** read the first ~50 lines — headings (`#`), tables (`|`), no garbled/encoding issues.
3. **Image/Video/Audio:** probe the result:
   ```bash
   ffprobe -hide_banner "{output_path}"
   ```
   Verify the codec/format, dimensions/duration, and that the file is non-empty.
4. If quality is poor:
   - **PDF**: try `--use-struct-tree`; suggest hybrid OCR mode for scanned/complex PDFs
   - **HWP**: empty output usually means scanned-image content → recommend OCR outside this skill;
     see `.agents/skills/oma-hwp/resources/troubleshooting.md`
   - **Media**: adjust CRF/bitrate/quality, or report a missing codec in the local ffmpeg build

---

## Step 5: Report

Tell the user:
- Output file path
- Source format → target format (e.g. `HWPX → Markdown`, `jpg → webp`)
- Quick quality assessment (document: headings/tables/images detected; media: codec, size, dimensions/duration, quality setting used)
- Any issues or recommendations

**Output example:**
```
Converted successfully.

Output: /path/to/photo.webp
- jpg → webp (lossy, quality 80)
- 1920×1080, 1.2 MB → 240 KB
- No issues found
```

---

## OCR Mode (Scanned PDFs)

Applies to the **PDF branch only**. If standard conversion produces empty or garbled output, start
the hybrid OCR server (console script of the `[hybrid]` extra — first run downloads a large OCR stack):

```bash
uvx --from "opendataloader-pdf[hybrid]" opendataloader-pdf-hybrid --port 5002 --force-ocr --ocr-lang "en"
uvx opendataloader-pdf --hybrid docling-fast "{input_path}" --format markdown --output-dir "{output_dir}"
```

For Korean documents, use `--ocr-lang "ko,en"`. HWP-family files have no OCR path here; an empty HWP
conversion means image-based content that must be OCR'd outside `oma-hwp`.

---

## Error Recovery

| Error | Branch | Recovery |
|-------|--------|----------|
| `uvx` not found | PDF | Guide: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `bun` / `bunx` not found | HWP | Ask user to install Bun |
| `Cannot find module "turndown"` | HWP | `bun install` in `.agents/skills/oma-hwp/resources/` |
| `ffmpeg` / `ffprobe` not found | media | Guide install (`brew install ffmpeg`); `oma video doctor` also provisions it |
| Unknown encoder / codec | media | The local ffmpeg lacks that codec (e.g. AVIF/HEIC) → suggest `sharp` or ImageMagick |
| File not found | all | Ask user to verify the path |
| Permission denied | all | Check file permissions |
| Empty output | document | PDF → suggest OCR mode; HWP → likely scanned-image content |
| Encrypted / DRM-locked | document | PDF → ask for password/unlocked copy; HWP → kordoc handles many DRM cases |
| Timeout on large file | all | Process page ranges / smaller batches; for video, transcode a clip first |

---

## Quick Reference

| Command | Effect |
|---------|--------|
| `/convert document.pdf` | PDF → Markdown (oma-pdf) |
| `/convert report.hwp` | HWP → Markdown (oma-hwp) |
| `/convert form.hwpx` | HWPX → Markdown (oma-hwp) |
| `/convert photo.jpg --to webp` | Image transcode via ffmpeg |
| `/convert clip.mov --to mp4` | Video transcode via ffmpeg |
| `/convert track.wav --to mp3` | Audio transcode via ffmpeg |
| `/convert clip.mp4 --to gif` | Video → animated GIF |
| `/convert *.pdf` | Batch convert all PDFs |
| `/convert *.jpg --to webp` | Batch image transcode |

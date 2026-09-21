# API Documentation Examples

This directory contains runnable example scripts that serve as the source of truth for code samples in the documentation.

## How It Works

1. **Scripts are runnable** - Each file can be executed as a smoke test
2. **Markers define sections** - Code between `# [docs:section-name]` and `# [/docs:section-name]` markers is extracted
3. **Docs import at build time** - MDX files use `raw-loader` to import scripts, then `CodeSnippet` extracts marked sections

## File Structure

| File | Documentation | Description |
|------|---------------|-------------|
| `quickstart.py/mjs/sh` | quickstart.md | Getting started examples |
| `retain.py/mjs/sh` | retain.md | Memory ingestion examples |
| `recall.py/mjs/sh` | recall.md | Memory retrieval examples |
| `reflect.py/mjs/sh` | reflect.md | AI reflection examples |
| `memory-banks.py/mjs` | memory-banks.md | Bank management examples |
| `documents.py/mjs` | documents.md | Document CRUD examples |
| `reflections.py` | reflections.md | Reflections CRUD examples |
| `main-methods.py` | main-methods.md | Core method examples |
| `cli-reference.sh` | cli.mdx | CLI command examples |

## Running Examples

```bash
# Run all Python examples
for f in *.py; do python "$f"; done

# Run all Node.js examples
for f in *.mjs; do node "$f"; done

# Run all CLI examples
for f in *.sh; do bash "$f"; done
```

Requires a running Hindsight server at `http://localhost:8888` (or set `HINDSIGHT_API_URL`).

## Legacy Examples

The `legacy/` folder contains deprecated example files kept only for backward compatibility with older documentation versions. These files are **not runnable** and are skipped by CI tests.

## What's NOT Covered

### 1. OpenAPI Auto-Generated Docs (`/api-reference/*`)

These pages are generated directly from the OpenAPI specification. The spec itself is the source of truth, and the generated docs reflect it automatically. No manual code examples to validate.

### 2. Interactive CLI Commands

| Command | Reason |
|---------|--------|
| `hindsight configure` | Requires interactive user input (prompts for API URL, credentials) |
| `hindsight configure --show` | Displays sensitive configuration, not suitable for automated tests |

### 3. Installation/Setup Instructions

Documentation sections covering `pip install`, `npm install`, or system setup are instructions, not executable code samples. These are validated by the CI environment setup itself.

### 4. Error Handling Examples

Some docs show error responses (e.g., "what happens when bank doesn't exist"). These require intentionally broken states that would fail smoke tests. Error behavior is covered by unit tests instead.

## Adding New Examples

1. Create or edit the appropriate script file
2. Add markers around the new code section:
   ```python
   # [docs:my-new-section]
   client.some_method(...)
   # [/docs:my-new-section]
   ```
3. Reference in the MDX file:
   ```mdx
   import myScript from '!!raw-loader!@site/examples/api/my-script.py';
   <CodeSnippet code={myScript} section="my-new-section" language="python" />
   ```
4. Run the script locally to verify it works

## Marker Format

- **Python/Bash**: `# [docs:section-name]` / `# [/docs:section-name]`
- **JavaScript**: `// [docs:section-name]` / `// [/docs:section-name]`

Section names should be kebab-case and descriptive (e.g., `retain-with-context`, `recall-basic`).

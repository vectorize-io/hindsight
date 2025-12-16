#!/usr/bin/env python3
"""
Documentation Example Tester

Tests code examples from documentation by running them directly.
Uses deterministic transformations (no LLM) for test generation.
LLM is only used to analyze failures and determine if they're real doc bugs.

Usage:
    python scripts/test-doc-examples.py

Environment variables:
    OPENAI_API_KEY: Required for failure analysis
    HINDSIGHT_API_URL: URL of running Hindsight server (default: http://localhost:8888)
"""

import os
import re
import sys
import site
import json
import glob
import subprocess
import tempfile
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from openai import OpenAI

# Thread-safe print
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)
        sys.stdout.flush()


@dataclass
class CodeExample:
    file_path: str
    language: str
    code: str
    context: str
    line_number: int


@dataclass
class TestResult:
    example: CodeExample
    success: bool
    output: str
    error: Optional[str] = None
    transformed_code: Optional[str] = None
    skip_reason: Optional[str] = None


@dataclass
class TestReport:
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[TestResult] = field(default_factory=list)

    def add_result(self, result: TestResult):
        self.total += 1
        self.results.append(result)
        if result.skip_reason:
            self.skipped += 1
        elif result.success:
            self.passed += 1
        else:
            self.failed += 1


# =============================================================================
# STEP 1: Extract code blocks from markdown
# =============================================================================

def find_markdown_files(repo_root: str) -> list[str]:
    """Find all markdown files, excluding auto-generated docs."""
    skip_patterns = [
        "node_modules", ".git", "venv", "__pycache__",
        "hindsight_client_api/docs", "hindsight-clients/typescript/docs",
        "target/", "dist/",
    ]
    md_files = []
    for pattern in ["*.md", "**/*.md"]:
        for f in glob.glob(os.path.join(repo_root, pattern), recursive=True):
            if os.path.islink(f):
                continue
            if any(skip in f for skip in skip_patterns):
                continue
            md_files.append(f)
    return sorted(set(md_files))


def extract_code_blocks(file_path: str) -> list[CodeExample]:
    """Extract code blocks from a markdown file."""
    with open(file_path, "r") as f:
        content = f.read()

    examples = []
    pattern = r"```(\w+)\n(.*?)```"

    for match in re.finditer(pattern, content, re.DOTALL):
        language = match.group(1).lower()
        code = match.group(2).strip()
        line_number = content[:match.start()].count('\n') + 1

        if language in ["python", "typescript", "javascript", "bash", "sh"]:
            start = max(0, match.start() - 150)
            end = min(len(content), match.end() + 150)
            context = content[start:end]

            examples.append(CodeExample(
                file_path=file_path,
                language=language,
                code=code,
                context=context,
                line_number=line_number
            ))

    return examples


# =============================================================================
# STEP 2: Determine if example should be skipped (no LLM needed)
# =============================================================================

def should_skip(code: str, language: str) -> Optional[str]:
    """Determine if example should be skipped. Returns reason or None."""
    code_lower = code.lower().strip()

    # Installation/setup commands
    if language in ["bash", "sh"]:
        if code_lower.startswith(("pip install", "npm install", "yarn add", "uv pip", "cargo install", "curl ", "wget ")):
            return "Installation command"
        if "docker" in code_lower or "docker-compose" in code_lower:
            return "Docker command"
        if code_lower.startswith("helm "):
            return "Helm command"
        if code_lower.startswith(("cargo build", "cargo test")):
            return "Cargo command"
        if "pytest" in code_lower:
            return "Test suite command"
        if code_lower.startswith("git clone"):
            return "Git clone"
        if "./scripts/" in code_lower:
            return "Development script"
        if any(x in code_lower for x in ["npm run dev", "npm run start", "npm run build", "npm run deploy"]):
            return "NPM script"
        if code_lower.startswith("cd ") and not code_lower.startswith("cd /tmp"):
            return "Directory change"
        if code_lower.startswith("export "):
            return "Environment variable"

    # Config files
    if language in ["yaml", "toml", "json", "env"]:
        return "Configuration file"

    # Too short
    if len(code.strip()) < 20:
        return "Too short"

    return None


# =============================================================================
# STEP 3: Transform code (LLM-assisted for accuracy)
# =============================================================================

def transform_code(client: OpenAI, example: CodeExample, hindsight_url: str, cli_available: bool, model: str) -> tuple[str, Optional[str]]:
    """Use LLM to transform doc code into runnable test. Returns (script, skip_reason)."""

    bank_id = f"doc-test-{uuid.uuid4()}"

    # Skip CLI examples if CLI not available
    if not cli_available and example.language in ["bash", "sh"] and "hindsight " in example.code.lower():
        return "", "CLI not available"

    lang_instructions = ""
    if example.language == "python":
        lang_instructions = f"""
OUTPUT FORMAT: Python script (.py)
- Add ALL necessary imports at the top (hindsight_client, requests, uuid, os, datetime, etc.)
- The Hindsight client is SYNCHRONOUS - do NOT use async/await
- Initialize client with: Hindsight(base_url="{hindsight_url}")
- Use bank_id = "{bank_id}" for all bank operations
- Wrap in try/finally for cleanup
- Cleanup: requests.delete("{hindsight_url}/v1/default/banks/{bank_id}")
- End with: print("TEST PASSED")
"""
    elif example.language in ["typescript", "javascript"]:
        lang_instructions = f"""
OUTPUT FORMAT: JavaScript ES module (.mjs) - NOT TypeScript
- REMOVE all TypeScript type annotations (: string, : Promise<void>, : {{ key: type }}, etc.)
- Use ES module import: import {{ HindsightClient }} from '@vectorize-io/hindsight-client';
- Do NOT use require()
- Initialize: new HindsightClient({{ baseUrl: '{hindsight_url}' }})
- Use bankId = '{bank_id}' for all bank operations
- Wrap in async IIFE: (async () => {{ ... }})();
- Use try/finally for cleanup
- Cleanup: await fetch("{hindsight_url}/v1/default/banks/{bank_id}", {{ method: "DELETE" }})
- End with: console.log("TEST PASSED")
- Use native fetch() and crypto.randomUUID(), no external packages
"""
    elif example.language in ["bash", "sh"]:
        lang_instructions = f"""
OUTPUT FORMAT: Bash script (.sh)
- Start with: #!/bin/bash and set -e
- Export: HINDSIGHT_API_URL="{hindsight_url}"
- Replace any placeholder bank IDs (<bank_id>, my-bank, demo) with: {bank_id}
- If the code references files (notes.txt, etc.), create temp files first
- Cleanup: curl -s -X DELETE "{hindsight_url}/v1/default/banks/{bank_id}" || true
- End with: echo "TEST PASSED"
"""

    prompt = f"""Transform this documentation code example into a complete, runnable test script.

DOCUMENTATION CODE ({example.language}):
```{example.language}
{example.code}
```

{lang_instructions}

RULES:
- Keep the original code logic intact
- Add whatever setup is needed to make it run standalone
- Replace placeholder values (my-bank, <bank_id>, localhost URLs) with real values
- Ensure proper cleanup of test resources
- The script must be complete and runnable as-is

Output ONLY the transformed code, no explanation."""

    is_reasoning = model.startswith(("o1", "o3"))
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if is_reasoning:
        kwargs["max_completion_tokens"] = 4000
    else:
        kwargs["temperature"] = 0
        kwargs["max_tokens"] = 4000

    try:
        response = client.chat.completions.create(**kwargs)
        script = response.choices[0].message.content

        # Clean up markdown code blocks if present
        script = re.sub(r'^```\w*\n', '', script)
        script = re.sub(r'\n```$', '', script)
        script = script.strip()

        return script, None
    except Exception as e:
        return "", f"Transform failed: {e}"


# =============================================================================
# STEP 4: Run tests
# =============================================================================

def run_python(script: str, timeout: int = 60) -> tuple[bool, str, Optional[str]]:
    """Run Python script."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        f.flush()
        try:
            site_packages = site.getsitepackages()
            pythonpath = ":".join(site_packages + [os.environ.get("PYTHONPATH", "")])

            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, text=True, timeout=timeout,
                env={**os.environ, "PYTHONPATH": pythonpath}
            )
            output = result.stdout + result.stderr
            if "TEST PASSED" in output:
                return True, output, None
            return result.returncode == 0, output, result.stderr if result.returncode != 0 else None
        except subprocess.TimeoutExpired:
            return False, "", "Timeout"
        except Exception as e:
            return False, "", str(e)
        finally:
            os.unlink(f.name)


def run_javascript(script: str, timeout: int = 60) -> tuple[bool, str, Optional[str]]:
    """Run JavaScript script."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mjs', delete=False, dir='/tmp') as f:
        f.write(script)
        f.flush()
        try:
            env = {**os.environ}
            env["NODE_PATH"] = f"/tmp/node_modules:{env.get('NODE_PATH', '')}"

            result = subprocess.run(
                ["node", f.name],
                capture_output=True, text=True, timeout=timeout,
                env=env, cwd="/tmp"
            )
            output = result.stdout + result.stderr
            if "TEST PASSED" in output:
                return True, output, None
            return result.returncode == 0, output, result.stderr if result.returncode != 0 else None
        except subprocess.TimeoutExpired:
            return False, "", "Timeout"
        except Exception as e:
            return False, "", str(e)
        finally:
            os.unlink(f.name)


def run_bash(script: str, timeout: int = 60) -> tuple[bool, str, Optional[str]]:
    """Run bash script."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(script)
        f.flush()
        os.chmod(f.name, 0o755)
        try:
            result = subprocess.run(
                ["bash", f.name],
                capture_output=True, text=True, timeout=timeout
            )
            output = result.stdout + result.stderr
            if "TEST PASSED" in output:
                return True, output, None
            return result.returncode == 0, output, result.stderr if result.returncode != 0 else None
        except subprocess.TimeoutExpired:
            return False, "", "Timeout"
        except Exception as e:
            return False, "", str(e)
        finally:
            os.unlink(f.name)


# =============================================================================
# STEP 5: Analyze failures with LLM
# =============================================================================

def get_source_context(example: CodeExample, repo_root: str) -> str:
    """Get relevant source code for failure analysis."""
    parts = []
    code_lower = example.code.lower()

    if example.language == "python":
        if "recall" in code_lower or "weight" in code_lower:
            try:
                with open(os.path.join(repo_root, "hindsight-clients/python/hindsight_client_api/models/recall_result.py")) as f:
                    parts.append("=== RecallResult Model ===\n" + f.read()[:2000])
            except: pass
        if "reflect" in code_lower:
            try:
                with open(os.path.join(repo_root, "hindsight-clients/python/hindsight_client_api/models/reflect_response.py")) as f:
                    parts.append("=== ReflectResponse Model ===\n" + f.read()[:2000])
            except: pass
        try:
            with open(os.path.join(repo_root, "hindsight-clients/python/hindsight_client/__init__.py")) as f:
                parts.append("=== Hindsight Client ===\n" + f.read()[:3000])
        except: pass

    elif example.language in ["typescript", "javascript"]:
        try:
            with open(os.path.join(repo_root, "hindsight-clients/typescript/src/index.ts")) as f:
                parts.append("=== TypeScript Client ===\n" + f.read()[:4000])
        except: pass

    elif example.language in ["bash", "sh"]:
        try:
            with open(os.path.join(repo_root, "hindsight-cli/src/main.rs")) as f:
                lines = f.read().split('\n')[:350]
                parts.append("=== CLI Commands ===\n" + '\n'.join(lines))
        except: pass

    return "\n\n".join(parts)


def get_doc_context(example: CodeExample) -> str:
    """Get the full documentation context around the failing code example."""
    try:
        with open(example.file_path, "r") as f:
            content = f.read()

        # Find the code block and get surrounding context (500 chars before/after)
        # This gives us the explanatory text around the code
        code_start = content.find(example.code[:50])  # Find by first 50 chars
        if code_start == -1:
            code_start = example.line_number * 50  # Rough estimate

        start = max(0, code_start - 500)
        end = min(len(content), code_start + len(example.code) + 500)

        return content[start:end]
    except:
        return example.context  # Fall back to the small context we already have


def analyze_failure(client: OpenAI, result: TestResult, repo_root: str, model: str) -> dict:
    """Use LLM to determine if failure is a real doc bug."""
    source = get_source_context(result.example, repo_root)
    doc_context = get_doc_context(result.example)

    prompt = f"""Analyze this documentation test failure.

## Documentation File: {result.example.file_path}

### Documentation Context (text around the code example)
```markdown
{doc_context}
```

### The Code Example Being Tested (line {result.example.line_number})
```{result.example.language}
{result.example.code}
```

## Error When Running
{result.error[:800] if result.error else "Unknown"}

## Transformed Test Code (what we actually ran)
```
{result.transformed_code[:1500] if result.transformed_code else "N/A"}
```

## Actual Source Code (ground truth - what the API really looks like)
{source[:6000] if source else "Not available"}

## Your Task
Compare the DOCUMENTATION against the ACTUAL SOURCE CODE.

1. Does the documentation show something that doesn't exist in the source code?
   - Wrong method names?
   - Wrong attribute names (e.g., .weight when there's no weight field)?
   - Wrong CLI commands?
   - Wrong parameters?

2. Or is the documentation correct, but our test transformation/execution failed?
   - Missing imports we didn't add?
   - Environment issues?
   - Timing/race conditions?

Respond JSON:
{{
    "is_doc_bug": true/false,
    "confidence": "high/medium/low",
    "reason": "brief explanation of what's wrong",
    "fix": "if doc bug, what should the doc say instead"
}}"""

    is_reasoning = model.startswith(("o1", "o3"))
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    if is_reasoning:
        kwargs["max_completion_tokens"] = 2000
    else:
        kwargs["temperature"] = 0

    try:
        response = client.chat.completions.create(**kwargs)
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"is_doc_bug": True, "confidence": "low", "reason": str(e)}


# =============================================================================
# Main test runner
# =============================================================================

def test_example(example: CodeExample, openai_client: OpenAI, hindsight_url: str, cli_available: bool, model: str) -> TestResult:
    """Test a single code example."""

    # Check if should skip
    skip = should_skip(example.code, example.language)
    if skip:
        return TestResult(example=example, success=True, output="", skip_reason=skip)

    # Transform using LLM
    try:
        transformed, skip = transform_code(openai_client, example, hindsight_url, cli_available, model)
        if skip:
            return TestResult(example=example, success=True, output="", skip_reason=skip)

        if not transformed:
            return TestResult(example=example, success=True, output="", skip_reason="Transform returned empty")

        # Run based on language
        if example.language == "python":
            success, output, error = run_python(transformed)
        elif example.language in ["typescript", "javascript"]:
            success, output, error = run_javascript(transformed)
        elif example.language in ["bash", "sh"]:
            success, output, error = run_bash(transformed)
        else:
            return TestResult(example=example, success=True, output="", skip_reason=f"Unsupported: {example.language}")

        return TestResult(
            example=example,
            success=success,
            output=output,
            error=error,
            transformed_code=transformed
        )
    except Exception as e:
        return TestResult(
            example=example,
            success=False,
            output="",
            error=f"Transform error: {e}\n{traceback.format_exc()}"
        )


def check_cli_available() -> bool:
    """Check if hindsight CLI is available."""
    try:
        result = subprocess.run(["hindsight", "--version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except:
        return False


def main():
    sys.stdout.reconfigure(line_buffering=True)

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("ERROR: OPENAI_API_KEY required")
        sys.exit(1)

    hindsight_url = os.environ.get("HINDSIGHT_API_URL", "http://localhost:8888")
    model = os.environ.get("DOC_TEST_MODEL", "gpt-4o")

    # Find repo root - go up from script location
    script_path = os.path.abspath(__file__)
    repo_root = os.path.dirname(os.path.dirname(script_path))

    # If running from a subdirectory (like hindsight-api), detect and fix
    if not os.path.exists(os.path.join(repo_root, "hindsight-docs")):
        # Try going up one more level
        repo_root = os.path.dirname(repo_root)
    if not os.path.exists(os.path.join(repo_root, "hindsight-docs")):
        # Fall back to REPO_ROOT env var or cwd
        repo_root = os.environ.get("REPO_ROOT", os.getcwd())

    print(f"Repo: {repo_root}")
    print(f"API: {hindsight_url}")
    print(f"Model: {model}")

    # Check CLI
    cli_available = check_cli_available()
    print(f"CLI: {'available' if cli_available else 'not available'}")

    # Check API health
    try:
        import urllib.request
        urllib.request.urlopen(f"{hindsight_url}/health", timeout=5)
        print("API: healthy")
    except Exception as e:
        print(f"API: WARNING - {e}")

    # Initialize OpenAI client early (needed for transforms and analysis)
    client = OpenAI(api_key=openai_key)

    # Find and extract examples
    md_files = find_markdown_files(repo_root)
    print(f"\nFound {len(md_files)} markdown files")

    all_examples = []
    for md_file in md_files:
        examples = extract_code_blocks(md_file)
        if examples:
            all_examples.extend(examples)

    print(f"Found {len(all_examples)} code examples")

    # Run tests
    report = TestReport()
    max_workers = int(os.environ.get("MAX_WORKERS", "4"))  # Lower default since LLM calls are slower

    print(f"\nRunning tests with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(test_example, ex, client, hindsight_url, cli_available, model): ex for ex in all_examples}
        for future in as_completed(futures):
            result = future.result()
            report.add_result(result)

            status = "SKIP" if result.skip_reason else ("PASS" if result.success else "FAIL")
            safe_print(f"  [{status}] {result.example.file_path}:{result.example.line_number}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"Total: {report.total} | Pass: {report.passed} | Fail: {report.failed} | Skip: {report.skipped}")
    print("=" * 60)

    # Analyze failures with LLM
    failures = [r for r in report.results if not r.success and not r.skip_reason]

    if failures:
        print(f"\n=== Analyzing {len(failures)} failures (parallel) ===")

        doc_bugs = []
        test_issues = []
        results_lock = threading.Lock()
        completed = [0]  # Use list for mutable counter in closure

        def analyze_one(result: TestResult) -> None:
            analysis = analyze_failure(client, result, repo_root, model)
            entry = {
                "file": result.example.file_path,
                "line": result.example.line_number,
                "error": result.error[:200] if result.error else "",
                "analysis": analysis
            }

            with results_lock:
                completed[0] += 1
                idx = completed[0]
                if analysis.get("is_doc_bug", True):
                    doc_bugs.append(entry)
                    safe_print(f"  [{idx}/{len(failures)}] {result.example.file_path}:{result.example.line_number}")
                    safe_print(f"       → DOC BUG: {analysis.get('reason', '')[:50]}")
                else:
                    test_issues.append(entry)
                    safe_print(f"  [{idx}/{len(failures)}] {result.example.file_path}:{result.example.line_number}")
                    safe_print(f"       → Test issue: {analysis.get('reason', '')[:50]}")

        # Run analysis in parallel (limit concurrency to avoid rate limits)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(analyze_one, result) for result in failures]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    safe_print(f"  Analysis error: {e}")

        # Write summary
        print(f"\n=== RESULTS ===")
        print(f"Documentation bugs: {len(doc_bugs)}")
        print(f"Test/CI issues: {len(test_issues)}")

        if doc_bugs:
            print(f"\n--- Documentation Bugs ---")
            for bug in doc_bugs:
                print(f"  {bug['file']}:{bug['line']}")
                print(f"    Reason: {bug['analysis'].get('reason', 'Unknown')}")
                if bug['analysis'].get('fix'):
                    print(f"    Fix: {bug['analysis']['fix']}")

        if test_issues:
            print(f"\n--- Test/CI Issues (not doc bugs) ---")
            for issue in test_issues:
                print(f"  {issue['file']}:{issue['line']}")
                print(f"    Reason: {issue['analysis'].get('reason', 'Unknown')}")

        # Write GitHub summary (include ALL failures for visibility)
        write_summary(report, doc_bugs, test_issues)

        # Exit code based on real doc bugs only
        sys.exit(1 if doc_bugs else 0)
    else:
        print("\nAll tests passed!")
        write_summary(report, [], [])
        sys.exit(0)


def write_summary(report: TestReport, doc_bugs: list, test_issues: list):
    """Write GitHub Actions summary file."""
    with open("/tmp/doc-test-summary.md", "w") as f:
        # Header
        status = "❌" if doc_bugs else "✅"
        f.write(f"# {status} Documentation Test Results\n\n")

        # Summary table
        f.write(f"| Metric | Count |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Total | {report.total} |\n")
        f.write(f"| ✅ Passed | {report.passed} |\n")
        f.write(f"| ❌ Failed | {report.failed} |\n")
        f.write(f"| ⏭️ Skipped | {report.skipped} |\n\n")

        if doc_bugs or test_issues:
            f.write(f"| Category | Count |\n")
            f.write(f"|----------|-------|\n")
            f.write(f"| 🐛 Documentation Bugs | {len(doc_bugs)} |\n")
            f.write(f"| ⚠️ Test/CI Issues | {len(test_issues)} |\n\n")

        # Documentation bugs section
        if doc_bugs:
            f.write(f"## 🐛 Documentation Bugs ({len(doc_bugs)})\n\n")
            f.write("These are real issues in the documentation that need to be fixed:\n\n")
            for bug in doc_bugs:
                file_short = bug['file'].split('/hindsight/')[-1] if '/hindsight/' in bug['file'] else bug['file']
                f.write(f"### `{file_short}:{bug['line']}`\n")
                f.write(f"- **Issue**: {bug['analysis'].get('reason', 'Unknown')}\n")
                if bug['analysis'].get('fix'):
                    f.write(f"- **Suggested Fix**: {bug['analysis']['fix']}\n")
                if bug.get('error'):
                    f.write(f"- **Error**: `{bug['error'][:150]}...`\n")
                f.write("\n")

        # Test/CI issues section
        if test_issues:
            f.write(f"## ⚠️ Test/CI Issues ({len(test_issues)})\n\n")
            f.write("These failures are NOT documentation bugs - they're issues with the test setup or CI environment:\n\n")
            for issue in test_issues:
                file_short = issue['file'].split('/hindsight/')[-1] if '/hindsight/' in issue['file'] else issue['file']
                f.write(f"### `{file_short}:{issue['line']}`\n")
                f.write(f"- **Reason**: {issue['analysis'].get('reason', 'Unknown')}\n")
                if issue.get('error'):
                    f.write(f"- **Error**: `{issue['error'][:150]}...`\n")
                f.write("\n")

        # No failures
        if not doc_bugs and not test_issues:
            if report.passed > 0:
                f.write(f"All {report.passed} tests passed! ({report.skipped} skipped)\n")
            else:
                f.write(f"All {report.skipped} examples were skipped (install commands, docker, etc.)\n")


if __name__ == "__main__":
    main()

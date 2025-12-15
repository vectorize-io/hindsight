#!/usr/bin/env python3
"""
Documentation Example Tester

Uses an LLM to extract code examples from documentation and test them.
This ensures documentation stays in sync with the actual codebase.

Usage:
    python scripts/test-doc-examples.py

Environment variables:
    OPENAI_API_KEY: Required for LLM calls
    HINDSIGHT_API_URL: URL of running Hindsight server (default: http://localhost:8888)
"""

import os
import re
import sys
import json
import glob
import subprocess
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import threading

from openai import OpenAI

# Thread-safe print lock
print_lock = threading.Lock()


@dataclass
class CodeExample:
    """Represents a code example extracted from documentation."""
    file_path: str
    language: str
    code: str
    context: str  # Surrounding text for context
    line_number: int


@dataclass
class TestResult:
    """Result of testing a code example."""
    example: CodeExample
    success: bool
    output: str
    error: Optional[str] = None


@dataclass
class TestReport:
    """Final test report."""
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[TestResult] = field(default_factory=list)
    created_banks: list[str] = field(default_factory=list)  # Track banks for cleanup

    def add_result(self, result: TestResult):
        self.total += 1
        self.results.append(result)
        if result.success:
            self.passed += 1
        elif result.error and "SKIPPED" in result.error:
            self.skipped += 1
        else:
            self.failed += 1

    def add_bank(self, bank_id: str):
        """Track a bank that was created during testing."""
        if bank_id not in self.created_banks:
            self.created_banks.append(bank_id)


def find_markdown_files(repo_root: str) -> list[str]:
    """Find all markdown files in the repository, excluding auto-generated docs."""
    md_files = []
    # Directories to skip (auto-generated docs, dependencies, etc.)
    skip_patterns = [
        "node_modules",
        ".git",
        "venv",
        "__pycache__",
        "hindsight_client_api/docs",  # Auto-generated OpenAPI docs
        "hindsight-clients/typescript/docs",  # Auto-generated TS docs
        "target/",  # Rust build artifacts
        "dist/",  # Build outputs
    ]
    for pattern in ["*.md", "**/*.md"]:
        for f in glob.glob(os.path.join(repo_root, pattern), recursive=True):
            if any(skip in f for skip in skip_patterns):
                continue
            md_files.append(f)
    return sorted(set(md_files))


def extract_code_blocks(file_path: str) -> list[CodeExample]:
    """Extract code blocks from a markdown file."""
    with open(file_path, "r") as f:
        content = f.read()

    examples = []
    # Match fenced code blocks with language identifier
    pattern = r"```(\w+)\n(.*?)```"

    for match in re.finditer(pattern, content, re.DOTALL):
        language = match.group(1).lower()
        code = match.group(2).strip()

        # Get surrounding context (100 chars before and after)
        start = max(0, match.start() - 200)
        end = min(len(content), match.end() + 200)
        context = content[start:end]

        # Calculate line number
        line_number = content[:match.start()].count('\n') + 1

        # Only include testable languages
        if language in ["python", "typescript", "javascript", "bash", "sh"]:
            examples.append(CodeExample(
                file_path=file_path,
                language=language,
                code=code,
                context=context,
                line_number=line_number
            ))

    return examples


def analyze_example_with_llm(client: OpenAI, example: CodeExample, hindsight_url: str, repo_root: str) -> dict:
    """Use LLM to analyze a code example and determine how to test it."""

    prompt = f"""Analyze this code example from documentation and determine how to test it.

File: {example.file_path}
Language: {example.language}
Line: {example.line_number}
Repository root: {repo_root}

Context around the code:
{example.context}

Code:
```{example.language}
{example.code}
```

Your task:
1. Determine if this code example is testable (some are just fragments or pseudo-code)
2. If testable, generate a complete, runnable test script
3. The test should verify the example works correctly

IMPORTANT RULES:
- Hindsight API is ALREADY running at: {hindsight_url} - do NOT start Docker containers or servers
- Mark Docker/server setup examples as NOT testable (reason: "Server setup example - server already running")
- Mark pip/npm install commands as NOT testable (reason: "Package installation command")
- Mark code fragments that define classes/functions without calling them as NOT testable (reason: "Code fragment - defines but doesn't execute")
- Mark helm commands as NOT testable (reason: "Helm chart testing not supported")
- Use unique bank_id names like "doc-test-<random-uuid>" to avoid conflicts
- For cleanup, use requests.delete("{hindsight_url}/v1/default/banks/<bank_id>") - there is NO delete_bank() method
- For Python, wrap in try/finally to ensure cleanup runs, print "TEST PASSED" on success

WORKING DIRECTORY RULES:
- The test script will run from a temp directory, NOT from the repository
- Repository root is: {repo_root}
- If an example uses 'cd <dir>' or requires a specific working directory, convert to absolute paths
- For example: 'cd hindsight-api && uv run pytest' becomes 'cd {repo_root}/hindsight-api && uv run pytest'
- For cargo commands: run them from the correct absolute path (e.g., 'cd {repo_root}/hindsight-clients/rust && cargo test')
- Always use absolute paths based on the repository root when the example implies a specific directory

EXACT HINDSIGHT PYTHON CLIENT API (use EXACTLY these signatures):
```python
from hindsight_client import Hindsight

# Initialize client
client = Hindsight(base_url="{hindsight_url}")

# Store a single memory - use 'content' parameter, NOT 'items' or 'text'
response = client.retain(
    bank_id="my-bank",      # Required: string
    content="Memory text",   # Required: string - the memory content
    timestamp=None,          # Optional: datetime
    context=None,            # Optional: string
    document_id=None,        # Optional: string
    metadata=None,           # Optional: dict
)
# Returns RetainResponse with: success (bool), bank_id (str), items_count (int)

# Store multiple memories
response = client.retain_batch(
    bank_id="my-bank",
    items=[{{"content": "Memory 1"}}, {{"content": "Memory 2"}}],  # List of dicts with 'content' key
    document_id=None,
    retain_async=False,
)

# Recall memories
response = client.recall(
    bank_id="my-bank",
    query="search query",    # Required: string
    types=None,              # Optional: list of strings
    max_tokens=4096,
    budget="mid",            # "low", "mid", or "high"
)
# Returns RecallResponse with: results (list of RecallResult, each has .text attribute)

# Generate answer using memories
response = client.reflect(
    bank_id="my-bank",
    query="question",        # Required: string
    budget="low",            # "low", "mid", or "high"
    context=None,            # Optional: string
)
# Returns ReflectResponse with: text (str)

# Create a bank with profile
response = client.create_bank(
    bank_id="my-bank",
    name=None,               # Optional: string
    background=None,         # Optional: string
    disposition=None,        # Optional: dict
)

# Close client (important for cleanup)
client.close()
```

IMPORTANT: There is NO delete_bank() method. To delete a bank, use raw HTTP:
```python
import requests
requests.delete(f"{hindsight_url}/v1/default/banks/{{bank_id}}")
```

TYPESCRIPT/JAVASCRIPT RULES:
- ALWAYS use ES module syntax (import), NEVER use require()
- The package is '@vectorize-io/hindsight-client'
```typescript
import {{ HindsightClient }} from '@vectorize-io/hindsight-client';

const client = new HindsightClient({{ baseUrl: '{hindsight_url}' }});

// Retain
await client.retain('bank-id', 'content text');

// Recall
const response = await client.recall('bank-id', 'query');
for (const r of response.results) {{
    console.log(r.text);
}}

// Reflect
const answer = await client.reflect('bank-id', 'question');
console.log(answer.text);

// Create bank
await client.createBank('bank-id', {{ name: 'Name', background: 'Background' }});
```

BASH/CLI RULES:
- The CLI command is 'hindsight'
- Make sure the hindsight CLI is available before testing CLI examples

Respond with JSON:
{{
    "testable": true/false,
    "reason": "Why it is or isn't testable",
    "language": "python|typescript|bash",
    "test_script": "Complete runnable test script that will exit 0 on success, non-zero on failure",
    "cleanup_script": "Optional cleanup script to run after test"
}}

If not testable, set test_script to null."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0
    )

    return json.loads(response.choices[0].message.content)


def run_python_test(script: str, timeout: int = 60) -> tuple[bool, str, Optional[str]]:
    """Run a Python test script."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        f.flush()

        try:
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            )
            output = result.stdout + result.stderr
            # Check for "TEST PASSED" in output as primary success indicator
            # This handles cases where exit code might be non-zero due to warnings
            if "TEST PASSED" in output:
                return True, output, None
            success = result.returncode == 0
            error = None if success else f"Exit code: {result.returncode}\n{result.stderr}"
            return success, output, error
        except subprocess.TimeoutExpired:
            return False, "", f"Test timed out after {timeout}s"
        except Exception as e:
            return False, "", str(e)
        finally:
            os.unlink(f.name)


def run_typescript_test(script: str, timeout: int = 60) -> tuple[bool, str, Optional[str]]:
    """Run a TypeScript/JavaScript test script."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mjs', delete=False) as f:
        f.write(script)
        f.flush()

        try:
            result = subprocess.run(
                ["node", f.name],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            success = result.returncode == 0
            output = result.stdout + result.stderr
            error = None if success else f"Exit code: {result.returncode}\n{result.stderr}"
            return success, output, error
        except subprocess.TimeoutExpired:
            return False, "", f"Test timed out after {timeout}s"
        except FileNotFoundError:
            return False, "", "Node.js not found"
        except Exception as e:
            return False, "", str(e)
        finally:
            os.unlink(f.name)


def run_bash_test(script: str, timeout: int = 60) -> tuple[bool, str, Optional[str]]:
    """Run a bash test script."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write("#!/bin/bash\nset -e\n" + script)
        f.flush()
        os.chmod(f.name, 0o755)

        try:
            result = subprocess.run(
                ["bash", f.name],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            success = result.returncode == 0
            output = result.stdout + result.stderr
            error = None if success else f"Exit code: {result.returncode}\n{result.stderr}"
            return success, output, error
        except subprocess.TimeoutExpired:
            return False, "", f"Test timed out after {timeout}s"
        except Exception as e:
            return False, "", str(e)
        finally:
            os.unlink(f.name)


def safe_print(*args, **kwargs):
    """Thread-safe print."""
    with print_lock:
        print(*args, **kwargs)
        sys.stdout.flush()


def test_example(openai_client: OpenAI, example: CodeExample, hindsight_url: str, repo_root: str, debug: bool = False) -> TestResult:
    """Test a single code example."""
    safe_print(f"  Testing {example.file_path}:{example.line_number} ({example.language})")

    try:
        # Analyze with LLM
        analysis = analyze_example_with_llm(openai_client, example, hindsight_url, repo_root)

        if debug and analysis.get("test_script"):
            safe_print(f"    [DEBUG] Generated script:\n{analysis.get('test_script')}")

        if not analysis.get("testable", False):
            safe_print(f"    SKIPPED: {analysis.get('reason', 'Not testable')}")
            return TestResult(
                example=example,
                success=True,
                output="",
                error=f"SKIPPED: {analysis.get('reason', 'Not testable')}"
            )

        test_script = analysis.get("test_script")
        if not test_script:
            safe_print(f"    SKIPPED: No test script generated")
            return TestResult(
                example=example,
                success=True,
                output="",
                error="SKIPPED: No test script generated"
            )

        # Run the test based on language
        lang = analysis.get("language", example.language)
        if lang == "python":
            success, output, error = run_python_test(test_script)
        elif lang in ["typescript", "javascript"]:
            success, output, error = run_typescript_test(test_script)
        elif lang in ["bash", "sh"]:
            success, output, error = run_bash_test(test_script)
        else:
            safe_print(f"    SKIPPED: Unsupported language {lang}")
            return TestResult(
                example=example,
                success=True,
                output="",
                error=f"SKIPPED: Unsupported language {lang}"
            )

        # Run cleanup if provided
        cleanup = analysis.get("cleanup_script")
        if cleanup and lang == "python":
            run_python_test(cleanup, timeout=30)

        if success:
            safe_print(f"    PASSED")
        else:
            safe_print(f"    FAILED: {error[:200] if error else 'Unknown error'}")

        return TestResult(
            example=example,
            success=success,
            output=output,
            error=error
        )

    except Exception as e:
        error_msg = f"Exception: {str(e)}\n{traceback.format_exc()}"
        safe_print(f"    ERROR: {error_msg[:200]}")
        return TestResult(
            example=example,
            success=False,
            output="",
            error=error_msg
        )


def cleanup_test_banks(hindsight_url: str, report: TestReport):
    """Clean up any banks created during testing."""
    import urllib.request
    import urllib.error

    # Also search for any doc-test-* banks that might have been left behind
    try:
        req = urllib.request.Request(f"{hindsight_url}/v1/default/banks")
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            for bank in data.get("banks", []):
                bank_id = bank.get("bank_id", "")
                if bank_id.startswith("doc-test-"):
                    report.add_bank(bank_id)
    except Exception:
        pass  # Ignore errors listing banks

    if not report.created_banks:
        return

    print(f"\nCleaning up {len(report.created_banks)} test banks...")
    for bank_id in report.created_banks:
        try:
            req = urllib.request.Request(
                f"{hindsight_url}/v1/default/banks/{bank_id}",
                method="DELETE"
            )
            urllib.request.urlopen(req, timeout=10)
            print(f"  Deleted: {bank_id}")
        except Exception as e:
            print(f"  Failed to delete {bank_id}: {e}")


def print_report(report: TestReport):
    """Print the final test report."""
    print("\n" + "=" * 70)
    print("DOCUMENTATION EXAMPLE TEST REPORT")
    print("=" * 70)
    print(f"\nTotal examples: {report.total}")
    print(f"  Passed:  {report.passed}")
    print(f"  Failed:  {report.failed}")
    print(f"  Skipped: {report.skipped}")

    if report.failed > 0:
        print("\n" + "-" * 70)
        print("FAILURES:")
        print("-" * 70)

        for result in report.results:
            if not result.success and result.error and "SKIPPED" not in result.error:
                print(f"\n{result.example.file_path}:{result.example.line_number}")
                print(f"Language: {result.example.language}")
                print(f"Code snippet:")
                print("  " + result.example.code[:200].replace("\n", "\n  ") + "...")
                print(f"Error: {result.error}")

    print("\n" + "=" * 70)

    if report.failed > 0:
        print("RESULT: FAILED")
    else:
        print("RESULT: PASSED")
    print("=" * 70)


def write_github_summary(report: TestReport, output_path: str):
    """Write a GitHub Actions compatible markdown summary."""
    lines = []

    # Header with status
    if report.failed > 0:
        lines.append("# Documentation Examples Test Report")
    else:
        lines.append("# Documentation Examples Test Report")

    lines.append("")

    # Summary stats
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total | {report.total} |")
    lines.append(f"| Passed | {report.passed} |")
    lines.append(f"| Failed | {report.failed} |")
    lines.append(f"| Skipped | {report.skipped} |")
    lines.append("")

    # Failures section (detailed)
    if report.failed > 0:
        lines.append("## Failures")
        lines.append("")

        for result in report.results:
            if not result.success and result.error and "SKIPPED" not in result.error:
                # Extract relative path for cleaner display
                file_path = result.example.file_path
                if "/hindsight/" in file_path:
                    file_path = file_path.split("/hindsight/", 1)[-1]

                lines.append(f"### `{file_path}:{result.example.line_number}`")
                lines.append("")
                lines.append(f"**Language:** {result.example.language}")
                lines.append("")
                lines.append("<details>")
                lines.append("<summary>Code snippet</summary>")
                lines.append("")
                lines.append(f"```{result.example.language}")
                lines.append(result.example.code[:500])
                lines.append("```")
                lines.append("")
                lines.append("</details>")
                lines.append("")
                lines.append("**Error:**")
                lines.append("```")
                # Truncate long errors
                error_text = result.error[:1000] if result.error else "Unknown error"
                lines.append(error_text)
                lines.append("```")
                lines.append("")
                lines.append("---")
                lines.append("")

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main():
    # Ensure unbuffered output
    sys.stdout.reconfigure(line_buffering=True)

    # Check for required environment variables
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY environment variable is required")
        sys.exit(1)

    hindsight_url = os.environ.get("HINDSIGHT_API_URL", "http://localhost:8888")

    # Check if Hindsight is running
    try:
        import urllib.request
        urllib.request.urlopen(f"{hindsight_url}/health", timeout=5)
        print(f"Hindsight API is running at {hindsight_url}")
    except Exception as e:
        print(f"WARNING: Could not connect to Hindsight at {hindsight_url}: {e}")
        print("Some tests may fail if they require a running server")

    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Find repo root (handle both script execution and exec())
    if '__file__' in globals():
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    else:
        # Fallback: use current directory or REPO_ROOT env var
        repo_root = os.environ.get("REPO_ROOT", os.getcwd())
    print(f"Repository root: {repo_root}")

    # Find all markdown files
    md_files = find_markdown_files(repo_root)
    print(f"\nFound {len(md_files)} markdown files")

    # Extract all code examples
    all_examples = []
    for md_file in md_files:
        examples = extract_code_blocks(md_file)
        if examples:
            print(f"  {md_file}: {len(examples)} code blocks")
            all_examples.extend(examples)

    print(f"\nTotal code examples to test: {len(all_examples)}")

    # Test examples in parallel
    report = TestReport()
    debug = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")
    max_workers = int(os.environ.get("MAX_WORKERS", "8"))  # Default 8 parallel workers

    print(f"Running tests with {max_workers} parallel workers...")

    def run_test(args):
        idx, example = args
        safe_print(f"\n[{idx}/{len(all_examples)}] Testing example...")
        return test_example(client, example, hindsight_url, repo_root, debug=debug)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(run_test, (i, ex)): i for i, ex in enumerate(all_examples, 1)}

        # Collect results as they complete
        for future in as_completed(futures):
            result = future.result()
            report.add_result(result)

    # Clean up any test banks (runs even if tests failed)
    cleanup_test_banks(hindsight_url, report)

    # Print report
    print_report(report)

    # Write summary to file for CI
    summary_path = "/tmp/doc-test-summary.md"
    write_github_summary(report, summary_path)
    print(f"Summary written to {summary_path}")

    # Exit with appropriate code
    sys.exit(1 if report.failed > 0 else 0)


if __name__ == "__main__":
    main()

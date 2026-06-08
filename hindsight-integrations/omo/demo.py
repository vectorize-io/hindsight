#!/usr/bin/env python3
"""Interactive demo of the OMO Hindsight integration against the local dev server.

Usage:
  1. Start the Hindsight dev server:  ./scripts/dev/start-api.sh
  2. Run this demo:                   python hindsight-integrations/omo/demo.py

This simulates OMO's hook lifecycle:
  SessionStart → UserPromptSubmit (recall) → Stop (retain) → SessionEnd

The demo uses http://localhost:8888 (or HINDSIGHT_API_URL) and creates a bank called "omo-demo".
"""

import importlib.util
import io
import json
import os
import sys
import tempfile
import time

# Point to the scripts directory
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
sys.path.insert(0, SCRIPTS_DIR)

# Demo config
API_URL = os.environ.get("HINDSIGHT_API_URL", "http://localhost:8888")
BANK_ID = "omo-demo"
SESSION_ID = f"demo-{int(time.time())}"

# Colors for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def banner(text):
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}\n")


def step(num, text):
    print(f"{BOLD}{GREEN}[Step {num}]{RESET} {text}")


def info(text):
    print(f"  {YELLOW}→{RESET} {text}")


def error(text):
    print(f"  {RED}✗{RESET} {text}")


def success(text):
    print(f"  {GREEN}✓{RESET} {text}")


def create_settings(tmp_dir):
    """Create a test settings.json pointing at local dev server."""
    settings = {
        "hindsightApiUrl": API_URL,
        "hindsightApiToken": None,
        "bankId": BANK_ID,
        "bankMission": "You are a demo OMO agent. Focus on technical discussions and coding patterns.",
        "retainMission": "Extract technical decisions and preferences.",
        "autoRecall": True,
        "autoRetain": True,
        "retainMode": "full-session",
        "recallBudget": "mid",
        "recallMaxTokens": 1024,
        "recallTypes": ["world", "experience"],
        "recallContextTurns": 1,
        "recallMaxQueryChars": 800,
        "recallRoles": ["user", "assistant"],
        "recallPromptPreamble": "Relevant memories from past conversations:",
        "retainRoles": ["user", "assistant"],
        "retainEveryNTurns": 1,
        "retainOverlapTurns": 2,
        "retainToolCalls": False,
        "retainTags": ["{session_id}"],
        "retainMetadata": {},
        "retainContext": "omo",
        "recallAdditionalBanks": [],
        "bankIdPrefix": "",
        "dynamicBankId": False,
        "resolveWorktrees": True,
        "directoryBankMap": {},
        "requestTimeoutSeconds": None,
        "debug": True,
    }
    settings_path = os.path.join(tmp_dir, "settings.json")
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)
    return settings_path


def create_transcript(tmp_dir, messages):
    """Create a JSONL transcript file."""
    transcript_path = os.path.join(tmp_dir, "transcript.jsonl")
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")
    return transcript_path


def run_hook_script(module_name, hook_input, tmp_dir):
    """Run a hook script with the given input, return (stdout, stderr)."""
    os.environ["PLUGIN_ROOT"] = tmp_dir
    os.environ["PLUGIN_DATA"] = tmp_dir

    stdin_data = io.StringIO(json.dumps(hook_input))
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    spec = importlib.util.spec_from_file_location(module_name, os.path.join(SCRIPTS_DIR, f"{module_name}.py"))
    mod = importlib.util.module_from_spec(spec)

    old_stdin, old_stdout, old_stderr = sys.stdin, sys.stdout, sys.stderr
    try:
        sys.stdin = stdin_data
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        spec.loader.exec_module(mod)
        mod.main()
    finally:
        sys.stdin, sys.stdout, sys.stderr = old_stdin, old_stdout, old_stderr

    return stdout_capture.getvalue(), stderr_capture.getvalue()


def check_server():
    """Check if Hindsight server is running."""
    from lib.client import HindsightClient

    client = HindsightClient(API_URL)
    return client.health_check(timeout=3)


def main():
    banner("OMO Hindsight Integration Demo")

    info(f"API URL: {API_URL}")
    info(f"Bank ID: {BANK_ID}")
    info(f"Session: {SESSION_ID}")
    print()

    # Pre-check: is the server running?
    step(0, "Checking Hindsight server...")
    if not check_server():
        error(f"Hindsight server not reachable at {API_URL}")
        error("Start it with: ./scripts/dev/start-api.sh")
        sys.exit(1)
    success("Server is healthy")

    with tempfile.TemporaryDirectory() as tmp_dir:
        settings_path = create_settings(tmp_dir)
        info(f"Settings: {settings_path}")

        # ─── Step 1: Session Start ───
        step(1, "Simulating SessionStart hook...")
        hook_input = {
            "session_id": SESSION_ID,
            "cwd": os.getcwd(),
        }
        stdout, stderr = run_hook_script("session_start", hook_input, tmp_dir)
        if stderr:
            for line in stderr.strip().split("\n"):
                info(f"(debug) {line}")
        success("Session started")

        # ─── Step 2: First Recall (should find nothing yet) ───
        step(2, "Simulating recall (UserPromptSubmit) — expect no memories yet...")
        hook_input = {
            "prompt": "What programming language does the user prefer?",
            "session_id": SESSION_ID,
            "cwd": os.getcwd(),
        }
        stdout, stderr = run_hook_script("recall", hook_input, tmp_dir)
        if stderr:
            for line in stderr.strip().split("\n"):
                info(f"(debug) {line}")
        if stdout.strip():
            data = json.loads(stdout)
            context = data.get("hookSpecificOutput", {}).get("additionalContext", "")
            info(f"Got memories ({len(context)} chars):")
            print(f"    {context[:200]}...")
        else:
            success("No memories found (expected for fresh bank)")

        # ─── Step 3: Retain a conversation ───
        step(3, "Simulating retain (Stop) — storing a conversation...")
        messages = [
            {"role": "user", "content": "I always prefer using Python with type hints for backend services."},
            {
                "role": "assistant",
                "content": "Got it! I'll use Python with type hints for all backend code. Would you like me to also set up mypy for strict type checking?",
            },
            {"role": "user", "content": "Yes, and I prefer FastAPI over Flask for REST APIs."},
            {
                "role": "assistant",
                "content": "Great choices. FastAPI with Pydantic models gives you automatic validation and OpenAPI docs. I'll use that as the default framework.",
            },
            {"role": "user", "content": "Also, always use uv instead of pip for dependency management."},
            {
                "role": "assistant",
                "content": "Noted — uv for dependency management. It's much faster than pip and handles lockfiles well.",
            },
        ]
        transcript_path = create_transcript(tmp_dir, messages)
        hook_input = {
            "session_id": SESSION_ID,
            "cwd": os.getcwd(),
            "transcript_path": transcript_path,
        }
        stdout, stderr = run_hook_script("retain", hook_input, tmp_dir)
        if stderr:
            for line in stderr.strip().split("\n"):
                info(f"(debug) {line}")
        success("Conversation retained")

        # ─── Step 4: Wait for async processing ───
        step(4, "Waiting for server-side fact extraction (5s)...")
        time.sleep(5)
        success("Done waiting")

        # ─── Step 5: Recall again (should find memories now) ───
        step(5, "Simulating recall — should find memories from step 3...")
        hook_input = {
            "prompt": "What programming language and framework should I use for this project?",
            "session_id": SESSION_ID,
            "cwd": os.getcwd(),
        }
        stdout, stderr = run_hook_script("recall", hook_input, tmp_dir)
        if stderr:
            for line in stderr.strip().split("\n"):
                info(f"(debug) {line}")
        if stdout.strip():
            data = json.loads(stdout)
            context = data.get("hookSpecificOutput", {}).get("additionalContext", "")
            success(f"Got memories! ({len(context)} chars)")
            print()
            print(f"  {CYAN}--- Recalled context ---{RESET}")
            for line in context.split("\n"):
                print(f"  {line}")
            print(f"  {CYAN}--- End context ---{RESET}")
        else:
            info("No memories found yet (extraction may still be processing)")
            info("Try running the demo again — memories persist across runs")

        # ─── Step 6: Retain a second conversation ───
        step(6, "Retaining a second conversation...")
        messages2 = [
            {
                "role": "user",
                "content": "Let's set up the project structure. I want a monorepo with separate packages.",
            },
            {"role": "assistant", "content": "I'll create a monorepo structure. For Python, we can use uv workspaces."},
            {"role": "user", "content": "Use PostgreSQL for the database, and always use Alembic for migrations."},
            {
                "role": "assistant",
                "content": "PostgreSQL + Alembic is a solid combo. I'll set up the migration infrastructure.",
            },
        ]
        session_id_2 = f"demo-{int(time.time())}-2"
        transcript_path_2 = create_transcript(tmp_dir, messages2)
        hook_input = {
            "session_id": session_id_2,
            "cwd": os.getcwd(),
            "transcript_path": transcript_path_2,
        }
        stdout, stderr = run_hook_script("retain", hook_input, tmp_dir)
        if stderr:
            for line in stderr.strip().split("\n"):
                info(f"(debug) {line}")
        success("Second conversation retained")

        # ─── Step 7: Session End ───
        step(7, "Simulating SessionEnd hook...")
        hook_input = {
            "session_id": SESSION_ID,
            "cwd": os.getcwd(),
            "transcript_path": transcript_path,
        }
        stdout, stderr = run_hook_script("session_end", hook_input, tmp_dir)
        if stderr:
            for line in stderr.strip().split("\n"):
                info(f"(debug) {line}")
        success("Session ended")

    banner("Demo Complete!")
    print(f"  Bank '{BANK_ID}' now has memories from 2 conversations.")
    print("  Run this demo again to see recalled memories from past runs.")
    print(f"  API: {API_URL}")
    print()
    print(f"  {BOLD}Try recalling manually:{RESET}")
    print(f"  curl -s -X POST {API_URL}/v1/default/banks/{BANK_ID}/memories/recall \\")
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"query": "What tools does the user prefer?", "max_tokens": 512}\'')
    print()


if __name__ == "__main__":
    main()

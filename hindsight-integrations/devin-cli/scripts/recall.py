#!/usr/bin/env python3
"""Auto-recall hook for UserPromptSubmit.

Port of the Claude Code plugin's recall.py, adapted for Devin CLI hooks:
Devin CLI's UserPromptSubmit stdin only carries `prompt` (plus the universal
`session_id`/`prompt_id`) — no `transcript_path`, no `cwd`. Multi-turn context
(recallContextTurns > 1) is read from Devin CLI's local session database via
`session_id` instead of a transcript file — see lib/devin_transcript.py.

Flow:
  1. Read hook input from stdin (prompt, session_id)
  2. Resolve API URL (external, existing local, or auto-start daemon)
  3. Derive bank ID (static or dynamic from project context)
  4. Ensure bank mission is set (first use only)
  5. Compose multi-turn query if recallContextTurns > 1
  6. Truncate to recallMaxQueryChars
  7. Call Hindsight recall API
  8. Format memories and output hookSpecificOutput.additionalContext
  9. Save last recall to state (for diagnostics)

Exit codes:
  0 — always (graceful degradation on any error)
"""

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.bank import derive_bank_id, ensure_bank_mission
from lib.client import HindsightClient
from lib.config import debug_log, load_config
from lib.content import (
    compose_recall_query,
    format_current_time,
    format_memories,
    truncate_recall_query,
)
from lib.daemon import get_api_url
from lib.devin_transcript import read_session_messages
from lib.state import write_state

LAST_RECALL_STATE = "last_recall.json"


def filter_by_min_scores(results: list[dict], min_scores: dict, config: dict) -> list[dict]:
    """Drop recall results whose numeric scores are below configured floors."""
    if not min_scores:
        return results
    # recallMinScores comes straight from user JSON, so it can be any shape. A
    # list or a string reached .items() and raised, which aborted the hook and
    # injected no memories at all — one malformed optional setting silently
    # turning recall off entirely.
    if not isinstance(min_scores, dict):
        debug_log(config, f"Ignoring recallMinScores: expected an object, got {type(min_scores).__name__}")
        return results

    floors = {}
    for field, floor in min_scores.items():
        try:
            value = float(floor)
        except (TypeError, ValueError):
            debug_log(config, f"Ignoring invalid recallMinScores floor for '{field}': {floor!r}")
            continue
        # float() accepts "nan" and "inf". A NaN floor makes `value < floor`
        # False for every score, so the floor is silently disabled and results
        # the config meant to reject come through — the opposite of a typo's
        # intent. Treated as invalid, like any other unusable value.
        if not math.isfinite(value):
            debug_log(config, f"Ignoring non-finite recallMinScores floor for '{field}': {floor!r}")
            continue
        floors[field] = value
    if not floors:
        return results

    def passes_floors(result: dict) -> bool:
        scores = result.get("scores") or {}
        for field, floor in floors.items():
            value = scores.get(field)
            if isinstance(value, (int, float)) and value < floor:
                return False
        return True

    before_count = len(results)
    filtered = [result for result in results if passes_floors(result)]
    dropped_count = before_count - len(filtered)
    debug_log(config, f"Score floors dropped {dropped_count}/{before_count} results")
    return filtered


def main():
    config = load_config()

    if not config.get("autoRecall"):
        debug_log(config, "Auto-recall disabled, exiting")
        return

    try:
        hook_input = json.load(sys.stdin)
    # UnicodeDecodeError is what invalid UTF-8 on stdin raises; it subclasses
    # ValueError, not either of the others, so it would escape as a traceback.
    except (json.JSONDecodeError, EOFError, UnicodeDecodeError):
        print("[Hindsight] Failed to read hook input", file=sys.stderr)
        return

    debug_log(config, f"Hook input keys: {list(hook_input.keys())}")

    # `or ""` covers null and an empty string but not a wrong *type*: a truthy
    # non-string prompt reaches .strip() and raises AttributeError. The payload
    # is JSON from another process, so its shape is not ours to assume, and this
    # is the first thing the hook does with it.
    raw_prompt = hook_input.get("prompt")
    prompt = raw_prompt.strip() if isinstance(raw_prompt, str) else ""
    if not prompt or len(prompt) < 5:
        debug_log(config, "Prompt too short for recall, skipping")
        return

    def _dbg(*a):
        debug_log(config, *a)

    try:
        api_url = get_api_url(config, debug_fn=_dbg, allow_daemon_start=False)
    except RuntimeError as e:
        print(f"[Hindsight] {e}", file=sys.stderr)
        return

    api_token = config.get("hindsightApiToken")
    try:
        client = HindsightClient(
            api_url,
            api_token,
            request_timeout_override=config.get("requestTimeoutSeconds"),
        )
    except ValueError as e:
        print(f"[Hindsight] Invalid API URL: {e}", file=sys.stderr)
        return

    bank_id = derive_bank_id(hook_input, config)

    ensure_bank_mission(client, bank_id, config, debug_fn=_dbg)

    recall_context_turns = config.get("recallContextTurns", 1)
    recall_max_query_chars = config.get("recallMaxQueryChars", 800)
    recall_roles = config.get("recallRoles", ["user", "assistant"])

    if recall_context_turns > 1:
        session_id = hook_input.get("session_id", "")
        messages = read_session_messages(session_id)
        debug_log(config, f"Multi-turn context: {recall_context_turns} turns, {len(messages)} messages from session db")
        query = compose_recall_query(prompt, messages, recall_context_turns, recall_roles)
    else:
        query = prompt

    query = truncate_recall_query(query, prompt, recall_max_query_chars)

    # truncate_recall_query() treats max_chars <= 0 as "unlimited" and returns
    # early. Repeating the cap unguarded turned the documented "0 disables the
    # limit" setting into query[:0] — an empty query, silently recalling nothing.
    if recall_max_query_chars > 0 and len(query) > recall_max_query_chars:
        query = query[:recall_max_query_chars]

    debug_log(config, f"Recalling from bank '{bank_id}', query length: {len(query)}")

    recall_tags = config.get("recallTags") or None
    tag_groups = config.get("recallTagGroups") or None
    tags_match = config.get("recallTagsMatch") if recall_tags or tag_groups else None
    additional_bank_filters = config.get("recallAdditionalBankFilters") or {}
    # The container gets the same treatment as its entries below: a list here
    # would raise on .get() before any per-bank guard could run.
    if not isinstance(additional_bank_filters, dict):
        debug_log(config, "Ignoring recallAdditionalBankFilters: expected an object")
        additional_bank_filters = {}

    try:
        response = client.recall(
            bank_id=bank_id,
            query=query,
            max_tokens=config.get("recallMaxTokens", 1024),
            budget=config.get("recallBudget", "mid"),
            types=config.get("recallTypes"),
            tags=recall_tags,
            tags_match=tags_match,
            tag_groups=tag_groups,
            timeout=10,
        )
    except Exception as e:
        print(f"[Hindsight] Recall failed: {e}", file=sys.stderr)
        return

    results = response.get("results", [])

    additional_banks = config.get("recallAdditionalBanks", [])
    seen_banks = {bank_id}
    for extra_bank_id in additional_banks:
        if extra_bank_id in seen_banks:
            continue
        seen_banks.add(extra_bank_id)
        extra_filter = additional_bank_filters.get(extra_bank_id, {})
        # Outside the per-bank try below, so a malformed entry here used to
        # abort the whole hook — throwing away memories already recalled from
        # the primary bank. One bad optional filter should cost one bank.
        if not isinstance(extra_filter, dict):
            debug_log(config, f"Ignoring recallAdditionalBankFilters for '{extra_bank_id}': expected an object")
            continue
        extra_tags = extra_filter.get("recallTags", recall_tags) or None
        extra_tag_groups = extra_filter.get("recallTagGroups", tag_groups) or None
        extra_tags_match = extra_filter.get(
            "recallTagsMatch",
            tags_match if extra_tags or extra_tag_groups else None,
        )
        try:
            extra_response = client.recall(
                bank_id=extra_bank_id,
                query=query,
                max_tokens=config.get("recallMaxTokens", 1024),
                budget=config.get("recallBudget", "mid"),
                types=config.get("recallTypes"),
                tags=extra_tags,
                tags_match=extra_tags_match,
                tag_groups=extra_tag_groups,
                timeout=10,
            )
            extra_results = extra_response.get("results", [])
            if extra_results:
                debug_log(config, f"Got {len(extra_results)} memories from additional bank '{extra_bank_id}'")
                results = results + extra_results
        except Exception as e:
            debug_log(config, f"Recall from additional bank '{extra_bank_id}' failed: {e}")

    results = filter_by_min_scores(results, config.get("recallMinScores") or {}, config)

    if not results:
        debug_log(config, "No memories found")
        return

    debug_log(config, f"Injecting {len(results)} memories")

    memories_formatted = format_memories(results)
    preamble = config.get("recallPromptPreamble", "")
    current_time = format_current_time()

    context_message = (
        f"<hindsight_memories>\n"
        f"{preamble}\n"
        f"Current time - {current_time}\n\n"
        f"{memories_formatted}\n"
        f"</hindsight_memories>"
    )

    # Purely diagnostic state — write_state raises on an unwritable state dir,
    # and letting that escape here would abort the hook *after* recall already
    # succeeded but *before* the memories are printed below. The user would get
    # nothing, for a failure that costs only a debug breadcrumb.
    try:
        write_state(
            LAST_RECALL_STATE,
            {
                "context": context_message,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "bank_id": bank_id,
                "result_count": len(results),
            },
        )
    except OSError as e:
        debug_log(config, f"Could not save last recall state: {e}")

    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context_message,
        }
    }
    json.dump(output, sys.stdout)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Hindsight] Unexpected error in recall: {e}", file=sys.stderr)
        # Always 0, as this module's docstring promises. Exiting 2 under debug
        # made a diagnostic flag change control flow: a non-zero hook exit is a
        # blocking error, so turning debug on to investigate a recall failure
        # escalated it from "no memories this turn" to a rejected prompt — and
        # changed the behaviour of the very bug being investigated. The error
        # is already on stderr, which is where a hook reports without blocking.
        sys.exit(0)

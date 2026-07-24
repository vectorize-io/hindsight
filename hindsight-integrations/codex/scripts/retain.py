#!/usr/bin/env python3
"""Auto-retain hook for Stop event.

Fires after each agent turn. Reads the Codex session transcript and stores
the conversation into Hindsight memory for future recall.

Flow:
  1. Read hook input from stdin (session_id, transcript_path, cwd)
  2. Read conversation transcript from transcript_path
  3. Apply chunked retention logic (retainEveryNTurns + overlap window)
  4. Resolve API URL (external, existing local, or auto-start daemon)
  5. Derive bank ID and ensure mission
  6. Format transcript (strip memory tags, filter roles)
  7. POST to Hindsight retain API

Exit codes:
  0 — always (graceful degradation on any error)
"""

import hashlib
import json
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.bank import derive_bank_id, ensure_bank_mission
from lib.client import HindsightClient, auth_context_id
from lib.config import debug_log, load_config
from lib.content import (
    prepare_retention_transcript,
    read_transcript,
    slice_last_turns_by_user_boundary,
)
from lib.daemon import get_api_url, get_intended_api_url
from lib.state import (
    PendingRetain,
    clear_deferred_retain,
    read_deferred_retain,
    read_pending_retains,
    read_retain_cadence,
    retain_submission_lock,
    write_deferred_retain,
    write_pending_retains,
    write_retain_cadence,
)

MAX_PENDING_RETAINS = 128
MAX_PENDING_RETAIN_BYTES = 64 * 1024 * 1024


def _matches_destination(submission, api_url, auth_context, bank_id):
    return (
        submission.get("api_url") == api_url
        and submission.get("auth_context_id") == auth_context
        and submission.get("bank_id") == bank_id
    )


def _queue_size_bytes(submissions):
    return len(json.dumps(submissions, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _build_request_args(
    config,
    transcript_path,
    session_id,
    bank_id,
    retain_mode,
    retain_every_n,
    include_tool_calls,
):
    """Read the latest transcript and freeze one retain request."""
    all_messages = read_transcript(transcript_path, include_tool_calls=include_tool_calls)
    if not all_messages:
        debug_log(config, "No messages in transcript, skipping current retain")
        return None, None

    debug_log(config, f"Read {len(all_messages)} messages from transcript")
    messages_to_retain = all_messages
    if retain_mode == "chunked" and retain_every_n > 1:
        overlap_turns = config.get("retainOverlapTurns", 0)
        window_turns = retain_every_n + overlap_turns
        messages_to_retain = slice_last_turns_by_user_boundary(all_messages, window_turns)
        debug_log(
            config,
            f"Chunked retain firing (window: {window_turns} turns, {len(messages_to_retain)} messages)",
        )
    else:
        debug_log(config, f"Full session retain: {len(all_messages)} messages")

    retain_roles = config.get("retainRoles", ["user", "assistant"])
    transcript, message_count = prepare_retention_transcript(
        messages_to_retain,
        retain_roles,
        True,
        include_tool_calls=include_tool_calls,
    )
    if not transcript:
        debug_log(config, "Empty transcript after formatting, skipping current retain")
        return None, None

    if retain_mode == "chunked" and retain_every_n > 1:
        document_id = f"{session_id}-{int(time.time() * 1000)}"
    else:
        document_id = session_id

    template_vars = {
        "session_id": session_id,
        "bank_id": bank_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    def resolve_template(value):
        for key, replacement in template_vars.items():
            value = value.replace(f"{{{key}}}", replacement)
        return value

    raw_tags = config.get("retainTags", [])
    tags = [resolve_template(tag) for tag in raw_tags] if raw_tags else None
    metadata = {
        "retained_at": template_vars["timestamp"],
        "message_count": str(message_count),
        "session_id": session_id,
    }
    for key, value in config.get("retainMetadata", {}).items():
        metadata[key] = resolve_template(str(value))

    debug_log(
        config,
        f"Retaining to bank '{bank_id}', doc '{document_id}', {message_count} messages, {len(transcript)} chars",
    )
    if tags:
        debug_log(config, f"Tags: {tags}")

    source_fingerprint = hashlib.sha256(
        json.dumps(all_messages, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    return (
        {
            "bank_id": bank_id,
            "content": transcript,
            "document_id": document_id,
            "context": config.get("retainContext", "codex"),
            "metadata": metadata,
            "tags": tags,
        },
        source_fingerprint,
    )


def main():
    config = load_config()

    if not config.get("autoRetain"):
        debug_log(config, "Auto-retain disabled, exiting")
        return

    # Read hook input from stdin
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        print("[Hindsight] Failed to read hook input", file=sys.stderr)
        return

    debug_log(config, f"Stop hook input keys: {list(hook_input.keys())}")

    session_id = hook_input.get("session_id", "unknown")
    transcript_path = hook_input.get("transcript_path", "")
    include_tool_calls = config.get("retainToolCalls", True)
    retain_mode = config.get("retainMode", "full-session")
    retain_every_n = max(1, config.get("retainEveryNTurns", 1))

    def _dbg(*a):
        debug_log(config, *a)

    api_token = config.get("hindsightApiToken")
    bank_id = derive_bank_id(hook_input, config)
    expected_auth_context = auth_context_id(api_token)
    intended_api_url = get_intended_api_url(config)

    # Serialize one session's checkpoint, lost-ack replay, and current
    # submission. Other sessions remain fully concurrent.
    with retain_submission_lock(session_id) as lock_acquired:
        if not lock_acquired:
            print(
                "[Hindsight] Retain state is busy; leaving this Stop event for the next retry",
                file=sys.stderr,
            )
            return
        request_args, source_fingerprint = _build_request_args(
            config,
            transcript_path,
            session_id,
            bank_id,
            retain_mode,
            retain_every_n,
            include_tool_calls,
        )
        try:
            pending_submissions = read_pending_retains(session_id)
            deferred_submission = read_deferred_retain(session_id)
            cadence = read_retain_cadence(session_id)
        except (OSError, ValueError) as e:
            print(
                f"[Hindsight] Retain state is unreadable: {e}; leaving it untouched for operator recovery",
                file=sys.stderr,
            )
            return
        durable_submissions = [
            *pending_submissions,
            *([deferred_submission] if deferred_submission is not None else []),
        ]
        if durable_submissions and not all(
            _matches_destination(
                submission,
                intended_api_url,
                expected_auth_context,
                bank_id,
            )
            for submission in durable_submissions
        ):
            print(
                "[Hindsight] Pending retain belongs to a different API, authentication context, or bank; "
                "leaving it untouched for operator recovery",
                file=sys.stderr,
            )
            return

        if request_args is not None and deferred_submission is None:
            matching_pending = next(
                (
                    submission
                    for submission in pending_submissions
                    if submission.get("source_fingerprint") == source_fingerprint
                ),
                None,
            )
            already_counted = cadence["source_fingerprint"] == source_fingerprint
            if matching_pending is not None:
                turn_count = matching_pending["turn_count"]
                if not already_counted:
                    try:
                        write_retain_cadence(
                            session_id,
                            {
                                "turn_count": turn_count,
                                "source_fingerprint": source_fingerprint,
                            },
                        )
                    except Exception as e:
                        print(f"[Hindsight] Failed to recover retain cadence: {e}", file=sys.stderr)
                        return
            elif already_counted:
                turn_count = cadence["turn_count"]
            else:
                turn_count = (
                    max([cadence["turn_count"]] + [submission["turn_count"] for submission in pending_submissions]) + 1
                )
                if turn_count % retain_every_n == 0:
                    submission = {
                        "idempotency_key": uuid.uuid4().hex,
                        "bank_id": bank_id,
                        "api_url": intended_api_url,
                        "auth_context_id": expected_auth_context,
                        "source_fingerprint": source_fingerprint,
                        "turn_count": turn_count,
                        "request": request_args,
                        "post_attempted": False,
                    }
                    candidate_submissions = list(pending_submissions)
                    if (
                        retain_mode == "full-session"
                        and candidate_submissions
                        and not candidate_submissions[-1].get("operation_id")
                        and candidate_submissions[-1].get("post_attempted", True) is False
                    ):
                        candidate_submissions[-1] = submission
                    else:
                        candidate_submissions.append(submission)
                    if (
                        len(candidate_submissions) > MAX_PENDING_RETAINS
                        or _queue_size_bytes(candidate_submissions) > MAX_PENDING_RETAIN_BYTES
                    ):
                        deferred_submission = submission
                        try:
                            write_deferred_retain(session_id, submission)
                        except Exception as e:
                            print(
                                f"[Hindsight] Failed to persist deferred retain submission: {e}",
                                file=sys.stderr,
                            )
                            return
                        print(
                            "[Hindsight] Pending retain queue limit reached; "
                            "draining acknowledged work before retrying the current transcript",
                            file=sys.stderr,
                        )
                    else:
                        pending_submissions = candidate_submissions
                        try:
                            write_pending_retains(session_id, pending_submissions)
                        except Exception as e:
                            print(f"[Hindsight] Failed to persist retain submission: {e}", file=sys.stderr)
                            return
                else:
                    next_at = ((turn_count // retain_every_n) + 1) * retain_every_n
                    debug_log(
                        config,
                        f"Turn {turn_count}/{retain_every_n}, skipping retain (next at turn {next_at})",
                    )
                if deferred_submission is None:
                    try:
                        write_retain_cadence(
                            session_id,
                            {
                                "turn_count": turn_count,
                                "source_fingerprint": source_fingerprint,
                            },
                        )
                    except Exception as e:
                        print(f"[Hindsight] Failed to persist retain cadence: {e}", file=sys.stderr)
                        return

        if not pending_submissions and deferred_submission is None:
            return

        try:
            api_url = get_api_url(config, debug_fn=_dbg, allow_daemon_start=True)
            client = HindsightClient(api_url, api_token)
        except (RuntimeError, ValueError) as e:
            print(f"[Hindsight] {e}", file=sys.stderr)
            return

        try:
            supports_durable_delivery = client.supports_durable_retain_delivery()
        except Exception as e:
            print(
                f"[Hindsight] Could not verify durable Retain support: {e}; leaving queued work untouched",
                file=sys.stderr,
            )
            return
        if not supports_durable_delivery:
            print(
                "[Hindsight] Server does not advertise idempotent serialized Retain support; "
                "leaving queued work untouched",
                file=sys.stderr,
            )
            return

        if any(
            not _matches_destination(
                submission,
                client.api_url,
                client.auth_context_id,
                bank_id,
            )
            for submission in [
                *pending_submissions,
                *([deferred_submission] if deferred_submission is not None else []),
            ]
        ):
            print(
                "[Hindsight] Pending retain belongs to a different API, authentication context, or bank; "
                "leaving it untouched for operator recovery",
                file=sys.stderr,
            )
            return

        ensure_bank_mission(client, bank_id, config, debug_fn=_dbg)

        while True:
            processing_deferred = not pending_submissions and deferred_submission is not None
            if not pending_submissions and not processing_deferred:
                return

            pending_submission = deferred_submission if processing_deferred else pending_submissions[0]
            if pending_submission is None:
                return
            try:
                if pending_submission.get("post_attempted", True) is False:
                    pending_submission["post_attempted"] = True
                    if processing_deferred:
                        write_deferred_retain(session_id, pending_submission)
                    else:
                        write_pending_retains(session_id, pending_submissions)
                response = client.retain(
                    **pending_submission["request"],
                    idempotency_key=pending_submission["idempotency_key"],
                    timeout=15,
                )
                operation_id = response.get("operation_id")
                if not operation_id:
                    raise RuntimeError("retain response did not include operation_id")
            except Exception as e:
                print(f"[Hindsight] Pending retain submission failed: {e}", file=sys.stderr)
                return

            debug_log(config, f"Retain response: {json.dumps(response)[:200]}")
            try:
                if processing_deferred:
                    write_retain_cadence(
                        session_id,
                        {
                            "turn_count": pending_submission["turn_count"],
                            "source_fingerprint": pending_submission["source_fingerprint"],
                        },
                    )
                    clear_deferred_retain(session_id)
                    deferred_submission = None
                else:
                    remaining_submissions = pending_submissions[1:]
                    write_pending_retains(session_id, remaining_submissions)
                    pending_submissions = remaining_submissions
            except Exception as e:
                print(
                    f"[Hindsight] Retain was acknowledged but retry state could not be updated: {e}; "
                    "leaving the durable submission for an idempotent replay",
                    file=sys.stderr,
                )
                return


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Hindsight] Unexpected error in retain: {e}", file=sys.stderr)
        try:
            from lib.config import load_config

            sys.exit(2 if load_config().get("debug") else 0)
        except Exception:
            sys.exit(0)

"""The memory watcher's rules: what in a session is worth a check, what to ask memory, and when to speak.

A session's client sends what happens (a request, what the agent says, the tools it calls and what
they return). The watcher reads those events as steps and keeps a small running state: the agent's
last steps, the request it is working on, and the moment waiting to be checked. A moment is one of:

- ``task``: a new request (checked for a heads-up: a recorded decision the request does not state);
- ``plan``: the agent saying what it will do;
- ``command``: a command that changes things (reads, searches and test runs are only context);
- ``edit``: a change to a file;
- ``failure``: a command or tool that failed.

Each checked moment is ONE reflect call with ``abstain``: the decision model settles most moments
silently with no LLM call, and a note is written from the memory it picked otherwise. The rules
around that call are the ones the sde-bench watcher measured (33 tasks x 3 runs, every task solved
on the first try): never the same memory twice, at most ``MAX_NOTES`` per request, the same
situation never judged twice, a heads-up does not use its memory up.

Pure functions over plain data: no I/O, so the same code runs in the append request (reading new
events into the state) and in the worker (checking the moment). Ported from the coding-agents
client (``hindsight-integrations/coding-agents/src/core/watch.ts``), where it ran on each machine.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

Trigger = Literal["task", "plan", "command", "edit", "failure"]

# A heads-up needs a request that says what it is about. "/ship-it", "obs cleanup?" or a bare issue link
# searched memory by the command's own words and got a note about whatever shared them: 10 of 37
# irrelevant or wrong notes in beta. The prompt's "reply with nothing" for these was ignored, so it is a
# gate here. The shortest request whose heads-up changed the work had 5 words.
MIN_REQUEST_WORDS = 5
_SLASH_OR_URL = re.compile(r"(?<!\S)/[\w:.-]+|https?://\S+")
_WORK_WORD = re.compile(r"[^\W_][\w'-]*")
# A heads-up abstains when the decision model puts "none of these applies" at or above this, not at
# reflect's own 0.9: in beta every irrelevant heads-up scored 0.5-0.89, every one that changed the
# agent's work 0.41 or less. Conflict checks keep reflect's bar: there the two overlapped.
HEADSUP_ABSTAIN_AT = 0.5
# A note sharing this much of its words with one the agent set aside says the same rule again. Lower
# than the repeat guard's 0.6: the beta rewordings of a set-aside rule measured 0.4-0.55.
SET_ASIDE_OVERLAP = 0.4
SET_ASIDE_KEPT = 20
# Scheduled jobs: 8 of 37 bad notes in beta told a job what its own instruction file already said, or
# what that file makes an exception to, because the check saw only "Read JOB.md and do what it says".
REQUEST_FILE = re.compile(r"[\w./~-]+\.(?:md|txt|ya?ml|json|toml)\b")
REQUEST_FILE_CAP = 8000
# A tool result as the watcher reads it: the client sends up to 8000 chars. At 3000 a job's 8.5 KB
# JOB.md came through cut at line 21, its linking rule (line 59) missing, and a note then contradicted
# that rule; a failure's tail, where the error is, was cut off the same way. Never stored in the state.
RESULT_CAP = 8000
# A session's commands and plans are checked at most this often; the latest one waits for the next
# turn (heaviest, then latest, wins the pending slot). In beta command checks were ~85% of all checks,
# 6 of 35 command notes changed the work and 25 were irrelevant or wrong, and on a laptop their
# reranks queued behind each other into timeouts. Edits, failures and requests are not held back.
COMMAND_CHECK_EVERY_S = 60
# At most this many checks run at once, across sessions: in beta a fan-out of 11 sessions at once
# queued their checks behind the one reranker slot, and 12 of 38 timed out in 8 minutes. A command,
# plan or failure check waits while that many run (its session's latest moment then takes its
# place); an edit or a new request is never held back.
MAX_RUNNING_CHECKS = 3
BUSY_RETRY_S = 10
# Words in a heads-up's "Before you change <X>" that name no code of their own.
_GENERIC_TARGET_WORDS = frozenset(
    "code handling change changes rule rules workflow workflows documentation installation deployment "
    "procedure issue invoke plugin interaction native chat lookup organization admission cleanup side "
    "serve test tests fixtures repair deletion routing development release default defaults".split()
)
# "default" joined after "enable it by default" matched "the dev-alpha multi-loop default" (beta: the
# request meant memlake's indexer loops; the agent set the heads-up aside).
# Per user request, not per session: a benchmark session is one task, but a real one runs for days
# across many requests, and a per-session cap left a session silent for hours after three notes
# (two of them wrong) — 49 checks an hour skipped as "note limit reached" (beta, 2026-10-22).
MAX_NOTES = 3
RECENT_STEPS = 12
JUDGED_KEYS = 200
# Tool calls waiting for their result. Bounded: a client that never sends results must not grow
# the state without end.
OPEN_TOOL_CALLS = 64
# The change so far, as the client last sent it. Bounded the same way the situation cuts it.
DIFF_CAP = 64_000

# A moment's weight when several arrive before the watcher is free: the heaviest, latest one wins.
PRIORITY: dict[str, int] = {"task": 0, "plan": 1, "failure": 2, "command": 3, "edit": 3}

# What counts as a decision, for both questions. In beta the check noted one run's own choice ("no new
# article this run"), the folder a past session ran a command in, and a past status ("not shipped") as
# project rules — each time against the request the agent had been given. And a user's "ok let's
# switch to python then" got a conflict quoting the Rust choice it replaced, two minutes earlier.
# An agent's own suggestion ("the real fix is to sign it with a certificate") came back as the rule
# against the approach the user chose instead. And a command that only edited button labels got a
# conflict about app signing, because the request was about a permission loop: the step, not the
# request, is what a conflict check judges. And "fully support the questions in the chat" got a conflict quoting the
# old "questions go to the terminal" rule it asked to change: a request replaces a rule without naming it.
# Three conflicts applied a rule to the wrong one of two similar parts: a rerank fallback rule to the
# decision model, a consolidation queue limit to the decision model, a memory-scan rule to a document read.
# And three notes in a day held up the last recorded pin of a dependency as a decision against moving it.
# "<that file, module or component, by its name>": the template said "<the code>", and Gemma 4 copied it
# word for word (3 of 4 heads-ups began "Before you change the code:"; 0 of 70 before).
# "speak to me as 'you'": after the switch to Gemma 4, 3 of 11 notes began "The user's manual edit ..."
# about the agent's own edit (1 in 175 before), which reads as if the person made the change.
# What the code does now is not a rule either: in beta a summary line ("text search still reads the bank
# page by page and stops at 25k memories") came back as "retain the rule" against the change the request
# asked for.
DECISION = (
    "A decision is a rule meant to hold for the project from then on, made by a person or recorded in a "
    "commit or a doc. Not a decision: a choice made for one past task or run, which folder, checkout or "
    "branch past work ran in, something an agent did or suggested, the last known status of some work, "
    "what the code does now or a limit it still has ('X still scans page by page'; changing it is ordinary "
    "work), or the commit or version something is pinned to now (moving a pin is ordinary work). "
    "Where the "
    "request or the job's own instructions say to do something, that is not a conflict. The request is "
    "the newest decision: where it changes or replaces a recorded one, even in a few words ('switch to "
    "X'), or asks for the very behaviour a recorded rule ruled out, without naming the rule, the recorded "
    "one no longer holds. I am the agent doing this work: speak to me as 'you', never as 'the user'."
)

# A request that changes nothing got "Before you change <x>" anyway in beta (a cost comparison, a
# watch request, an hourly job whose own file states the rules): 9 of 20 notes the agent called not
# needed. The conflict question still checks what such a request ends up doing. So it does for a
# request that is only a link ("add this to the docs <issue url>"): the check cannot open it, and
# guessed an unrelated docs rule.
HEADSUP_Q = (
    "Before I start on this request: which project decision recorded in memory (an exact value, a rule, "
    "an exception) about exactly the code this request will change must my fix respect, that the request "
    "itself does not state? Only what someone decided (a chosen value or rule, an exception, a rejected "
    "option), not what the code already does: I can read that myself. Tell me in at most 70 words, as "
    "'Before you change <that file, module or component, by its name>: <every such rule the memory's "
    "source text records for this code, "
    "in its final form, not only its summary; each rule a fix could get wrong>', copying "
    "exact literals verbatim, no advice beyond the rules. If memory records no such decision about this "
    "code, reply with nothing. If the request changes nothing (it asks a question, an analysis, a check or "
    "a status, to watch something, or to run a job by its own instructions), reply with nothing. If the "
    "request does not say what it will change (it only links an issue or a page), reply with nothing. " + DECISION
)

CONFLICT_Q = (
    "Does what I am doing right now (my plan, command or code change) contradict a project decision recorded "
    "in memory, miss one, or repeat an option recorded as rejected? If so, tell me in at most 50 words the "
    "recorded value or rule (exact literals verbatim) and what in my work disagrees. How I run, install or "
    "check tests is never a conflict. Judge only the code or files this step changes: a rule about other "
    "code is not a conflict here, even when my request is about it. A rule about one component (a "
    "service, a model, a scan, a queue) is not a rule for a similar one with a shared name. If nothing "
    "conflicts, reply with "
    "nothing. " + DECISION
)

# A tool is known by what its name says, not by one agent's names: Claude Code's Edit/Write/Bash,
# Codex's apply_patch/shell and Hermes's write_file/patch/terminal read the same. (A fixed list of
# Claude Code's names came first; every other client had to rename its tools to be watched.)
EDIT_WORDS = ("edit", "write", "patch", "replace", "insert", "multiedit", "notebookedit")
COMMAND_WORDS = frozenset({"bash", "shell", "sh", "terminal", "exec", "execute", "run", "command", "cmd"})
# Keys a tool's arguments name its target with, across agents.
TARGET_KEYS = ("command", "cmd", "file_path", "path", "notebook_path", "filename", "pattern", "query", "url")
# Keys that carry the new text of an edit, when the client sends no diff.
EDIT_TEXT_KEYS = ("new_string", "content", "patch", "diff", "new_str", "file_text", "input")
# Commands that change nothing, or only run checks: no moment.
# A `cd` may end its line instead of `&&`/`;` (beta: 11 of a session's 14 checks were `cd <repo>` then
# `git grep` / `gh run view` on the next line, each a reflect that found nothing).
READ_ONLY_CMD = re.compile(
    r"^\s*(cd [^;&|\n]+(&&|;|\n)\s*)?(cat|ls|find|grep|rg|head|tail|wc|tree|file|which|pwd|echo|sed -n|awk|"
    r"command -v|git (status|diff|log|show|blame|grep|fetch|rev-parse|ls-files)|"
    r"gh (run (view|list)|pr (view|list|diff|checks)|issue (view|list))|python3? -m pytest|pytest|"
    r"python3? -c|python3? -I -c|pip3? install|npm (test|run test)|npx (vitest|jest))\b"
)
# The sandbox failing, not the project: no moment.
# So is the harness refusing or questioning a command before it ran (approval prompts, quoting
# checks, a browser tool that lost its tab): in beta 55 failure checks in a week, none with a note.
ENV_TROUBLE = re.compile(
    r"command not found|No module named (pytest|pip)|require approval|requires approval"
    r"|can't be checked before it runs|Couldn't determine which page|^Brace expansion"
    r"|Newline followed by # inside a quoted argument"
)
FAILED = re.compile(r"\b\d+ failed\b|Traceback \(most recent call last\)|\bFAILED\b|\bError:")
# How a delivered note reads when the agent's transcript carries it back.
# Notes are numbered since the client names the one it credits ("Note #12 from this project's memory"):
# the plain substring missed every numbered note, leaving only the sent-text match to catch them.
OWN_NOTE_MARK = re.compile(r"Note (#\d+ )?from this project's memory|Hindsight thought #\d+:")


class Pending(BaseModel):
    trigger: Trigger
    text: str = ""
    seq: int = 0  # the log position of the event that made the moment


class WatchState(BaseModel):
    """What the watcher keeps per session, between appends. Stored as the session's ``watch_state``."""

    recent: list[str] = Field(default_factory=list)
    tools: dict[str, tuple[str, dict[str, Any]]] = Field(default_factory=dict)
    pending: Pending | None = None
    goal: str = ""
    # The two requests before the latest, newest last: a follow-up ("ok continue", "open the PR") alone
    # hid what the session was for. In beta a conflict check saw only "ok continue your work" and held
    # an old rule against the change the first request asked for.
    earlier_goals: list[str] = Field(default_factory=list)
    diff: str = ""
    sent: list[str] = Field(default_factory=list)
    evidence_sent: list[str] = Field(default_factory=list)
    judged: list[str] = Field(default_factory=list)
    # The notes (by seq) the agent told the user it used, and the ones it said it did not need.
    # The conflicts already sent (text): the same rule comes back under new memory ids as the
    # session's saves are consolidated, so the id guard alone let it through again.
    conflicts: list[str] = Field(default_factory=list)
    credited: list[int] = Field(default_factory=list)
    dismissed: list[int] = Field(default_factory=list)
    # Notes sent since the user's latest request: what MAX_NOTES caps.
    request_notes: int = 0
    # The notes the agent set aside (their text): the check is told not to raise them again, and a
    # note mostly saying the same is dropped. In beta one rule the agent had set aside twice ("only my
    # suggestion") came back a third time, reworded each time past the repeat guard.
    set_aside: list[str] = Field(default_factory=list)
    # The file the request points to, as the agent read it ("Read JOB.md and do what it says"): a
    # scheduled job's request is one line, so its rules, and the exceptions they make, live there.
    request_file: str = ""
    # When this session's last command or plan was checked (database time, ISO): see COMMAND_CHECK_EVERY_S.
    last_command_check: str = ""


class Step(BaseModel):
    kind: Literal["ask", "say", "tool", "result", "state", "end"]
    text: str = ""
    id: str = ""
    tool: str = ""
    input: dict[str, Any] = Field(default_factory=dict)
    error: bool = False


class Moment(BaseModel):
    trigger: Trigger
    text: str
    recent: list[str]
    diff: str
    seq: int = 0


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n]


def step_of(kind: str, role: str, event: dict[str, Any]) -> Step | None:
    """One session event as a watcher step, or None when the watcher does not read it."""
    text = str(event.get("text") or "")
    if kind == "message":
        if role == "user":
            return Step(kind="ask", text=_clip(text, 4000))
        if role == "assistant" and text:
            return Step(kind="say", text=_clip(text, 3000))
        return None
    if kind == "tool_call":
        raw = event.get("input")
        return Step(
            kind="tool",
            id=str(event.get("call_id") or ""),
            tool=str(event.get("tool") or ""),
            input=raw if isinstance(raw, dict) else {},
        )
    if kind == "tool_result":
        return Step(
            kind="result",
            id=str(event.get("call_id") or ""),
            text=_clip(text, RESULT_CAP),
            error=bool(event.get("is_error")),
        )
    if kind == "state":
        data = event.get("data") or {}
        if isinstance(data, dict) and isinstance(data.get("diff"), str):
            return Step(kind="state", text=data["diff"])
    if kind == "turn_end":
        return Step(kind="end")
    return None


def tool_family(name: str) -> Literal["edit", "command", "other"]:
    # Whole words of the name (`str_replace_editor`, `TodoWrite`, `write_file`), so `dispatch` is no patch.
    words = re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1_\2", name).lower())
    if "todo" in words:  # a to-do list the agent writes is not the project's code
        return "other"
    if any(w.startswith(EDIT_WORDS) for w in words):
        return "edit"
    if COMMAND_WORDS.intersection(words):
        return "command"
    return "other"


def headsup_names_other_code(note: str, request: str) -> bool:
    """Whether a heads-up ("Before you change X: ...") is about code the request never mentions. The
    answer model guesses which code the request will change; in beta every heads-up whose X shared no
    word with the request was irrelevant or only confirmed (10 of 10), while all 11 that changed the
    agent's work named something the request did (a prefix is enough: "cataglog" for "catalog")."""
    m = re.match(r"\s*Before you change (.+?):", note)
    if not m:
        return False
    words = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9]*", m.group(1))} - _GENERIC_TARGET_WORDS
    words = {w for w in words if len(w) >= 4}
    req = request.lower()
    return bool(words) and not any(w[:5] in req for w in words)


def _file_the_request_names(goal: str, target: str) -> str:
    """The file the request names that this tool call read, or ""."""
    for path in REQUEST_FILE.findall(goal):
        if path in target or path.rsplit("/", 1)[-1] in target:
            return path
    return ""


def _target(inp: dict[str, Any]) -> str:
    return str(next((inp[k] for k in TARGET_KEYS if inp.get(k)), ""))


def _edit_text(inp: dict[str, Any]) -> str:
    """An edit in words when there is no diff: the file and its new text."""
    new = next((inp[k] for k in EDIT_TEXT_KEYS if isinstance(inp.get(k), str) and inp[k].strip()), "")
    return f"Change to {_target(inp)}:\n{new}"[:3000] if new else ""


def _is_own_note(state: WatchState, text: str) -> bool:
    return bool(OWN_NOTE_MARK.search(text)) or any(s[:80] in text for s in state.sent if len(s) >= 40)


def read_step(state: WatchState, step: Step, seq: int = 0) -> None:
    """One step into the state: the recent-steps line, and the moment it makes (if any) as pending."""
    said = ""
    trigger: Trigger | None = None
    text = ""
    if step.kind == "state":
        state.diff = step.text[:DIFF_CAP]
        return
    if step.kind == "end":
        # The turn is over, so the assistant's last words describe work it has finished ("All tests
        # pass. I changed..."): there is nothing left for a note to change before the next turn, and
        # the sde-bench measurements showed these closing checks to be most of a session's wasted
        # ones (13 of 77 reflects). A pending edit, command or failure is still checked: its note
        # can be handed over at the end of the turn.
        if state.pending is not None and state.pending.trigger == "plan":
            state.pending = None
        return
    if step.kind == "ask":
        # A system reminder or our own note rides in the user's turn too: neither is a new request.
        if _is_own_note(state, step.text) or step.text.startswith("<"):
            return
        said = f"the user asked: {_clip(step.text, 200)}"
        trigger, text = "task", step.text
        if state.goal:
            state.earlier_goals = [*state.earlier_goals, _clip(state.goal, 1000)][-2:]
        state.goal = step.text
        state.request_notes = 0
        state.request_file = ""
    elif step.kind == "say":
        said = f"agent said: {_clip(step.text, 300)}"
        if len(step.text) >= 60:
            trigger, text = "plan", step.text
    elif step.kind == "tool":
        tool = step.tool.lower()
        state.tools[step.id] = (tool, step.input)
        if len(state.tools) > OPEN_TOOL_CALLS:
            state.tools = dict(list(state.tools.items())[-OPEN_TOOL_CALLS:])
        what = _target(step.input)
        if tool_family(tool) == "command" and not READ_ONLY_CMD.match(what):
            # Claude Code backgrounds a command by a flag, not a trailing `&`: without it a check read
            # `index_check.py > out.txt 2>&1` as a foreground run against a job rule to background it
            # (beta: twice, on two runs that had backgrounded it from the start).
            if step.input.get("run_in_background"):
                what += "  # started in the background"
            said = f"agent runs: {_clip(what, 300)}"
            trigger, text = "command", what
        else:
            said = f"agent {tool}: {_clip(what, 200)}"
    else:
        tool, inp = state.tools.pop(step.id, ("", {}))
        named = _file_the_request_names(state.goal, _target(inp))
        # The longest read wins and a failed one never counts: in beta the first try was a `cat` the
        # harness refused for approval, and its refusal was kept as the job's file over the real Read.
        read = f"{named}:\n{step.text[:REQUEST_FILE_CAP]}"
        if named and not step.error and tool_family(tool) != "edit" and len(read) > len(state.request_file):
            state.request_file = read
        if _is_own_note(state, step.text):
            said = "the agent was given the watcher's note"
        elif tool_family(tool) == "edit":
            said = f"agent edited {_target(inp)}"
            # Judged by the client's diff when it sends one (a `state`), else by the edit's own text.
            trigger, text = "edit", _edit_text(inp)
        elif (step.error or FAILED.search(step.text)) and not ENV_TROUBLE.search(step.text):
            said = f"it failed: {step.text[-300:]}"
            trigger, text = "failure", step.text[-2000:]
    if said:
        state.recent = [*state.recent, said][-RECENT_STEPS:]
    # A new request is never pushed out: read in one pass with the agent's first "I'll check..." (a
    # plan, heavier), it used to be replaced, and the heads-up for the request was never given.
    keep = state.pending is not None and state.pending.trigger == "task" and trigger != "task"
    if trigger and not keep and (state.pending is None or PRIORITY[trigger] >= PRIORITY[state.pending.trigger]):
        state.pending = Pending(trigger=trigger, text=text, seq=seq)


def take_moment(state: WatchState) -> Moment | None:
    """The pending moment, removed from the state, with the steps and change it is judged against."""
    p = state.pending
    if p is None:
        return None
    state.pending = None
    return Moment(trigger=p.trigger, text=p.text, recent=list(state.recent), diff=state.diff, seq=p.seq)


def query_from_diff(diff: str, cap: int = 1500) -> str:
    """A diff as a search: the files and functions touched, then the changed lines."""
    files: list[str] = []
    funcs: list[str] = []
    lines: list[str] = []
    for line in diff.split("\n"):
        if line.startswith("+++ b/"):
            files.append(line[6:])
        elif line.startswith("@@"):
            funcs.append(line.split("@@")[-1].strip())
        elif line[:1] in ("+", "-") and not line.startswith(("+++", "---")) and line[1:].strip():
            lines.append(line.strip())
    q = f"Change to {', '.join(files)} ({'; '.join(f for f in funcs if f)}):\n" + "\n".join(lines)
    return q[:cap]


# A command or plan is searched by its first 300 chars: the program and its target. At 1500 chars a
# long shell line filled every reranker pair to the model's limit, and reranking 95 candidates took
# 30-58 s on a busy laptop — 8 of 19 checks hit the 60 s limit (beta, 2026-10-22).
COMMAND_QUERY_CAP = 300


def search_queries(m: Moment) -> list[str]:
    """What the moment searches. A request is searched by its sentences naming code, then its
    paragraphs: as one query, a harness's boilerplate (role, rules, "work efficiently") outweighed the
    request and the heads-up never saw the decision. An edit is searched by its diff in words,
    anything else by its own text."""
    if m.trigger == "edit" and m.diff.strip():
        return [query_from_diff(m.diff)]
    if m.trigger != "task":
        return [m.text[:COMMAND_QUERY_CAP]]
    paras = [p.strip() for p in re.split(r"\n\s*\n", m.text) if len(p.strip()) >= 40]
    named = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", m.text)
        if re.search(r"\b[a-z]+_[a-z_]+\b|\w\(\)|`[^`]+`", s) and not re.search("test", s, re.I)
    ]
    # A one-paragraph request that names code is both; searching it twice finds nothing new.
    queries = list(dict.fromkeys(q[:1500] for q in (named[:3] + paras)[:8]))
    return queries or [m.text[:1500]]


def _clip_diff(diff: str, cap: int) -> str:
    return diff if len(diff) <= cap else f"{diff[:cap]}\n... (diff cut at {cap} chars)"


def question(m: Moment) -> str:
    return HEADSUP_Q if m.trigger == "task" else CONFLICT_Q


def _by_diff(m: Moment) -> bool:
    """An edit is judged by the client's diff of the whole change when there is one."""
    return m.trigger == "edit" and bool(m.diff.strip())


def situation(
    goal: str,
    m: Moment,
    sent: list[str],
    set_aside: list[str] | None = None,
    request_file: str = "",
    earlier_goals: list[str] | None = None,
) -> str:
    """The moment as reflect's context: what the question is about, in the agent's own voice."""
    parts = [f"MY REQUEST:\n{goal[:3000]}"]
    if m.trigger != "task":
        # Not for a heads-up: it is about the new request alone.
        if earlier_goals:
            parts.insert(0, "EARLIER IN THIS SESSION I ASKED (still part of my request):\n" + "\n".join(earlier_goals))
        doing = _clip_diff(m.diff, 6000) if _by_diff(m) else m.text[:3000]
        parts.append(f"WHAT I JUST DID ({m.trigger}):\n{doing}")
        if m.recent:
            parts.append("MY RECENT STEPS:\n" + "\n".join(m.recent[-8:]))
        # The change so far is not shown for a command or a plan: each edit is judged by its own diff
        # when it is made. Shown, every later command re-judged it, and one session got the same rule
        # about one edited file, reworded, on 48 commands in an hour (beta); the repeat guard held them
        # all back, but each cost an answer call.
    if request_file:
        # Current, so it wins: in beta a check holding a job's file ("5 GiB since the last run") still
        # cited an older memory of the same job ("10 GiB") against the run.
        parts.append(
            "THE FILE MY REQUEST POINTS TO (its rules are part of my request and are current: where memory "
            "says otherwise, the file wins and memory is out of date):\n" + request_file
        )
    if sent:
        parts.append("ALREADY TOLD ME:\n" + "\n".join(sent))
    if set_aside:
        parts.append("I SET THESE ASIDE OR ALREADY DID THEM (never raise them again):\n" + "\n".join(set_aside))
    return "\n\n".join(parts)


def moment_key(m: Moment) -> str:
    """The same situation is never judged twice (an edit is its diff)."""
    return hashlib.sha1(
        json.dumps([m.trigger, "" if _by_diff(m) else m.text.strip(), m.diff.strip()]).encode(),
        usedforsecurity=False,
    ).hexdigest()


def skip_reason(state: WatchState, m: Moment) -> str | None:
    """Why the moment is not worth a reflect call, or None (and then it is recorded as judged)."""
    if state.request_notes >= MAX_NOTES:
        return "note limit reached"
    if m.trigger == "task" and len(_WORK_WORD.findall(_SLASH_OR_URL.sub(" ", m.text))) < MIN_REQUEST_WORDS:
        return "request names no work"
    if not (m.diff if _by_diff(m) else m.text).strip():
        return "nothing to judge"
    key = moment_key(m)
    if key in state.judged:
        return "already judged"
    state.judged = [*state.judged, key][-JUDGED_KEYS:]
    return None


# An answer that says there is nothing to say. Questions ask for an empty reply then, but after the
# switch to Gemma 4 one came back as "No conflict. The user's log entry ... is consistent with ..." and
# went out as a thought.
NOTHING_TO_SAY = re.compile(r"^\W*(no conflicts?\b|no (relevant )?decision\b|(nothing|none)\W*$)", re.I)
# The same said in a whole sentence ("The user is not contradicting any decision; ... aligns with the
# recorded rule", beta 03:15; "You are following your request to update the pricing ...", 13:36): judged
# on the first sentence only, so "X is consistent, but Y breaks
# the rule" in a later one still goes out.
AGREES_WITH_RULE = re.compile(
    r"\b(not|no|nothing|never) (contradict|conflict)\w*|\b(is|are|stays?|remains?) consistent with\b|\baligns? with\b"
    r"|\b(are|is) (already )?following (your|the)( \w+){0,2} (request|decision|rule)\b",
    re.I,
)


CLAIMS_CONFLICT = re.compile(
    r"\b(contradicts?|conflicts? with|violat\w*|breaks?|must not|should not|instead of)\b", re.I
)


def accept_note(state: WatchState, m: Moment, text: str, memory_ids: list[str]) -> bool:
    """Whether a note reaches the agent, recording it when it does: never the same memory twice.
    A heads-up does not use its memory up: going against it later is still worth saying."""
    # The agreement can come after a sentence that only narrates the step ("The user is adding scenarios
    # to test_consolidation.py. This aligns with the rule ...", beta 11:22): look at the first two, and
    # keep the note when it also claims a real conflict ("X is fine, but Y breaks the rule").
    lead = " ".join(re.split(r"(?<=[.;!?])\s", text.strip(), maxsplit=2)[:2])
    if not text.strip() or NOTHING_TO_SAY.match(text):
        return False
    if AGREES_WITH_RULE.search(lead) and not CLAIMS_CONFLICT.search(AGREES_WITH_RULE.sub("", text)):
        return False
    # The same words twice say nothing new, whatever the moment (a heads-up for a request the
    # client sent twice wrote the same note again).
    if text.strip() in state.sent:
        return False
    if memory_ids and all(i in state.evidence_sent for i in memory_ids):
        return False
    # A conflict saying mostly what an earlier conflict said is the same rule again (beta: one
    # article flagged three times within minutes, each from a rewritten memory). Mostly = 60% of the
    # shorter note's words: repeats measured 0.6-0.92, a different rule at most 0.45. A heads-up is
    # not compared: going against it later is worth saying.
    if m.trigger != "task" and any(_overlap(text, c) >= 0.6 for c in state.conflicts):
        return False
    # The same rule quoted word for word is the same rule, whatever file the work moved to: in beta
    # "scoped to pointwise ranking only" came back on a bench script, the code and its docs within
    # 8 minutes, each from a rewritten memory and under 0.6 overlap with the others.
    if m.trigger != "task" and _quoted(text) & {q for c in state.conflicts for q in _quoted(c)}:
        return False
    # A heads-up for a new request says nothing new when the session was already told the same rule:
    # in beta a dashboards rule came back, reworded, for the session's next request. (A conflict
    # after a heads-up still goes: that one says the work now breaks it.)
    if m.trigger == "task" and any(_overlap(text, t) >= 0.6 for t in state.sent):
        return False
    if any(_overlap(text, s) >= SET_ASIDE_OVERLAP for s in state.set_aside):
        return False
    state.sent.append(text.strip())
    state.request_notes += 1
    if m.trigger != "task":
        state.evidence_sent = list(dict.fromkeys([*state.evidence_sent, *memory_ids]))
        state.conflicts.append(text.strip())
    return True


def _overlap(a: str, b: str) -> float:
    """Share of the shorter text's words that the other one has."""
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / max(1, min(len(wa), len(wb)))


def readable(text: str) -> str:
    """A chat stored as JSON lines (or one JSON array), as its messages' words: a quote must be
    text the reader can find."""
    try:
        whole = json.loads(text)
    except ValueError:
        whole = None
    if isinstance(whole, list):
        return "\n".join(str(m.get("content", "")) if isinstance(m, dict) else str(m) for m in whole)
    out = []
    for line in text.split("\n"):
        try:
            msg = json.loads(line)
        except ValueError:
            out.append(line)
            continue
        out.append(str(msg.get("content", "")) if isinstance(msg, dict) else line)
    return "\n".join(out)


def _quoted(s: str) -> set[str]:
    """The phrases of three words or more a note quotes ("..." or curly quotes), lowercased."""
    return {q.strip().lower() for q in re.findall(r'["\u201c]([^"\u201c\u201d]+)["\u201d]', s) if len(q.split()) >= 3}


def _words(s: str) -> set[str]:
    # Numbers of any length count: a changed value ("7", not "10") is often the whole story.
    return set(re.findall(r"[a-z_][a-z0-9_]{2,}|\d+(?:\.\d+)?", s.lower()))


def best_sentence(text: str, like: str) -> str:
    """The source sentence sharing the most words with the note, less two per number the note does not
    have: verbatim by construction, no model call. By overlap alone the line stating the old value
    ("10 # Current setting") beat the rule (7). On a tie the later sentence wins: a chat settles on its
    last word ("leave it at 7" after trying 5)."""
    best, best_score = "", float("-inf")
    for s in (x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", text)):
        if len(s) < 20 or len(s) > 300:
            continue
        score = quote_score(s, like)
        if score is not None and score >= best_score:
            best, best_score = s, score
    return best


def quote_score(sentence: str, like: str) -> float | None:
    """How well a source sentence backs the note (see best_sentence), or None when it shares no word."""
    want = _words(like)
    shared = foreign = 0
    for w in _words(sentence):
        if w in want:
            shared += 1
        elif w[0].isdigit():
            foreign += 1
    # The penalty only ranks: "K is 1024, never 1000" still quotes a note that says 1024.
    return shared - 2 * foreign if shared > 0 else None

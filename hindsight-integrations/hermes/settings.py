"""Hindsight plugin constants and pure config normalizers (no I/O, no origin imports)."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import re
import string
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import Any, List

# Log under the plugin package's own logger name (loader-path independent).
logger = logging.getLogger(__name__.rpartition(".")[0])

_DEFAULT_API_URL = "https://api.hindsight.vectorize.io"
_DEFAULT_LOCAL_URL = "http://localhost:8888"
# Keep in sync with plugin.yaml and pyproject.toml. Raised to 0.10.1 with the embed
# floor (see pyproject.toml): _maybe_upgrade_client() then pulls the 0.10.0 cohort's
# client forward on session start instead of waiting for a `hermes update`.
_MIN_CLIENT_VERSION = "0.10.1"
_DEFAULT_TIMEOUT = 120  # seconds — cloud API can take 30-40s per request
_DEFAULT_IDLE_TIMEOUT = 300  # seconds — Hindsight embedded daemon default
# ``metadata.source`` on retained memories is OPT-IN (AGENTS.md forbids
# on-by-default attribution tags): ``retain_source`` / HINDSIGHT_RETAIN_SOURCE.
_DEFAULT_RETAIN_SOURCE = ""
# Named retain strategy applied to every item this plugin stores. Empty means the bank
# decides (its ``retain_default_strategy``, else unmissioned ``concise`` extraction).
# The bank must already define the name under ``retain_strategies``.
_DEFAULT_RETAIN_STRATEGY = ""
# Hermes tags every retain with volatile ``session:``/``parent:`` lineage, so the engine's
# ``combined`` default (one scope per exact tag set) would keep observations per session.
# ``shared`` merges them across sessions; the lineage tags stay on the facts.
_DEFAULT_OBSERVATION_SCOPES = "shared"
# Hindsight brand mark (eye ringed by graph nodes) for the recall/retain indicators.
_HINDSIGHT_GLYPH = "💭"
# Hindsight 0.5.0 added ``update_mode='append'``; older APIs would silently
# overwrite prior turns under a stable document_id, so they keep the per-process id.
# Mirrors hindsight-integrations/openclaw — Hindsight 0.5.0 added `update_mode='append'` semantics on retain
# (vectorize-io/hindsight#932).
_MIN_VERSION_FOR_UPDATE_MODE_APPEND = "0.5.0"
_VALID_BUDGETS = {"low", "mid", "high"}
_PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-5-nano",
    "anthropic": "claude-haiku-5-5",
    "gemini": "gemini-3.8-flash",
    "groq": "openai/gpt-oss-120b",
    "openrouter": "qwen/qwen3.5-9b",
    "minimax": "MiniMax-M3",
    "ollama": "gemma3:12b",
    "lmstudio": "local-model",
    "openai_compatible": "your-model-name",
}
# The embedded daemon speaks OpenAI wire format for these providers.
_OPENAI_WIRE_PROVIDERS = {"openai_compatible", "openrouter"}
_OBSERVATION_SCOPE_KEYWORDS = {"per_tag", "combined", "all_combinations", "shared"}


def _parse_int_setting(value: Any, default: int) -> int:
    """Parse an integer config/env value, falling back on invalid input."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("Invalid integer Hindsight setting %r; using default %s", value, default)
        return default


def _parse_float_setting(value: Any, default: float) -> float:
    """Parse a float config/env value, falling back on invalid input."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("Invalid number for Hindsight setting %r; using default %s", value, default)
        return default


def _daemon_llm_provider(provider: str) -> str:
    return "openai" if provider in _OPENAI_WIRE_PROVIDERS else provider


def _normalize_string_list(value: Any) -> List[str]:
    """Normalize a list-valued setting (retain tags, bank lists) to a deduplicated list of
    non-empty strings. Accepts a list, a JSON-encoded list (``'["a", "b"]'``) or
    comma-separated text (``"a, b"``): the settings panel stores ``KIND_TEXT`` fields as
    text, while a hand-written ``config.json`` usually holds a real list."""
    if value is None:
        return []
    raw_items = value if isinstance(value, list) else [value]
    if isinstance(value, str):
        text = value.strip()
        parsed = None
        if text.startswith("["):
            with contextlib.suppress(Exception):
                parsed = json.loads(text)
        raw_items = parsed if isinstance(parsed, list) else text.split(",")
    normalized: list[str] = []
    for item in raw_items:
        entry = str(item).strip()
        if entry and entry not in normalized:
            normalized.append(entry)
    return normalized


def _local_date(value: Any, tz: tzinfo | None) -> str:
    """A server timestamp (UTC ISO 8601, or a datetime) as ``YYYY-MM-DD`` in *tz*, else ``""``.

    Converted before cutting the day off: an 18:30 UTC fact is already the next day in UTC+8.
    A naive value is UTC (the server's zone). Anything that does not parse renders nothing, so a
    malformed field never becomes a fake date in the prompt."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return ""
    else:
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(tz).date().isoformat()


def _format_memory(result: Any, tz: tzinfo | None = None) -> str:
    """One recalled memory with its dates, so the model can tell an old fact from a fresh one (#4697).

    ``mentioned`` is when the fact was stated; ``occurred`` is when the event itself happened, which
    is what dates the claim (a fact new to the bank can describe an old state). Same shape as the
    OpenClaw plugin's ``formatMemories``; each bound can be missing, so each case reads plainly."""
    text = getattr(result, "text", "") or ""
    mentioned = _local_date(getattr(result, "mentioned_at", None), tz)
    start = _local_date(getattr(result, "occurred_start", None), tz)
    end = _local_date(getattr(result, "occurred_end", None), tz)
    if start and end:
        occurred = f" [occurred: {start}]" if start == end else f" [occurred: {start} → {end}]"
    elif start:
        occurred = f" [occurred from: {start}]"
    elif end:
        occurred = f" [occurred until: {end}]"
    else:
        occurred = ""
    return f"{text}{f' (mentioned: {mentioned})' if mentioned else ''}{occurred}"


def _compile_line_patterns(value: Any) -> list[re.Pattern[str]]:
    """``recall_query_strip_patterns`` -> compiled regexes. A list, a JSON list or one pattern per
    line (commas are regex syntax, so they do not split). An invalid pattern is skipped with a warning."""
    if isinstance(value, str):
        text = value.strip()
        parsed = None
        if text.startswith("["):
            with contextlib.suppress(ValueError):
                parsed = json.loads(text)
        value = parsed if isinstance(parsed, list) else text.splitlines()
    patterns: list[re.Pattern[str]] = []
    for raw in value or []:
        if not str(raw).strip():
            continue
        try:
            patterns.append(re.compile(str(raw)))
        except re.error as exc:
            logger.warning("Ignoring invalid recall_query_strip_patterns entry %r: %s", raw, exc)
    return patterns


def _strip_query_lines(query: str, patterns: list[re.Pattern[str]]) -> str:
    """Drop the query lines any pattern matches (#5639). If nothing would be left, keep the query
    as it was: a too-broad pattern then degrades to the old behaviour instead of disabling recall."""
    if not patterns:
        return query
    kept = "\n".join(line for line in query.splitlines() if not any(p.search(line) for p in patterns)).strip()
    return kept or query


@dataclass(frozen=True)
class SessionContextRef:
    """One ``session_context`` entry: a mental model or a page, by id."""

    kind: str  # "mental_model" | "page"
    ref: str


def _parse_session_context(value: Any) -> list[SessionContextRef]:
    """``session_context`` -> the entries in the configured order (#5285).

    Each entry is ``mental_model:<id>`` or ``page:<id>``; a bare id is a mental model. Unknown
    kinds are dropped with a warning."""
    entries: list[SessionContextRef] = []
    for item in _normalize_string_list(value):
        kind, sep, ref = item.partition(":")
        kind, ref = (kind.strip(), ref.strip()) if sep else ("mental_model", item)
        if kind not in ("mental_model", "page") or not ref:
            logger.warning("Ignoring session_context entry %r: expected mental_model:<id> or page:<id>", item)
            continue
        entries.append(SessionContextRef(kind, ref))
    return entries


def _normalize_min_scores(value: Any) -> dict[str, float] | None:
    """Normalize ``recall_min_scores`` to ``{score_name: floor}``, or ``None`` for no floor.

    Accepts a mapping or a JSON object string. The score names are Hindsight's own
    (``reranker``, ``semantic``, ...), so they are not validated here: the server decides
    which exist. A floor that is not a finite number is dropped with a warning instead of
    being sent, so a typo can never turn into an invalid recall request."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            logger.warning("Invalid recall_min_scores %r (not valid JSON); no score floor applied", value)
            return None
    if not isinstance(value, dict):
        logger.warning("Invalid recall_min_scores %r (expected an object); no score floor applied", value)
        return None
    floors: dict[str, float] = {}
    for name, floor in value.items():
        if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not math.isfinite(floor):
            logger.warning("Ignoring recall_min_scores[%r]=%r: a floor must be a finite number", name, floor)
            continue
        floors[str(name)] = float(floor)
    return floors or None


def _clears_min_scores(result: Any, floors: dict[str, float]) -> bool:
    """Whether a recall result reports every score named in *floors* and clears each (``>=``).

    Recall fuses four retrieval arms and returns a result surfaced by any of them, so the
    ``semantic`` and ``keyword`` scores are ``None`` on a result the other arms found. The
    server only guarantees ``reranker`` and ``final`` floors (a ``semantic`` floor prunes just
    its own arm), so a result that does not report a stage named in a floor is rejected here:
    that is what lets a ``semantic`` floor abstain on an off-topic query. A result carrying no
    ``scores`` at all (a server that does not report them) is kept, there being nothing to judge it by."""
    scores = getattr(result, "scores", None)
    if scores is None:
        return True
    return all((value := getattr(scores, name, None)) is not None and value >= floor for name, floor in floors.items())


def _normalize_observation_scopes(value: Any) -> Any:
    """Normalize observation_scopes to a keyword string, ``list[list[str]]`` (one inner
    list per consolidation pass), or ``None`` (Hindsight's ``combined`` default).
    Accepts a keyword, a JSON-encoded list, a flat tag list (one scope) or a list of
    tag-lists; anything unrecognized -> ``None`` so we never send an invalid payload."""
    if isinstance(value, str):
        text = value.strip()
        if text in _OBSERVATION_SCOPE_KEYWORDS:
            return text
        if text.startswith("["):
            try:
                return _normalize_observation_scopes(json.loads(text))
            except Exception:
                return None
        return None
    if not isinstance(value, (list, tuple)):
        return None
    if all(isinstance(entry, str) for entry in value):  # flat tag list -> one scope
        value = [value]
    scopes = [
        [str(tag).strip() for tag in entry if str(tag).strip()]
        if isinstance(entry, (list, tuple))
        else [entry.strip()]
        if isinstance(entry, str) and entry.strip()
        else []
        for entry in value
    ]
    return [s for s in scopes if s] or None


def _sanitize_bank_segment(value: str) -> str:
    """URL/filesystem-safe bank_id placeholder: runs outside ``[A-Za-z0-9_-]`` (per
    ``str.isalnum``) become one dash; leading/trailing ``-``/``_`` are stripped."""
    # \w == str.isalnum() + "_" for str patterns, so this matches the per-char rule.
    return re.sub(r"[^\w-]+", "-", str(value)).strip("-_") if value else ""


def _resolve_bank_id_template(template: str, fallback: str, **placeholders: str) -> str:
    """Render a bank_id template ({profile}, {workspace}, {project}, {platform}, {user}, {session}),
    sanitizing each placeholder; the ``-``/``_`` runs empty placeholders leave are
    collapsed (``hermes-{user}`` -> ``hermes``). Empty/invalid template -> *fallback*."""
    if not template:
        return fallback
    try:
        rendered = template.format(**{k: _sanitize_bank_segment(v) for k, v in placeholders.items()})
    except Exception as exc:
        # str.format raises KeyError, IndexError, ValueError, AttributeError or TypeError
        # depending on how the template is malformed; none of them may stop the provider.
        logger.warning("Invalid bank_id_template %r: %s — using fallback %r", template, exc, fallback)
        return fallback
    return re.sub(r"([-_])\1+", r"\1", rendered).strip("-_") or fallback


def _repository_root(start_dir: str) -> Path | None:
    """The nearest folder at or above *start_dir* holding ``.git`` (a directory, or a
    worktree's ``.git`` file). ``None`` outside a repository, and for a repository rooted
    at the home directory or the filesystem root."""
    start = Path(start_dir).resolve()
    home = Path.home().resolve()
    for folder in (start, *start.parents):
        if folder in {home, Path(folder.root)}:
            return None
        if (folder / ".git").exists():
            return folder
    return None


@dataclass(frozen=True)
class _MainRepository:
    path: Path
    bare: bool


def _main_repository(root: Path) -> _MainRepository:
    """The main repository behind *root* and whether it is bare. A linked worktree's
    ``.git`` file names its ``gitdir``, whose ``commondir`` points at the shared git
    directory: ``<main>/.git`` for a normal checkout, the repository itself when bare.
    Anything else, submodules included, is its own main repository."""
    git_file = root / ".git"
    if not git_file.is_file():
        return _MainRepository(root, bare=False)
    text = git_file.read_text(encoding="utf-8").strip()
    if not text.startswith("gitdir:"):
        return _MainRepository(root, bare=False)
    gitdir = (root / text[len("gitdir:") :].strip()).resolve()
    commondir_file = gitdir / "commondir"
    if not commondir_file.is_file():
        return _MainRepository(root, bare=False)
    common = (gitdir / commondir_file.read_text(encoding="utf-8").strip()).resolve()
    if common.name == ".git":
        return _MainRepository(common.parent, bare=False)
    return _MainRepository(common, bare=True)


def _project_name(root: Path | None) -> str:
    """Value of the ``{project}`` placeholder: the main repository's folder name, ``""``
    when there is no repository. Only a bare main repository drops its ``.git`` suffix,
    so a checkout in a folder named ``x.git`` and its worktrees all resolve to ``x.git``."""
    if root is None:
        return ""
    main = _main_repository(root)
    return main.path.name.removesuffix(".git") if main.bare else main.path.name


def _discover_cwd_bank_id(start_dir: str, root: Path | None, trusted_dirs: List[str]) -> str | None:
    """``bank_id`` from the nearest ``.hindsight/config.toml`` inside a trusted repository.

    A repository can carry this file to name its bank, but a cloned repository is not
    trusted by default: the file is read only when *root* itself lies inside one of
    *trusted_dirs*. A linked worktree is judged by its own location, since the files that
    link it to a main repository are under its own control. Relative entries are ignored,
    since they would resolve against whatever directory Hermes was started from. The walk goes from *start_dir* up to
    *root* and never above it. Only ``bank_id`` is read, with stdlib ``tomllib``. Never
    raises: an unreadable or malformed file logs a warning and the walk continues.
    """
    if root is None or not trusted_dirs:
        return None

    try:
        trusted = []
        for entry in trusted_dirs:
            path = Path(entry).expanduser()
            if path.is_absolute():
                trusted.append(path.resolve())
            else:
                logger.warning("hindsight: ignoring relative trusted_project_dirs entry %r", entry)
        # Trust follows where the repository physically is. A worktree's .git file and
        # commondir are plain files the folder itself controls, so they cannot vouch for it.
        if not any(root == t or t in root.parents for t in trusted):
            logger.debug("hindsight: %s is not under trusted_project_dirs — ignoring its config", root)
            return None
        start = Path(start_dir).resolve()
        for folder in (start, *start.parents):
            candidate = folder / ".hindsight" / "config.toml"
            if candidate.is_file():
                try:
                    data = tomllib.loads(candidate.read_text(encoding="utf-8", errors="replace"))
                    bank = data.get("bank_id")
                    if isinstance(bank, str) and bank.strip():
                        return bank.strip()
                    logger.warning("hindsight: %s has no usable bank_id — walking up", candidate)
                except Exception as exc:
                    logger.warning("hindsight: cannot read %s (%s) — walking up", candidate, exc)
            if folder == root:
                break
    except Exception as exc:
        logger.debug("hindsight: cwd walk failed: %s", exc)
    return None


def _template_fields(template: str) -> set[str]:
    """Placeholder names *template* uses, parsed the way ``str.format`` does, so
    ``{project:.30}`` and ``{project!s}`` count as ``project``."""
    try:
        return {field.split(".")[0].split("[")[0] for _, field, _, _ in string.Formatter().parse(template) if field}
    except ValueError:
        return set()

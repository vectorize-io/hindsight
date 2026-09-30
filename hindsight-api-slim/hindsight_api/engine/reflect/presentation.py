"""How a retrieval tool's result is written into the reflect prompt.

The tools return the full serialized records; this decides what the MODEL reads.
Measured on the refresh-cost eval, the fact text was a fifth of the synthesis
prompt. The rest was the frame around it: a 36-character UUID per item, full
ISO timestamps to the microsecond, a ``fact_type`` that is the same on every
item of a list, chunk keys built from the bank id plus a document UUID, and
items the model had already been shown by an earlier search. Every one of those
is re-sent on every later turn of the loop.

So, per reflect:

* ids become short aliases (``f1`` fact, ``o1`` observation, ``p1`` page,
  ``c1`` chunk), and every alias the model writes back — in ``done``, in
  ``expand`` — is resolved to the real id before anything reads it. The API
  trace keeps the raw output; only the prompt is shortened.
* timestamps keep their minute (``2026-03-07 12:00``), dropping seconds,
  microseconds and the UTC offset (they are all UTC). Dates are evidence:
  supersession is decided by them, so they are shortened, never dropped.
* ``occurred_end`` is dropped when it equals ``occurred_start``; a list's
  constant ``fact_type`` (every observation is an observation) is dropped.
* chunk ``chunk_index`` is dropped, and ``truncated`` kept only when true.
* an item already shown earlier in this reflect is listed by alias under
  ``already_shown`` instead of being written out again: the model has it, and
  that a later search matched it too is still said.
"""

from __future__ import annotations

import re
from typing import Any

_ALIAS_RE = re.compile(r"^[fopc]\d+$")
_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]00:?00)?$")
_DATE_FIELDS = ("mentioned_at", "occurred_start", "occurred_end", "updated_at", "created_at")
#: Result lists and the alias prefix of their items.
_LISTS = {"memories": "f", "observations": "o", "mental_models": "p"}
#: A list's constant type field, dropped because the list key already says it.
_CONSTANT_TYPE = {"observations": "observation"}


def compact_timestamp(value: Any) -> Any:
    """``2026-03-07T12:00:00.123+00:00`` -> ``2026-03-07 12:00``; other values unchanged."""
    if not isinstance(value, str):
        return value
    m = _TIMESTAMP_RE.match(value)
    return f"{m.group(1)} {m.group(2)}" if m else value


class ToolResultPresenter:
    """Per-reflect state: the alias table and which items the model has already read."""

    def __init__(self) -> None:
        self._alias_by_id: dict[str, str] = {}
        self._id_by_alias: dict[str, str] = {}
        self._counters: dict[str, int] = {}
        self._shown: set[tuple[str, str, bool]] = set()

    def alias(self, raw_id: str, prefix: str) -> str:
        existing = self._alias_by_id.get(raw_id)
        if existing is not None:
            return existing
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        alias = f"{prefix}{self._counters[prefix]}"
        self._alias_by_id[raw_id] = alias
        self._id_by_alias[alias] = raw_id
        return alias

    def resolve(self, value: Any) -> Any:
        """Map every alias in a tool call's arguments back to its real id.

        Only exact alias strings are touched (a list element or a whole value), so
        free text — a query that happens to contain ``f1`` — is left alone.
        """
        if isinstance(value, str):
            return self._id_by_alias.get(value, value) if _ALIAS_RE.match(value) else value
        if isinstance(value, list):
            return [self.resolve(v) for v in value]
        if isinstance(value, dict):
            return {k: (v if k in ("query", "reason", "answer") else self.resolve(v)) for k, v in value.items()}
        return value

    def present(self, output: Any) -> Any:
        """The prompt form of one tool result. Never mutates ``output``."""
        if not isinstance(output, dict) or "error" in output:
            return output
        shown: dict[str, list[str]] = {}
        out: dict[str, Any] = {}
        for key, value in output.items():
            if key in _LISTS and isinstance(value, list):
                items = []
                for item in value:
                    if not isinstance(item, dict) or "id" not in item:
                        items.append(item)
                        continue
                    alias = self.alias(str(item["id"]), _LISTS[key])
                    # Keyed by the list AND by whether this is the full text: a mental
                    # model comes back as a snippet from a search and as full text from
                    # a read, and the read is the one the model asked for. Keying on the
                    # id alone suppressed that read entirely — the page came back as
                    # `already_shown` and the model never saw more than the snippet.
                    seen_key = (key, str(item["id"]), "content" in item)
                    if seen_key in self._shown:
                        shown.setdefault(key, []).append(alias)
                        continue
                    self._shown.add(seen_key)
                    items.append(self._present_item(item, key, alias))
                out[key] = items
            elif key == "chunks" and isinstance(value, dict):
                out[key] = {self.alias(str(cid), "c"): self._present_chunk(chunk) for cid, chunk in value.items()}
            elif key == "source_facts" and isinstance(value, dict):
                out[key] = {
                    self.alias(str(fid), "f"): self._present_item(fact, "memories", self.alias(str(fid), "f"))
                    for fid, fact in value.items()
                }
            elif key == "count":
                continue  # the length of the list right below it
            else:
                out[key] = value
        if shown:
            out["already_shown"] = shown
        return out

    def _present_item(self, item: dict[str, Any], list_key: str, alias: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        constant = _CONSTANT_TYPE.get(list_key)
        for field, value in item.items():
            if field == "id":
                out["id"] = alias
            elif field in ("fact_type", "type") and constant is not None and value == constant:
                continue
            elif field == "source_fact_ids" and isinstance(value, list):
                out[field] = [self.alias(str(v), "f") for v in value]
            elif field in _DATE_FIELDS:
                out[field] = compact_timestamp(value)
            else:
                out[field] = value
        if "occurred_end" in out and out.get("occurred_end") == out.get("occurred_start"):
            del out["occurred_end"]
        return out

    @staticmethod
    def _present_chunk(chunk: Any) -> Any:
        if not isinstance(chunk, dict):
            return chunk
        out = {k: v for k, v in chunk.items() if k not in ("chunk_index", "truncated")}
        if chunk.get("truncated"):
            out["truncated"] = True
        return out

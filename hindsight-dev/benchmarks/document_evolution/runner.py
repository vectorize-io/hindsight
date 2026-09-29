"""Drive one build of the server through a case's rounds and record what happened.

The runner talks HTTP only. That is the point: the same harness runs against a
server built from any revision, which is how "is the new pipeline better than the
old one" gets answered without a feature flag inside the code under test. Run it
once per build, then hand both artifacts to ``compare``.

Seeding has two modes because they answer different questions:

- ``authored`` writes an identical starting document into both builds (needs
  ``--db-url``, since the HTTP API deliberately has no way to set a document's
  text). Same input on both sides, so any divergence afterwards is the pipeline.
- ``generated`` lets each build write its own first version from the same
  memories. Less controlled, but it is what a real page does, and it exercises
  the generation path rather than only the editing path.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import httpx
from pydantic import BaseModel, Field

from .corpus import Case
from .metrics import DocumentShape, ShapeDelta, compare_shape, describe, untouched_sections_drifted


class OperationOutcome(BaseModel):
    """How an async operation ended."""

    status: str
    error_message: str | None = None
    result_metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "completed"


class CreatedMentalModel(BaseModel):
    mental_model_id: str
    operation_id: str | None = None


class RoundResult(BaseModel):
    """One refresh: what the document became, and what the pipeline did to it."""

    index: int = Field(description="0 is the seed refresh; 1..n are the fact rounds.")
    fact: str | None
    document: str
    shape: DocumentShape
    delta_from_previous: ShapeDelta
    drifted_sections: list[str] = Field(
        default_factory=list, description="Sections that changed although no operation named them."
    )
    delta_applied: bool = False
    delta_skipped_reason: str | None = None
    refresh_skipped: str | None = None
    ops_applied: int = 0
    ops_skipped: int = 0
    touched_sections: list[str] = Field(default_factory=list)
    duration_ms: int = Field(
        default=0,
        description=(
            "Wall clock for the refresh. Token usage is deliberately absent: the stored "
            "reflect_response does not carry it, and a column that is always zero reads as "
            "'this costs nothing' rather than 'this is not measured here'."
        ),
    )
    error: str | None = None


class CaseRun(BaseModel):
    """Every round of one case against one build."""

    case: str
    fragile: bool
    topic: str
    seeding: str
    bank_id: str
    mental_model_id: str
    rounds: list[RoundResult] = Field(default_factory=list)

    @property
    def final_document(self) -> str:
        return self.rounds[-1].document if self.rounds else ""


class BenchmarkArtifact(BaseModel):
    """Everything one build produced — the file ``compare`` reads."""

    build: str = Field(description="Label for the build under test, e.g. a git ref.")
    api_url: str
    api_version: str = ""
    model: str = ""
    runs: list[CaseRun] = Field(default_factory=list)


class _Client:
    """Thin wrapper over the endpoints this benchmark needs."""

    def __init__(self, api_url: str, timeout: float = 600.0) -> None:
        self._base = api_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._http.aclose()

    async def _json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        response = await self._http.request(method, f"{self._base}{path}", json=payload)
        response.raise_for_status()
        return response.json() if response.content else {}

    async def version(self) -> str:
        return (await self._json("GET", "/version")).get("api_version", "")

    async def ensure_bank(self, bank_id: str) -> None:
        await self._json("GET", f"/v1/default/banks/{bank_id}/stats")

    async def retain(self, bank_id: str, content: str) -> None:
        """Retain one memory and wait for it to land.

        A fact whose retain has not completed is not yet in the delta window, so
        refreshing before it lands measures nothing.
        """
        response = await self._json(
            "POST", f"/v1/default/banks/{bank_id}/memories", {"items": [{"content": content}], "async": True}
        )
        operation_ids = list(response.get("operation_ids") or [])
        if response.get("operation_id"):
            operation_ids.insert(0, response["operation_id"])
        for operation_id in operation_ids:
            await self.await_operation(bank_id, operation_id)

    async def await_operation(self, bank_id: str, operation_id: str, timeout: float = 600.0) -> OperationOutcome:
        """Block until an async operation settles.

        Every write here — retain, create, refresh — is submitted asynchronously
        and returns an operation id. Sleeping a fixed interval instead silently
        benchmarks a document that is still being written: the first version of
        this runner did exactly that and recorded five rounds of
        "Generating content..." as though they were real documents.
        """
        deadline = time.time() + timeout
        delay = 0.4
        while True:
            operation = await self._json("GET", f"/v1/default/banks/{bank_id}/operations/{operation_id}")
            status = str(operation.get("status") or "")
            if status in {"completed", "failed", "cancelled", "not_found"}:
                return OperationOutcome(
                    status=status,
                    error_message=operation.get("error_message"),
                    result_metadata=operation.get("result_metadata") or {},
                )
            if time.time() > deadline:
                return OperationOutcome(status="timeout", error_message=f"still {status} after {timeout}s")
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 5.0)

    async def create_mental_model(
        self, bank_id: str, name: str, topic: str, trigger: dict[str, Any]
    ) -> CreatedMentalModel:
        created = await self._json(
            "POST",
            f"/v1/default/banks/{bank_id}/mental-models",
            {"name": name, "source_query": topic, "trigger": trigger},
        )
        return CreatedMentalModel(mental_model_id=created["mental_model_id"], operation_id=created.get("operation_id"))

    async def refresh(self, bank_id: str, mental_model_id: str) -> OperationOutcome:
        """Submit a refresh and wait for it — the endpoint is submit-only."""
        submitted = await self._json("POST", f"/v1/default/banks/{bank_id}/mental-models/{mental_model_id}/refresh", {})
        operation_id = submitted.get("operation_id")
        if not operation_id:
            return OperationOutcome(status="failed", error_message=f"no operation id in {submitted!r}")
        return await self.await_operation(bank_id, operation_id)

    async def get_mental_model(self, bank_id: str, mental_model_id: str) -> dict[str, Any]:
        return await self._json("GET", f"/v1/default/banks/{bank_id}/mental-models/{mental_model_id}")

    async def delete_bank(self, bank_id: str) -> None:
        await self._json("DELETE", f"/v1/default/banks/{bank_id}")


async def _write_seed_document(db_url: str, bank_id: str, mental_model_id: str, document: str) -> None:
    """Install an authored document straight into the row.

    The HTTP API has no way to set a document's text (creation only takes a
    topic), and the whole point of this mode is that both builds start from
    exactly the same bytes. ``structured_content`` is cleared so the next refresh
    imports the document the way an upgraded install would.
    """
    import asyncpg

    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(
            "UPDATE mental_models SET content = $1, structured_content = NULL WHERE id = $2 AND bank_id = $3",
            document,
            mental_model_id,
            bank_id,
        )
    finally:
        await conn.close()


# What the API stores while an async build is still running. A round that reads
# one of these is measuring a placeholder, not a document.
_PLACEHOLDERS = ("generating content...", "no answer provided.", "")


def _is_placeholder(document: str) -> bool:
    return document.strip().lower() in _PLACEHOLDERS


def _touched_headings(document: str, applied_ops: list[dict[str, Any]]) -> list[str]:
    """Map applied operations back to the headings they named.

    Operations carry section *ids*, which are slugs of headings, and the two
    builds slug identically — so matching a heading by its slug works across
    both without either build having to report headings.
    """
    import re

    touched_ids = {op.get("section_id") for op in applied_ops} | {op.get("assigned_id") for op in applied_ops}
    touched_ids = {i for i in touched_ids if i}
    headings: list[str] = []
    for line in document.splitlines():
        match = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if not match:
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", match.group(2).strip().lower()).strip("-")
        if slug in touched_ids or any(slug == t or t.startswith(f"{slug}-") for t in touched_ids):
            headings.append(line.strip())
    return headings


async def run_case(
    client: _Client,
    case: Case,
    *,
    seeding: str,
    db_url: str | None,
    settle_seconds: float,
) -> CaseRun:
    """Seed a document, then feed it one fact per round, recording each version."""
    bank_id = f"docevo-{case.name}-{uuid.uuid4().hex[:8]}"
    await client.ensure_bank(bank_id)
    for memory in case.seed_memories:
        await client.retain(bank_id, memory)
    # Consolidation runs behind the retain operations and the delta window reads
    # observations, so give it a moment to produce them before the first refresh.
    await asyncio.sleep(settle_seconds)

    created = await client.create_mental_model(bank_id, f"{case.name} reference", case.topic, {"mode": "delta"})
    mental_model_id = created.mental_model_id
    if created.operation_id:
        outcome = await client.await_operation(bank_id, created.operation_id)
        if not outcome.ok:
            raise RuntimeError(f"creating the mental model {outcome.status}: {outcome.error_message}")

    if seeding == "authored":
        if not db_url:
            raise ValueError("authored seeding needs --db-url")
        await _write_seed_document(db_url, bank_id, mental_model_id, case.seed_document)

    run = CaseRun(
        case=case.name,
        fragile=case.fragile,
        topic=case.topic,
        seeding=seeding,
        bank_id=bank_id,
        mental_model_id=mental_model_id,
    )

    stored = await client.get_mental_model(bank_id, mental_model_id)
    previous = stored.get("content") or ""
    if _is_placeholder(previous):
        raise RuntimeError(
            f"the seed document is still a placeholder ({previous.strip()!r}) — the create operation "
            "reported success but wrote nothing"
        )
    run.rounds.append(
        RoundResult(
            index=0,
            fact=None,
            document=previous,
            shape=describe(previous),
            delta_from_previous=ShapeDelta(),
        )
    )

    for index, round_spec in enumerate(case.rounds, start=1):
        await client.retain(bank_id, round_spec.fact)
        await asyncio.sleep(settle_seconds)

        started = time.time()
        error: str | None = None
        try:
            outcome = await client.refresh(bank_id, mental_model_id)
            if not outcome.ok:
                # A refused refresh is a result, not a crash: the document is
                # preserved and this round is recorded as failed.
                error = f"{outcome.status}: {outcome.error_message}"
        except httpx.HTTPStatusError as exc:
            error = f"{exc.response.status_code}: {exc.response.text[:200]}"
        duration_ms = int((time.time() - started) * 1000)

        refreshed = await client.get_mental_model(bank_id, mental_model_id)
        document = refreshed.get("content") or ""
        reflect_response = refreshed.get("reflect_response") or {}
        applied = reflect_response.get("delta_operations_applied") or []
        touched = _touched_headings(previous, applied)

        run.rounds.append(
            RoundResult(
                index=index,
                fact=round_spec.fact,
                document=document,
                shape=describe(document),
                delta_from_previous=compare_shape(describe(previous), describe(document)),
                drifted_sections=untouched_sections_drifted(previous, document, set(touched)),
                delta_applied=bool(reflect_response.get("delta_applied")),
                delta_skipped_reason=reflect_response.get("delta_skipped_reason"),
                refresh_skipped=reflect_response.get("refresh_skipped"),
                ops_applied=len(applied),
                ops_skipped=len(reflect_response.get("delta_operations_skipped") or []),
                touched_sections=touched,
                duration_ms=duration_ms,
                error=error,
            )
        )
        previous = document

    return run


async def run_build(
    api_url: str,
    cases: list[Case],
    *,
    build: str,
    seeding: str,
    db_url: str | None,
    repetitions: int,
    settle_seconds: float,
    keep_banks: bool,
) -> BenchmarkArtifact:
    """Run every case (``repetitions`` times) against one server."""
    client = _Client(api_url)
    try:
        artifact = BenchmarkArtifact(build=build, api_url=api_url, api_version=await client.version())
        for repetition in range(repetitions):
            for case in cases:
                run = await run_case(client, case, seeding=seeding, db_url=db_url, settle_seconds=settle_seconds)
                # Repetitions share a case name on purpose: the report aggregates
                # over them, because one LLM run proves nothing about a model.
                artifact.runs.append(run)
                print(
                    f"[{build}] {case.name} rep {repetition + 1}/{repetitions}: "
                    f"{len(run.rounds) - 1} rounds, "
                    f"{sum(r.delta_from_previous.damaged for r in run.rounds)} damaged",
                    # A run takes tens of minutes; buffered progress that only
                    # appears at the end is no progress at all.
                    flush=True,
                )
                if not keep_banks:
                    await client.delete_bank(run.bank_id)
        return artifact
    finally:
        await client.close()

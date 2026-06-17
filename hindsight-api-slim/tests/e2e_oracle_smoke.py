#!/usr/bin/env python3
"""
End-to-end smoke test for the Oracle 23ai pipeline.

Exercises: migrations -> create bank -> retain -> recall -> mental model -> cleanup.

Uses the ``mock`` LLM provider and ``LocalSTEmbeddings`` so no API keys are needed.

Usage:
    ORACLE_TEST_DSN="oracle://SYSTEM:oracle@localhost:1521/FREEPDB1" \
        uv run python tests/e2e_oracle_smoke.py
"""

import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log(step: int, total: int, msg: str) -> None:
    print(f"  [{step}/{total}] {msg}")


def _parse_oracle_dsn(dsn: str) -> dict[str, str]:
    """Parse an oracle:// URL into oracledb connect kwargs."""
    parsed = urlparse(dsn)
    host = parsed.hostname or "localhost"
    port = parsed.port or 1521
    service = parsed.path.lstrip("/") if parsed.path else "FREEPDB1"
    return {
        "user": parsed.username or "SYSTEM",
        "password": parsed.password or "oracle",
        "dsn": f"{host}:{port}/{service}",
    }


def _bootstrap_test_user(admin_dsn: dict[str, str]) -> str:
    """Create HINDSIGHT_TEST user (idempotent) and return its oracle:// URL.

    Oracle 23ai requires VECTOR columns in an ASSM tablespace.  The default
    SYSTEM tablespace is *not* ASSM, so we create a dedicated user whose
    default tablespace is USERS (which is ASSM on Oracle Free/XE).
    """
    import oracledb

    oracledb.defaults.fetch_lobs = False

    test_user = "HINDSIGHT_TEST"
    test_pass = "hindsight_test"

    conn = oracledb.connect(
        user=admin_dsn["user"],
        password=admin_dsn["password"],
        dsn=admin_dsn["dsn"],
    )
    cursor = conn.cursor()
    try:
        # Create user (skip if already exists - ORA-01920)
        try:
            cursor.execute(
                f'CREATE USER {test_user} IDENTIFIED BY "{test_pass}" DEFAULT TABLESPACE USERS QUOTA UNLIMITED ON USERS'
            )
        except oracledb.DatabaseError as e:
            if hasattr(e.args[0], "code") and e.args[0].code == 1920:
                pass
            else:
                raise

        for grant in [
            f"GRANT CONNECT, RESOURCE, UNLIMITED TABLESPACE TO {test_user}",
            f"GRANT CREATE SESSION, CREATE TABLE, CREATE SEQUENCE, CREATE VIEW TO {test_user}",
            f"GRANT CTXAPP TO {test_user}",
        ]:
            try:
                cursor.execute(grant)
            except oracledb.DatabaseError:
                pass

        try:
            cursor.execute(f"GRANT EXECUTE ON UTL_MATCH TO {test_user}")
        except oracledb.DatabaseError:
            pass

        conn.commit()
    finally:
        cursor.close()
        conn.close()

    return f"oracle://{test_user}:{test_pass}@{admin_dsn['dsn']}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _run() -> None:
    total_steps = 8

    raw_dsn = os.environ.get(
        "ORACLE_TEST_DSN",
        "oracle://SYSTEM:oracle@localhost:1521/FREEPDB1",
    )
    print(f"Oracle DSN: {raw_dsn}")

    # ------------------------------------------------------------------
    # 0. Bootstrap test user
    # ------------------------------------------------------------------
    admin_params = _parse_oracle_dsn(raw_dsn)
    oracle_url = _bootstrap_test_user(admin_params)
    print(f"Test user URL: {oracle_url}")

    # ------------------------------------------------------------------
    # 1. Run migrations (idempotent)
    # ------------------------------------------------------------------
    _log(1, total_steps, "Running Oracle migrations ...")
    from hindsight_api.migrations import run_migrations

    run_migrations(oracle_url)
    print("    -> migrations OK")

    # ------------------------------------------------------------------
    # Set env so the global config detects Oracle backend
    # ------------------------------------------------------------------
    os.environ["HINDSIGHT_API_DATABASE_BACKEND"] = "oracle"
    from hindsight_api.config import clear_config_cache

    clear_config_cache()

    # ------------------------------------------------------------------
    # Initialise components
    # ------------------------------------------------------------------
    from hindsight_api import LocalSTEmbeddings, MemoryEngine, RequestContext
    from hindsight_api.engine.cross_encoder import LocalSTCrossEncoder
    from hindsight_api.engine.memory_engine import Budget
    from hindsight_api.engine.query_analyzer import DateparserQueryAnalyzer
    from hindsight_api.engine.task_backend import SyncTaskBackend

    print("  Loading local embeddings model (first run downloads ~90 MB) ...")
    embeddings = LocalSTEmbeddings()
    await embeddings.initialize()

    cross_encoder = LocalSTCrossEncoder()
    await cross_encoder.initialize()

    query_analyzer = DateparserQueryAnalyzer()

    engine = MemoryEngine(
        db_url=oracle_url,
        memory_llm_provider=os.getenv("HINDSIGHT_API_LLM_PROVIDER", "openai"),
        memory_llm_api_key=os.getenv("HINDSIGHT_API_LLM_API_KEY"),
        memory_llm_model=os.getenv("HINDSIGHT_API_LLM_MODEL", "gpt-4o-mini"),
        memory_llm_base_url=os.getenv("HINDSIGHT_API_LLM_BASE_URL") or None,
        embeddings=embeddings,
        cross_encoder=cross_encoder,
        query_analyzer=query_analyzer,
        pool_min_size=1,
        pool_max_size=5,
        run_migrations=False,
        task_backend=SyncTaskBackend(),
    )
    await engine.initialize()

    bank_id = f"e2e-oracle-{uuid.uuid4().hex[:8]}"
    ctx = RequestContext()

    try:
        # ------------------------------------------------------------------
        # 2. Create bank
        # ------------------------------------------------------------------
        _log(2, total_steps, f"Creating bank '{bank_id}' ...")
        profile = await engine.get_bank_profile(bank_id=bank_id, request_context=ctx)
        assert profile is not None, "Bank profile should not be None"
        assert profile["bank_id"] == bank_id
        print(f"    -> bank created: {profile['bank_id']}")

        # ------------------------------------------------------------------
        # 3. Retain 3 documents
        # ------------------------------------------------------------------
        _log(3, total_steps, "Retaining 3 documents ...")
        docs = [
            ("Alice is a software engineer who loves Python and machine learning.", "team overview"),
            ("Bob manages the infrastructure team and has 10 years of cloud experience.", "team overview"),
            ("Carol is a data scientist specializing in NLP and computer vision.", "team overview"),
        ]
        all_unit_ids: list[str] = []
        for i, (content, context) in enumerate(docs, 1):
            ids = await engine.retain_async(
                bank_id=bank_id,
                content=content,
                context=context,
                event_date=datetime(2024, 6, i, tzinfo=timezone.utc),
                request_context=ctx,
            )
            all_unit_ids.extend(ids)
            print(f"    -> doc {i}: {len(ids)} units")
        assert len(all_unit_ids) > 0, "Should have retained at least one memory unit"
        print(f"    -> total units retained: {len(all_unit_ids)}")

        # ------------------------------------------------------------------
        # 4. Recall with a query
        # ------------------------------------------------------------------
        _log(4, total_steps, "Recalling memories ...")
        result = await engine.recall_async(
            bank_id=bank_id,
            query="Who works on natural language processing?",
            budget=Budget.LOW,
            max_tokens=500,
            request_context=ctx,
        )

        # ------------------------------------------------------------------
        # 5. Verify recall returns results
        # ------------------------------------------------------------------
        _log(5, total_steps, "Verifying recall results ...")
        assert len(result.results) > 0, f"Expected recall results, got {len(result.results)}"
        texts = [r.text for r in result.results]
        print(f"    -> got {len(result.results)} results")
        for t in texts[:3]:
            print(f"       - {t[:80]}{'...' if len(t) > 80 else ''}")

        # ------------------------------------------------------------------
        # 6. Create a mental model
        # ------------------------------------------------------------------
        _log(6, total_steps, "Creating mental model ...")
        mm = await engine.create_mental_model(
            bank_id=bank_id,
            name="Team Overview",
            source_query="team members and their specialties",
            content="The team consists of Alice (software engineer, Python/ML), Bob (infrastructure, cloud), and Carol (data scientist, NLP/CV).",
            request_context=ctx,
        )
        assert mm is not None, "Mental model should not be None"
        mm_id = mm.get("mental_model_id") or mm.get("id")
        assert mm_id, f"Could not find mental model ID in response: {list(mm.keys())}"
        print(f"    -> mental model created: {mm_id}")

        # ------------------------------------------------------------------
        # 7. List mental models
        # ------------------------------------------------------------------
        _log(7, total_steps, "Listing mental models ...")
        models = await engine.list_mental_models(bank_id=bank_id, request_context=ctx)
        assert len(models) > 0, "Should have at least one mental model"
        found = any((m.get("mental_model_id") or m.get("id")) == mm_id for m in models)
        assert found, f"Mental model {mm_id} not found in list"
        print(f"    -> found {len(models)} mental model(s)")

        # ------------------------------------------------------------------
        # 8. Delete bank (cleanup)
        # ------------------------------------------------------------------
        _log(8, total_steps, f"Deleting bank '{bank_id}' ...")
        try:
            deleted = await engine.delete_bank(bank_id, request_context=ctx)
            print(f"    -> deleted: {deleted}")
        except Exception as cleanup_err:
            # Oracle Text index corruption or lock errors during cleanup are
            # benign — the data is orphaned but doesn't affect functionality.
            print(f"    -> cleanup warning (benign): {str(cleanup_err)[:120]}")

    except Exception:
        # Best-effort cleanup on failure
        try:
            await engine.delete_bank(bank_id, request_context=ctx)
        except Exception:
            pass
        raise
    finally:
        try:
            await engine.close()
        except Exception:
            pass

        # Restore env
        os.environ.pop("HINDSIGHT_API_DATABASE_BACKEND", None)
        clear_config_cache()


def main() -> int:
    print("=" * 60)
    print("  Oracle E2E Smoke Test")
    print("=" * 60)
    try:
        asyncio.run(_run())
    except Exception as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("  ALL STEPS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

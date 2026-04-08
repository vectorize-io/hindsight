"""
Large-bank recall load test (no LLM).

Populates a synthetic bank using the real retain pipeline with a mocked LLM
for fact extraction, then benchmarks recall latency at production scale.

Usage (run from hindsight-api/):
    cd hindsight-api

    # Generate a small bank (~10K memory units):
    uv run python ../hindsight-dev/benchmarks/perf/recall_perf.py generate \\
        --bank-id recall-perf-small2 --scale small

    # Benchmark recall:
    uv run python ../hindsight-dev/benchmarks/perf/recall_perf.py benchmark \\
        --bank-id recall-perf-small --query "database migration" --iterations 5

    # Clean up:
    uv run python ../hindsight-dev/benchmarks/perf/recall_perf.py clean \\
        --bank-id recall-perf-small
"""

import argparse
import asyncio
import os
import statistics
import time
from typing import Any

# Capture DB URL early before hindsight_api imports trigger dotenv override
# (config.py uses load_dotenv(override=True) which stomps env vars)
_EARLY_DB_URL = os.environ.get("HINDSIGHT_API_DATABASE_URL")

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Fact corpus
# ---------------------------------------------------------------------------

# ~200 entity names spanning people, technologies, and places.
ENTITIES = [
    # People
    "Alice Chen",
    "Bob Martinez",
    "Carol Thompson",
    "David Kim",
    "Eva Rodriguez",
    "Frank Johnson",
    "Grace Liu",
    "Henry Park",
    "Irene Nguyen",
    "James Wilson",
    "Karen Davis",
    "Leo Brown",
    "Mia Patel",
    "Nathan Clark",
    "Olivia Walker",
    "Paul Harris",
    "Quinn Lewis",
    "Rachel Young",
    "Sam Scott",
    "Tina Adams",
    "Uma Turner",
    "Victor Hall",
    "Wendy Allen",
    "Xavier Hill",
    "Yara Wright",
    "Zoe King",
    "Aaron Green",
    "Beth Baker",
    "Chris Nelson",
    "Diana Carter",
    "Ethan Mitchell",
    "Fiona Perez",
    "George Roberts",
    "Hannah Turner",
    "Ivan Phillips",
    "Julia Campbell",
    "Kevin Parker",
    "Laura Evans",
    "Mike Edwards",
    "Nina Collins",
    "Oscar Stewart",
    "Penny Sanchez",
    "Ryan Morris",
    "Sandra Rogers",
    "Tom Reed",
    # Technologies
    "PostgreSQL",
    "Redis",
    "Kubernetes",
    "Docker",
    "Python",
    "Rust",
    "TypeScript",
    "React",
    "FastAPI",
    "GraphQL",
    "gRPC",
    "Kafka",
    "Elasticsearch",
    "Prometheus",
    "Grafana",
    "Terraform",
    "Ansible",
    "Nginx",
    "SQLite",
    "MongoDB",
    "Cassandra",
    "RabbitMQ",
    "Celery",
    "Pandas",
    "NumPy",
    "PyTorch",
    "TensorFlow",
    "OpenAI API",
    "Anthropic API",
    "LangChain",
    "ChromaDB",
    "Pinecone",
    "Weaviate",
    "pgvector",
    "Alembic",
    "SQLAlchemy",
    "asyncpg",
    "Pydantic",
    "pytest",
    "Ruff",
    "GitHub Actions",
    "CircleCI",
    "AWS S3",
    "AWS Lambda",
    "GCP BigQuery",
    "Azure DevOps",
    "Datadog",
    "Sentry",
    "OpenTelemetry",
    "Jaeger",
    # Places / teams / projects
    "San Francisco",
    "New York",
    "Seattle",
    "Austin",
    "London",
    "Berlin",
    "Tokyo",
    "Singapore",
    "Platform Team",
    "Infrastructure Team",
    "Data Science Team",
    "Frontend Team",
    "Backend Team",
    "Security Team",
    "DevOps Team",
    "ML Platform Team",
    "Project Orion",
    "Project Helios",
    "Project Atlas",
    "Project Nexus",
    "Project Titan",
    "Project Echo",
    "Project Phoenix",
    "Hindsight",
    "Memory Engine",
    "Control Plane",
    "Data Warehouse",
    "API Gateway",
    "Auth Service",
    "Billing Service",
    "Search Service",
    "Notification Service",
    "Analytics Dashboard",
    "Admin Console",
    "CI Pipeline",
    "Staging Environment",
    "Production Environment",
    "Development Environment",
    "Load Balancer",
    "Service Mesh",
    "Feature Flag Service",
    "Observability Stack",
    "Data Lake",
    "Event Bus",
    "Message Queue",
    "Cache Layer",
    "CDN",
    "DNS",
    "VPN",
    "SSO",
]

# ~300 fact templates.  Placeholders {E0}..{E4} are filled with entity names
# drawn from ENTITIES using a Zipf-like distribution so that ~20 entities
# recur frequently across templates.
FACT_TEMPLATES = [
    "{E0} deployed a new version of {E1} to {E2} on {E3}.",
    "{E0} reported a performance regression in {E1} affecting {E2}.",
    "{E0} and {E1} completed the migration of {E2} from {E3} to {E4}.",
    "{E0} updated the {E1} configuration to use {E2} for caching.",
    "{E0} reviewed {E1}'s pull request for the {E2} integration.",
    "{E0} noticed that {E1} latency increased after the {E2} upgrade.",
    "{E0} created a dashboard in {E1} to monitor {E2} performance.",
    "{E0} onboarded {E1} to the {E2} platform.",
    "{E0} resolved the {E1} incident that caused {E2} downtime.",
    "{E0} led a design review for {E1} with {E2} and {E3}.",
    "{E0} wrote unit tests for the {E1} module in {E2}.",
    "{E0} configured {E1} rate limiting in {E2} for {E3} endpoints.",
    "{E0} set up {E1} alerts in {E2} for the {E3} service.",
    "{E0} refactored the {E1} layer to use {E2} instead of {E3}.",
    "{E0} mentored {E1} on {E2} best practices.",
    "{E0} presented the {E1} roadmap to {E2} leadership in {E3}.",
    "{E0} integrated {E1} with {E2} for real-time streaming.",
    "{E0} opened a ticket for {E1} memory leak in {E2}.",
    "{E0} benchmarked {E1} against {E2} for the {E3} use case.",
    "{E0} scheduled a migration window for {E1} maintenance in {E2}.",
    "{E0} documented the {E1} API changes for {E2}.",
    "{E0} paired with {E1} to debug the {E2} timeout issue.",
    "{E0} upgraded {E1} from version 3.1 to 4.0 in {E2}.",
    "{E0} provisioned new {E1} instances in {E2} to handle load.",
    "{E0} fixed a data race in {E1} caused by {E2} concurrency.",
    "{E0} added {E1} tracing spans to the {E2} service.",
    "{E0} rotated the {E1} credentials used by {E2}.",
    "{E0} enabled {E1} compression in {E2} to reduce storage costs.",
    "{E0} flagged a security issue in {E1} shared with {E2}.",
    "{E0} ran chaos tests against {E1} in {E2}.",
    "{E0} opened a feature request for {E1} pagination in {E2}.",
    "{E0} trained the {E1} model on data from {E2}.",
    "{E0} synced {E1} state to {E2} using {E3}.",
    "{E0} tuned {E1} connection pool size for {E2} workloads.",
    "{E0} evaluated {E1} vs {E2} for the {E3} project.",
    "{E0} set up {E1} CI pipeline for {E2}.",
    "{E0} migrated {E1} secrets from {E2} to {E3}.",
    "{E0} identified a {E1} bottleneck in the {E2} hot path.",
    "{E0} implemented {E1} retry logic in {E2}.",
    "{E0} reviewed the {E1} threat model with {E2}.",
    "{E0} enabled {E1} audit logging in {E2}.",
    "{E0} scaled {E1} to handle {E2} throughput requirements.",
    "{E0} added {E1} health checks for {E2}.",
    "{E0} published the {E1} release notes for {E2}.",
    "{E0} ran a load test against {E1} using {E2}.",
    "{E0} submitted a PR to add {E1} support to {E2}.",
    "{E0} kicked off {E1} data backfill in {E2}.",
    "{E0} filed a post-mortem for the {E1} outage affecting {E2}.",
    "{E0} enabled {E1} feature flag for {E2} users.",
    "{E0} profiled {E1} memory usage in the {E2} environment.",
    "{E0} opened a {E1} security advisory for {E2}.",
    "{E0} created a {E1} runbook for {E2} on-call rotation.",
    "{E0} shipped a hotfix for {E1} parsing bug in {E2}.",
    "{E0} demoed the {E1} prototype to {E2} stakeholders.",
    "{E0} archived old {E1} indexes in {E2} to free storage.",
    "{E0} configured {E1} TLS termination at {E2}.",
    "{E0} added {E1} request validation to the {E2} API.",
    "{E0} updated {E1} Helm charts for {E2} deployment.",
    "{E0} set up {E1} blue-green deployment for {E2}.",
    "{E0} onboarded {E1} as a dependency for {E2}.",
    "{E0} improved {E1} query performance by 40% in {E2}.",
    "{E0} compiled a report on {E1} adoption across {E2}.",
    "{E0} fixed {E1} deadlock under high concurrency in {E2}.",
    "{E0} restarted {E1} to clear stale state in {E2}.",
    "{E0} paired {E1} with {E2} to unblock {E3} migration.",
    "{E0} shipped the {E1} v2 API for {E2} consumers.",
    "{E0} reduced {E1} cold start time by optimizing {E2} imports.",
    "{E0} enabled {E1} dark launch for {E2} traffic.",
    "{E0} set up {E1} canary release for {E2}.",
    "{E0} resolved {E1} certificate expiry alert for {E2}.",
    "{E0} added {E1} caching layer to reduce {E2} load.",
    "{E0} restructured {E1} schema in {E2} for performance.",
    "{E0} published {E1} metrics to {E2} for alerting.",
    "{E0} opened a discussion on {E1} alternatives for {E2}.",
    "{E0} implemented {E1} circuit breaker pattern in {E2}.",
    "{E0} rolled back {E1} after breaking changes in {E2}.",
    "{E0} enabled {E1} distributed tracing across {E2} services.",
    "{E0} ran a {E1} audit to identify {E2} vulnerabilities.",
    "{E0} deprecated {E1} endpoints in {E2} for {E3}.",
    "{E0} tested {E1} failover behavior in {E2}.",
    "{E0} enabled {E1} CORS configuration in {E2}.",
    "{E0} created {E1} cost allocation tags in {E2}.",
    "{E0} optimized {E1} batch processing in {E2}.",
    "{E0} added {E1} pagination to the {E2} listing endpoint.",
    "{E0} integrated {E1} SSO with {E2}.",
    "{E0} shipped {E1} feature for {E2} enterprise customers.",
    "{E0} updated {E1} dependencies to fix {E2} vulnerabilities.",
    "{E0} presented {E1} capacity plan for {E2} growth.",
    "{E0} configured {E1} auto-scaling for {E2}.",
    "{E0} generated {E1} API client for {E2} consumers.",
    "{E0} added {E1} soft-delete support to {E2}.",
    "{E0} resolved {E1} DNS resolution failure in {E2}.",
    "{E0} deployed {E1} across three regions starting with {E2}.",
    "{E0} benchmarked {E1} embedding throughput for {E2}.",
    "{E0} set up {E1} read replicas in {E2} to offload load.",
    "{E0} cleaned up {E1} orphaned resources in {E2}.",
    "{E0} merged the {E1} feature branch into {E2} main.",
    "{E0} performed a {E1} code review for {E2} security standards.",
    "{E0} generated synthetic data using {E1} for {E2} tests.",
    "{E0} integrated {E1} observability into {E2} pipeline.",
    "{E0} added {E1} retry budget to the {E2} client.",
    "{E0} compressed {E1} backups stored in {E2}.",
    "{E0} enabled {E1} slow query logging in {E2}.",
    "{E0} resolved {E1} config drift between {E2} environments.",
    "{E0} automated {E1} provisioning with {E2} scripts.",
    "{E0} fixed {E1} pagination bug in the {E2} API.",
    "{E0} increased {E1} timeout from 30s to 60s in {E2}.",
    "{E0} set up {E1} cross-region replication for {E2}.",
    "{E0} migrated {E1} workloads from {E2} to {E3}.",
    "{E0} reviewed {E1} schema change proposal for {E2}.",
    "{E0} shipped {E1} structured logging for {E2}.",
    "{E0} ran {E1} end-to-end tests against {E2}.",
    "{E0} added {E1} graceful shutdown to {E2} workers.",
    "{E0} investigated {E1} anomaly detected in {E2} metrics.",
    "{E0} enabled {E1} write-ahead logging in {E2}.",
    "{E0} decommissioned legacy {E1} in favor of {E2}.",
    "{E0} added {E1} vector index to {E2} for similarity search.",
    "{E0} trained {E1} classifier on {E2} labeled dataset.",
    "{E0} profiled {E1} CPU usage spike in {E2}.",
    "{E0} set up {E1} webhook integration for {E2} events.",
    "{E0} implemented {E1} rate limiting using {E2}.",
    "{E0} exported {E1} traces to {E2} for analysis.",
    "{E0} configured {E1} connection pooling for {E2}.",
    "{E0} updated {E1} documentation with {E2} examples.",
    "{E0} resolved {E1} type mismatch between {E2} versions.",
    "{E0} shipped {E1} bulk import feature for {E2}.",
    "{E0} enabled {E1} query caching in {E2}.",
    "{E0} ran security scan on {E1} container images for {E2}.",
    "{E0} added {E1} idempotency keys to {E2} endpoints.",
    "{E0} migrated {E1} logs to {E2} for centralized search.",
    "{E0} opened discussion on {E1} retention policy in {E2}.",
    "{E0} tagged {E1} release candidate for {E2} deployment.",
    "{E0} configured {E1} resource quotas in {E2} namespace.",
    "{E0} opened a {E1} incident for {E2} degradation.",
    "{E0} validated {E1} schema migration on {E2} staging.",
    "{E0} submitted {E1} change request for {E2} production.",
    "{E0} identified {E1} as critical dependency for {E2}.",
    "{E0} refactored {E1} plugin interface for {E2}.",
    "{E0} enabled {E1} delta compression for {E2} exports.",
    "{E0} built {E1} smoke tests for {E2} deployment checks.",
    "{E0} resolved {E1} clock skew issue in {E2} cluster.",
    "{E0} set up {E1} chaos mesh for {E2} resilience testing.",
    "{E0} ran {E1} migration dry-run against {E2} production data.",
    "{E0} added {E1} custom metrics to {E2} dashboards.",
    "{E0} enabled {E1} multi-region failover for {E2}.",
    "{E0} tested {E1} rollback procedure for {E2}.",
    "{E0} audited {E1} access logs for {E2}.",
    "{E0} updated {E1} routing rules in {E2}.",
    "{E0} added {E1} API versioning to {E2}.",
    "{E0} created {E1} architecture diagram for {E2}.",
    "{E0} configured {E1} resource limits for {E2} pods.",
    "{E0} shipped {E1} streaming response for {E2} endpoints.",
    "{E0} added {E1} content negotiation to {E2}.",
    "{E0} enabled {E1} request signing for {E2}.",
    "{E0} fixed {E1} goroutine leak in {E2}.",
    "{E0} tuned {E1} GC parameters for {E2}.",
    "{E0} deployed {E1} hotfix to unblock {E2}.",
    "{E0} ran capacity review for {E1} ahead of {E2} launch.",
    "{E0} configured {E1} alerting thresholds for {E2}.",
    "{E0} migrated {E1} config to environment variables in {E2}.",
    "{E0} hardened {E1} container image for {E2}.",
    "{E0} added {E1} exponential backoff to {E2} client.",
    "{E0} enabled {E1} connection keep-alive in {E2}.",
    "{E0} built {E1} synthetic monitor for {E2}.",
    "{E0} added {E1} compression to {E2} API responses.",
    "{E0} enabled {E1} query explain plans in {E2}.",
    "{E0} reduced {E1} package size in {E2} by tree-shaking.",
    "{E0} added {E1} dark mode support to {E2}.",
    "{E0} configured {E1} output caching in {E2}.",
    "{E0} submitted {E1} benchmarks comparing {E2} and {E3}.",
    "{E0} enabled {E1} mTLS between {E2} and {E3}.",
    "{E0} integrated {E1} error tracking into {E2}.",
    "{E0} enabled {E1} feature gates for {E2} beta users.",
    "{E0} shipped {E1} analytics events for {E2} funnel.",
    "{E0} identified {E1} as cause of {E2} tail latency.",
    "{E0} standardized {E1} logging format across {E2}.",
    "{E0} ran {E1} regression tests before {E2} release.",
    "{E0} set up {E1} PR preview environments for {E2}.",
    "{E0} fixed {E1} index missing in {E2} query.",
    "{E0} enabled {E1} statement timeout in {E2}.",
    "{E0} reviewed {E1} data model with {E2} data team.",
    "{E0} refactored {E1} middleware stack in {E2}.",
    "{E0} ran {E1} fuzzing tests against {E2}.",
    "{E0} set {E1} memory limits for {E2} workers.",
    "{E0} added {E1} observability hooks to {E2}.",
    "{E0} resolved {E1} permission issue between {E2} and {E3}.",
    "{E0} enabled {E1} auto-vacuum in {E2}.",
    "{E0} shipped {E1} batch delete API for {E2}.",
    "{E0} added {E1} soft-delete flag to {E2} records.",
    "{E0} created {E1} integration test suite for {E2}.",
    "{E0} configured {E1} load shedding in {E2}.",
    "{E0} set up {E1} incident response playbook for {E2}.",
    "{E0} fixed {E1} N+1 query in {E2} listing endpoint.",
    "{E0} validated {E1} SLOs for {E2} over the past quarter.",
    "{E0} enabled {E1} continuous profiling in {E2}.",
    "{E0} configured {E1} service discovery for {E2}.",
    "{E0} deployed {E1} update with zero downtime to {E2}.",
    "{E0} ran {E1} penetration test against {E2}.",
    "{E0} added {E1} DKIM signing for {E2} emails.",
    "{E0} resolved {E1} OOM kill in {E2} under load.",
    "{E0} tuned {E1} work queue parallelism for {E2}.",
    "{E0} deployed {E1} read replica for {E2} reporting.",
    "{E0} shipped {E1} event replay feature for {E2}.",
    "{E0} added {E1} schema registry support to {E2}.",
    "{E0} configured {E1} dead letter queue for {E2}.",
    "{E0} implemented {E1} RBAC for {E2} admin endpoints.",
    "{E0} enabled {E1} mutual authentication for {E2}.",
    "{E0} profiled {E1} GC pressure in {E2}.",
    "{E0} ran {E1} compliance audit for {E2} data.",
    "{E0} added {E1} multi-tenant isolation to {E2}.",
    "{E0} configured {E1} network policies for {E2}.",
    "{E0} shipped {E1} webhook retry logic for {E2}.",
    "{E0} enabled {E1} prepared statements in {E2}.",
    "{E0} deployed {E1} for zero-trust networking in {E2}.",
    "{E0} enabled {E1} query result caching in {E2}.",
    "{E0} added {E1} circuit breaker to {E2} outbound calls.",
    "{E0} ran {E1} disaster recovery drill for {E2}.",
    "{E0} tuned {E1} thread pool for {E2} workload.",
    "{E0} shipped {E1} export endpoint for {E2} data.",
    "{E0} configured {E1} anomaly detection in {E2}.",
    "{E0} set up {E1} chaos experiment for {E2} resilience.",
    "{E0} added {E1} distributed lock to {E2} cron jobs.",
    "{E0} migrated {E1} codebase from {E2} to {E3}.",
    "{E0} enabled {E1} structured error responses in {E2}.",
    "{E0} shipped {E1} async processing for {E2} heavy tasks.",
    "{E0} verified {E1} data integrity after {E2} migration.",
    "{E0} added {E1} request deduplication to {E2}.",
    "{E0} configured {E1} log sampling in {E2}.",
    "{E0} enabled {E1} query parallelism in {E2}.",
    "{E0} shipped {E1} data masking for {E2} PII fields.",
    "{E0} reviewed {E1} deployment procedure for {E2}.",
    "{E0} fixed {E1} connection leak under {E2} high load.",
    "{E0} set up {E1} fan-out pattern for {E2} events.",
    "{E0} enabled {E1} request coalescing in {E2}.",
    "{E0} shipped {E1} GraphQL federation for {E2}.",
    "{E0} enabled {E1} distributed caching for {E2}.",
    "{E0} ran {E1} smoke test suite after {E2} deploy.",
    "{E0} resolved {E1} config injection issue in {E2}.",
    "{E0} shipped {E1} multi-region write support for {E2}.",
]

# Scale configuration: number of content items to submit as a single async batch
SCALES = {
    "tiny": 1,
    "mini": 50,
    "small": 2_000,
    "medium": 10_000,
    "large": 33_000,
    "very-large": 100_000,
}

# ---------------------------------------------------------------------------
# Zipf-like entity selector
# ---------------------------------------------------------------------------


def _make_entity_selector(seed: int = 42) -> "Callable[[int], list[str]]":
    """Return a function that draws N entity names with Zipf-like distribution."""
    import random

    rng = random.Random(seed)
    n = len(ENTITIES)
    # Weights: entity i gets weight 1/(i+1)
    weights = [1.0 / (i + 1) for i in range(n)]

    def pick(count: int) -> list[str]:
        seen = set()
        result = []
        while len(result) < count:
            choice = rng.choices(ENTITIES, weights=weights, k=1)[0]
            if choice not in seen:
                seen.add(choice)
                result.append(choice)
        return result

    return pick


_pick_entities = _make_entity_selector()


def _fill_template(template: str) -> str:
    """Replace {E0}..{E4} placeholders in a template with entity names."""
    placeholders = [f"{{E{i}}}" for i in range(5)]
    needed = sum(1 for p in placeholders if p in template)
    if needed == 0:
        return template
    entities = _pick_entities(needed)
    result = template
    for i, entity in enumerate(entities):
        result = result.replace(f"{{E{i}}}", entity)
    return result


# ---------------------------------------------------------------------------
# Mock LLM callback
# ---------------------------------------------------------------------------

from collections.abc import Callable  # noqa: E402  (after stdlib)


def _make_fact_callback() -> tuple[Callable[[list[dict], str], Any], list[int]]:
    """
    Return (callback, call_counter) where call_counter[0] tracks invocations.

    The callback cycles through FACT_TEMPLATES and returns a valid
    FactExtractionResponse-compatible dict for the retain pipeline.
    """
    call_counter = [0]

    # Realistic fact_type distribution matching production observations:
    # ~60% world, ~30% experience, ~10% mental_model
    _FACT_TYPE_CYCLE = (["world"] * 6 + ["experience"] * 3 + ["mental_model"] * 1) * 10  # 100-element cycle

    def callback(messages: list[dict], scope: str) -> Any:
        if scope == "retain_extract_facts":
            idx = call_counter[0] % len(FACT_TEMPLATES)
            fact_type = _FACT_TYPE_CYCLE[call_counter[0] % len(_FACT_TYPE_CYCLE)]
            call_counter[0] += 1
            template = FACT_TEMPLATES[idx]
            fact_text = _fill_template(template)
            # Extract a few entity names from the filled text to populate entities
            # field (very rough — enough to drive entity link creation)
            entity_names = [e for e in ENTITIES if e in fact_text][:3]
            entities = [{"text": e} for e in entity_names]
            return {
                "facts": [
                    {
                        "what": fact_text,
                        "when": "N/A",
                        "where": "N/A",
                        "who": "N/A",
                        "why": "N/A",
                        "fact_type": fact_type,
                        "entities": entities,
                    }
                ]
            }
        # All other scopes (entity resolution, etc.) — return empty
        return {"facts": []}

    return callback, call_counter


# ---------------------------------------------------------------------------
# Engine helpers
# ---------------------------------------------------------------------------


def _build_engine(*, disable_observations: bool = False) -> "Any":
    """Create a MemoryEngine using mock LLM and DB from env."""
    from hindsight_api import MemoryEngine

    db_url = _EARLY_DB_URL or os.getenv("HINDSIGHT_API_DATABASE_URL", "pg0")
    if disable_observations:
        os.environ["HINDSIGHT_API_ENABLE_OBSERVATIONS"] = "false"
    engine = MemoryEngine(
        db_url=db_url,
        memory_llm_provider="mock",
        memory_llm_api_key="",
        memory_llm_model="mock",
        skip_llm_verification=True,
        db_command_timeout=600,  # Long timeout needed for large-bank inserts
    )
    return engine


# ---------------------------------------------------------------------------
# Synthetic observation insertion
# ---------------------------------------------------------------------------

_BATCH_SIZE = 500


async def _insert_synthetic_observations(pool: Any, bank_id: str) -> int:
    """
    For every non-observation memory unit in *bank_id*, insert one synthetic
    observation with the same text, embedding, and tags, pointing back to that
    unit as its sole source fact.

    Returns the number of observations inserted.
    """
    import uuid

    from hindsight_api.engine.task_backend import fq_table

    table = fq_table("memory_units")

    # Fetch all non-observation units
    rows = await pool.fetch(
        f"""
        SELECT id, text, embedding, tags, event_date, occurred_start, occurred_end, mentioned_at
        FROM {table}
        WHERE bank_id = $1 AND fact_type != 'observation'
        ORDER BY id
        """,
        bank_id,
    )

    if not rows:
        return 0

    inserted = 0
    for offset in range(0, len(rows), _BATCH_SIZE):
        batch = rows[offset : offset + _BATCH_SIZE]
        await pool.executemany(
            f"""
            INSERT INTO {table} (
                id, bank_id, text, fact_type, embedding,
                proof_count, source_memory_ids, history,
                tags, event_date, occurred_start, occurred_end, mentioned_at
            ) VALUES (
                $1, $2, $3, 'observation', $4::vector,
                1, ARRAY[$5::uuid], '[]'::jsonb,
                $6, $7, $8, $9, $10
            )
            ON CONFLICT DO NOTHING
            """,
            [
                (
                    uuid.uuid4(),  # new observation id
                    bank_id,  # bank_id
                    row["text"],
                    row["embedding"],
                    row["id"],  # source fact id
                    row["tags"] or [],
                    row["event_date"],
                    row["occurred_start"],
                    row["occurred_end"],
                    row["mentioned_at"],
                )
                for row in batch
            ],
        )
        inserted += len(batch)

    return inserted


# ---------------------------------------------------------------------------
# Subcommand: generate
# ---------------------------------------------------------------------------


async def _wait_for_operation(pool: Any, operation_id: str, timeout: float = 86400.0) -> str:
    """
    Poll async_operations every second until the parent reaches completed or failed.

    Raises immediately if:
    - the parent itself reaches 'failed'
    - any direct child operation reaches 'failed'
    """
    import uuid

    from hindsight_api.engine.task_backend import fq_table

    table = fq_table("async_operations")
    deadline = asyncio.get_event_loop().time() + timeout
    parent_uuid = uuid.UUID(operation_id)

    while asyncio.get_event_loop().time() < deadline:
        # Check parent status
        row = await pool.fetchrow(
            f"SELECT status, error_message FROM {table} WHERE operation_id = $1",
            parent_uuid,
        )
        if row:
            if row["status"] == "completed":
                return "completed"
            if row["status"] == "failed":
                raise RuntimeError(f"Operation {operation_id} failed: {row['error_message'] or 'unknown error'}")

        # Fast-fail: any direct child that has already failed
        failed_child = await pool.fetchrow(
            f"""
            SELECT operation_id, error_message
            FROM {table}
            WHERE result_metadata::jsonb->>'parent_operation_id' = $1
              AND status = 'failed'
            LIMIT 1
            """,
            operation_id,
        )
        if failed_child:
            err = failed_child["error_message"] or "unknown error"
            raise RuntimeError(f"Child operation {failed_child['operation_id']} failed: {err}")

        await asyncio.sleep(1.0)

    raise TimeoutError(f"Operation {operation_id} did not complete within {timeout}s")


async def cmd_generate(bank_id: str, scale: str, workers: int = 16, with_observations: bool = False) -> None:
    """Submit all content as a single async batch and process with an in-process worker."""
    from hindsight_api.models import RequestContext
    from hindsight_api.worker.poller import WorkerPoller

    total_items = SCALES[scale]
    console.print(
        f"\n[bold cyan]Generate[/bold cyan] bank=[bold]{bank_id}[/bold] scale=[bold]{scale}[/bold] workers=[bold]{workers}[/bold]"
    )
    console.print(f"  Total items : {total_items:,}\n")

    engine = _build_engine(disable_observations=True)
    await engine.initialize()

    # Attach mock callback to retain LLM config
    callback, call_counter = _make_fact_callback()
    engine._retain_llm_config.set_response_callback(callback)
    engine._llm_config.set_response_callback(callback)

    # Build all content items upfront
    all_contents = [{"content": _fill_template(FACT_TEMPLATES[i % len(FACT_TEMPLATES)])} for i in range(total_items)]

    # Submit the whole batch as a single async operation (auto-splits by token budget)
    console.print(f"  Submitting {total_items:,} items as async batch…")
    result = await engine.submit_async_retain(
        bank_id=bank_id,
        contents=all_contents,
        request_context=RequestContext(),
    )
    operation_id = result["operation_id"]
    console.print(f"  Operation  : {operation_id}")

    # Start in-process worker to drain the queue
    pool = await engine._get_pool()
    poller = WorkerPoller(
        pool=pool,
        worker_id="recall-perf-worker",
        executor=engine.execute_task,
        poll_interval_ms=200,
        max_slots=workers,
        consolidation_max_slots=0,
    )
    poller_task = asyncio.create_task(poller.run())
    console.print("  Worker     : started\n")

    # Wait for the parent operation to reach a terminal state
    t0 = time.perf_counter()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        progress.add_task("Processing async retain…")
        final_status = await _wait_for_operation(pool, operation_id)

    elapsed = time.perf_counter() - t0

    # Graceful shutdown: stop accepting new tasks and wait for all in-flight
    # tasks to complete (including post-transaction flush_pending_stats).
    await poller.shutdown_graceful(timeout=60.0)
    poller_task.cancel()
    try:
        await poller_task
    except asyncio.CancelledError:
        pass

    status_color = "green" if final_status == "completed" else "red"
    console.print(
        f"\n[{status_color}]Done[/{status_color}] — status=[bold]{final_status}[/bold] "
        f"in {elapsed:.1f}s  ({total_items / elapsed:.0f} items/s)"
    )
    console.print(f"LLM callback invoked {call_counter[0]:,} times.")

    if with_observations:
        console.print("\n  Inserting synthetic observations (1 per fact)…")
        t_obs = time.perf_counter()
        n_obs = await _insert_synthetic_observations(pool, bank_id)
        elapsed_obs = time.perf_counter() - t_obs
        console.print(f"  Inserted {n_obs:,} observations in {elapsed_obs:.1f}s")

    await pool.close()


# ---------------------------------------------------------------------------
# RRF-only reranker (bypasses cross-encoder for DB-focused benchmarking)
# ---------------------------------------------------------------------------


class _RRFReranker:
    """
    Drop-in replacement for CrossEncoderReranker that uses RRF scores only.

    Eliminates cross-encoder (CPU-bound ML inference) so recall timings
    reflect pure DB interaction costs.
    """

    async def ensure_initialized(self) -> None:
        pass

    async def rerank(self, query: str, candidates: list) -> list:
        from hindsight_api.engine.search.types import ScoredResult

        scored = [ScoredResult(candidate=c, weight=c.rrf_score) for c in candidates]
        scored.sort(key=lambda x: x.weight, reverse=True)
        return scored


# ---------------------------------------------------------------------------
# Subcommand: benchmark
# ---------------------------------------------------------------------------


async def cmd_benchmark(
    bank_id: str, query: str, iterations: int, concurrency: int, reranker: str, fact_types: list[str] | None = None
) -> None:
    """Run recall in parallel and report p50/p95/p99 timings with per-step breakdown."""
    from hindsight_api.models import RequestContext

    console.print(f"\n[bold cyan]Benchmark[/bold cyan] bank=[bold]{bank_id}[/bold]")
    console.print(f"  Query       : {query}")
    console.print(f"  Iterations  : {iterations}  (total recall calls)")
    console.print(f"  Concurrency : {concurrency}")
    console.print(f"  Reranker    : {reranker}")
    console.print(f"  Fact types  : {', '.join(fact_types) if fact_types else 'all'}\n")

    engine = _build_engine()
    await engine.initialize()

    if reranker == "rrf":
        engine._cross_encoder_reranker = _RRFReranker()

    request_context = RequestContext()
    durations: list[float] = []
    all_phase_timings: dict[str, list[float]] = {}

    async def recall_one() -> float:
        from hindsight_api.engine.memory_engine import Budget

        t0 = time.perf_counter()
        result = await engine.recall_async(
            bank_id=bank_id,
            query=query,
            budget=Budget.HIGH,
            max_tokens=4096,
            enable_trace=True,
            include_chunks=True,
            include_entities=True,
            include_source_facts=True,
            fact_type=fact_types,
            request_context=request_context,
            _quiet=True,
        )
        elapsed = time.perf_counter() - t0
        if result.trace:
            summary = result.trace.get("summary", {})
            for pm in summary.get("phase_metrics", []):
                name = pm["phase_name"]
                dur = pm["duration_seconds"]
                all_phase_timings.setdefault(name, []).append(dur)
        return elapsed

    # Run in parallel batches of `concurrency` until `iterations` total calls are done
    remaining = iterations
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running recall…", total=iterations)

        while remaining > 0:
            batch_size = min(concurrency, remaining)
            batch = await asyncio.gather(*[recall_one() for _ in range(batch_size)])
            durations.extend(batch)
            remaining -= batch_size
            progress.advance(task, batch_size)

    pool = await engine._get_pool()
    await pool.close()

    # Compute percentiles
    sorted_d = sorted(durations)
    n = len(sorted_d)

    def pct(p: float) -> float:
        idx = min(int(p / 100 * n), n - 1)
        return sorted_d[idx]

    table = Table(title=f"Recall Latency — bank={bank_id!r}  query={query!r}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Total calls", str(n))
    table.add_row("Concurrency", str(concurrency))
    table.add_row("Mean", f"{statistics.mean(durations):.3f}s")
    table.add_row("p50", f"{pct(50):.3f}s")
    table.add_row("p95", f"{pct(95):.3f}s")
    table.add_row("p99", f"{pct(99):.3f}s")
    table.add_row("Max", f"{max(durations):.3f}s")
    table.add_row("Min", f"{min(durations):.3f}s")

    console.print("\n")
    console.print(table)

    if all_phase_timings:
        phase_table = Table(title="Per-Step Timing Breakdown (across all calls)")
        phase_table.add_column("Step", style="cyan")
        phase_table.add_column("Mean", style="green", justify="right")
        phase_table.add_column("p50", style="green", justify="right")
        phase_table.add_column("p95", style="yellow", justify="right")
        phase_table.add_column("Max", style="red", justify="right")

        # Sort by mean duration descending so the bottleneck is at the top
        sorted_phases = sorted(all_phase_timings.items(), key=lambda x: statistics.mean(x[1]), reverse=True)
        for name, times in sorted_phases:
            st = sorted(times)
            m = statistics.mean(times)
            p50_v = st[min(int(0.5 * len(st)), len(st) - 1)]
            p95_v = st[min(int(0.95 * len(st)), len(st) - 1)]
            mx = max(times)
            phase_table.add_row(name, f"{m:.3f}s", f"{p50_v:.3f}s", f"{p95_v:.3f}s", f"{mx:.3f}s")

        console.print(phase_table)


# ---------------------------------------------------------------------------
# Subcommand: stats
# ---------------------------------------------------------------------------


async def cmd_stats(bank_ids: list[str]) -> None:
    """Print memory / entity / link counts for one or more banks."""
    from hindsight_api.models import RequestContext

    engine = _build_engine()
    await engine.initialize()
    ctx = RequestContext()

    table = Table(title="Bank Statistics")
    table.add_column("Bank ID", style="cyan")
    table.add_column("Units", style="green", justify="right")
    table.add_column("Links (total)", style="yellow", justify="right")
    table.add_column("Links by type", style="white")

    for bank_id in bank_ids:
        try:
            stats = await engine.get_bank_stats(bank_id=bank_id, request_context=ctx)
            total_units = sum(stats.get("node_counts", {}).values())
            total_links = sum(stats.get("link_counts", {}).values())
            links_detail = "  ".join(f"{k}={v:,}" for k, v in sorted(stats.get("link_counts", {}).items()))
            table.add_row(bank_id, f"{total_units:,}", "-", f"{total_links:,}", links_detail)
        except Exception as e:
            table.add_row(bank_id, "ERROR", "", "", str(e))

    pool = await engine._get_pool()
    await pool.close()

    console.print("\n")
    console.print(table)


# ---------------------------------------------------------------------------
# Subcommand: clean
# ---------------------------------------------------------------------------


async def cmd_clean(bank_id: str) -> None:
    """Delete all data for a bank."""
    from hindsight_api.models import RequestContext

    console.print(f"\n[bold red]Clean[/bold red] bank=[bold]{bank_id}[/bold]")

    engine = _build_engine()
    await engine.initialize()

    result = await engine.delete_bank(bank_id=bank_id, request_context=RequestContext())

    pool = await engine._get_pool()
    await pool.close()

    table = Table(title=f"Deleted from bank={bank_id!r}")
    table.add_column("Table", style="cyan")
    table.add_column("Rows deleted", style="red", justify="right")
    for k, v in result.items():
        table.add_row(k, str(v))
    console.print("\n")
    console.print(table)
    console.print("\n[green]Done.[/green]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Large-bank recall load test (no LLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # generate
    gen = sub.add_parser("generate", help="Populate a synthetic bank")
    gen.add_argument("--bank-id", required=True)
    gen.add_argument("--scale", choices=list(SCALES), default="small")
    gen.add_argument("--workers", type=int, default=8, help="Max concurrent worker slots (default: 8)")
    gen.add_argument(
        "--with-observations",
        action="store_true",
        default=False,
        help="After retain, insert one synthetic observation per fact (same text, same embedding)",
    )

    # benchmark
    bm = sub.add_parser("benchmark", help="Run recall and report latency")
    bm.add_argument("--bank-id", required=True)
    bm.add_argument("--query", required=True)
    bm.add_argument("--iterations", type=int, default=10, help="Total number of recall calls (default: 10)")
    bm.add_argument("--concurrency", type=int, default=1, help="Parallel recalls per batch (default: 1)")
    bm.add_argument(
        "--reranker",
        choices=["rrf", "cross-encoder"],
        default="rrf",
        help="Reranker to use: rrf=RRF scores only (no ML), cross-encoder=neural reranker (default: rrf)",
    )
    bm.add_argument(
        "--fact-types",
        nargs="+",
        choices=["world", "experience", "observation"],
        default=None,
        metavar="TYPE",
        help="Fact types to include in recall (default: all). E.g. --fact-types observation",
    )

    # stats
    st = sub.add_parser("stats", help="Print memory/entity/link counts for banks")
    st.add_argument("bank_ids", nargs="+", metavar="BANK_ID")

    # clean
    cl = sub.add_parser("clean", help="Delete all data for a bank")
    cl.add_argument("--bank-id", required=True)

    args = parser.parse_args()

    if args.cmd == "generate":
        asyncio.run(
            cmd_generate(args.bank_id, args.scale, workers=args.workers, with_observations=args.with_observations)
        )
    elif args.cmd == "benchmark":
        asyncio.run(
            cmd_benchmark(args.bank_id, args.query, args.iterations, args.concurrency, args.reranker, args.fact_types)
        )
    elif args.cmd == "stats":
        asyncio.run(cmd_stats(args.bank_ids))
    elif args.cmd == "clean":
        asyncio.run(cmd_clean(args.bank_id))


if __name__ == "__main__":
    main()

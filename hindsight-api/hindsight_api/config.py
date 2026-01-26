"""
Centralized configuration for Hindsight API.

All environment variables and their defaults are defined here.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

from dotenv import find_dotenv, load_dotenv

# Load .env file, searching current and parent directories (overrides existing env vars)
load_dotenv(find_dotenv(usecwd=True), override=True)

logger = logging.getLogger(__name__)

# Environment variable names
ENV_DATABASE_URL = "HINDSIGHT_API_DATABASE_URL"
ENV_LLM_PROVIDER = "HINDSIGHT_API_LLM_PROVIDER"
ENV_LLM_API_KEY = "HINDSIGHT_API_LLM_API_KEY"
ENV_LLM_MODEL = "HINDSIGHT_API_LLM_MODEL"
ENV_LLM_BASE_URL = "HINDSIGHT_API_LLM_BASE_URL"
ENV_LLM_MAX_CONCURRENT = "HINDSIGHT_API_LLM_MAX_CONCURRENT"
ENV_LLM_TIMEOUT = "HINDSIGHT_API_LLM_TIMEOUT"
ENV_LLM_GROQ_SERVICE_TIER = "HINDSIGHT_API_LLM_GROQ_SERVICE_TIER"

# Per-operation LLM configuration (optional, falls back to global LLM config)
ENV_RETAIN_LLM_PROVIDER = "HINDSIGHT_API_RETAIN_LLM_PROVIDER"
ENV_RETAIN_LLM_API_KEY = "HINDSIGHT_API_RETAIN_LLM_API_KEY"
ENV_RETAIN_LLM_MODEL = "HINDSIGHT_API_RETAIN_LLM_MODEL"
ENV_RETAIN_LLM_BASE_URL = "HINDSIGHT_API_RETAIN_LLM_BASE_URL"

ENV_REFLECT_LLM_PROVIDER = "HINDSIGHT_API_REFLECT_LLM_PROVIDER"
ENV_REFLECT_LLM_API_KEY = "HINDSIGHT_API_REFLECT_LLM_API_KEY"
ENV_REFLECT_LLM_MODEL = "HINDSIGHT_API_REFLECT_LLM_MODEL"
ENV_REFLECT_LLM_BASE_URL = "HINDSIGHT_API_REFLECT_LLM_BASE_URL"

ENV_CONSOLIDATION_LLM_PROVIDER = "HINDSIGHT_API_CONSOLIDATION_LLM_PROVIDER"
ENV_CONSOLIDATION_LLM_API_KEY = "HINDSIGHT_API_CONSOLIDATION_LLM_API_KEY"
ENV_CONSOLIDATION_LLM_MODEL = "HINDSIGHT_API_CONSOLIDATION_LLM_MODEL"
ENV_CONSOLIDATION_LLM_BASE_URL = "HINDSIGHT_API_CONSOLIDATION_LLM_BASE_URL"

ENV_EMBEDDINGS_PROVIDER = "HINDSIGHT_API_EMBEDDINGS_PROVIDER"
ENV_EMBEDDINGS_LOCAL_MODEL = "HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL"
ENV_EMBEDDINGS_TEI_URL = "HINDSIGHT_API_EMBEDDINGS_TEI_URL"
ENV_EMBEDDINGS_OPENAI_API_KEY = "HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY"
ENV_EMBEDDINGS_OPENAI_MODEL = "HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL"
ENV_EMBEDDINGS_OPENAI_BASE_URL = "HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL"

ENV_COHERE_API_KEY = "HINDSIGHT_API_COHERE_API_KEY"
ENV_EMBEDDINGS_COHERE_MODEL = "HINDSIGHT_API_EMBEDDINGS_COHERE_MODEL"
ENV_EMBEDDINGS_COHERE_BASE_URL = "HINDSIGHT_API_EMBEDDINGS_COHERE_BASE_URL"
ENV_RERANKER_COHERE_MODEL = "HINDSIGHT_API_RERANKER_COHERE_MODEL"
ENV_RERANKER_COHERE_BASE_URL = "HINDSIGHT_API_RERANKER_COHERE_BASE_URL"

# LiteLLM gateway configuration (for embeddings and reranker via LiteLLM proxy)
ENV_LITELLM_API_BASE = "HINDSIGHT_API_LITELLM_API_BASE"
ENV_LITELLM_API_KEY = "HINDSIGHT_API_LITELLM_API_KEY"
ENV_EMBEDDINGS_LITELLM_MODEL = "HINDSIGHT_API_EMBEDDINGS_LITELLM_MODEL"
ENV_RERANKER_LITELLM_MODEL = "HINDSIGHT_API_RERANKER_LITELLM_MODEL"

ENV_RERANKER_PROVIDER = "HINDSIGHT_API_RERANKER_PROVIDER"
ENV_RERANKER_LOCAL_MODEL = "HINDSIGHT_API_RERANKER_LOCAL_MODEL"
ENV_RERANKER_LOCAL_MAX_CONCURRENT = "HINDSIGHT_API_RERANKER_LOCAL_MAX_CONCURRENT"
ENV_RERANKER_TEI_URL = "HINDSIGHT_API_RERANKER_TEI_URL"
ENV_RERANKER_TEI_BATCH_SIZE = "HINDSIGHT_API_RERANKER_TEI_BATCH_SIZE"
ENV_RERANKER_TEI_MAX_CONCURRENT = "HINDSIGHT_API_RERANKER_TEI_MAX_CONCURRENT"
ENV_RERANKER_MAX_CANDIDATES = "HINDSIGHT_API_RERANKER_MAX_CANDIDATES"
ENV_RERANKER_FLASHRANK_MODEL = "HINDSIGHT_API_RERANKER_FLASHRANK_MODEL"
ENV_RERANKER_FLASHRANK_CACHE_DIR = "HINDSIGHT_API_RERANKER_FLASHRANK_CACHE_DIR"

ENV_HOST = "HINDSIGHT_API_HOST"
ENV_PORT = "HINDSIGHT_API_PORT"
ENV_LOG_LEVEL = "HINDSIGHT_API_LOG_LEVEL"
ENV_LOG_FORMAT = "HINDSIGHT_API_LOG_FORMAT"
ENV_WORKERS = "HINDSIGHT_API_WORKERS"
ENV_MCP_ENABLED = "HINDSIGHT_API_MCP_ENABLED"
ENV_GRAPH_RETRIEVER = "HINDSIGHT_API_GRAPH_RETRIEVER"
ENV_MPFP_TOP_K_NEIGHBORS = "HINDSIGHT_API_MPFP_TOP_K_NEIGHBORS"
ENV_RECALL_MAX_CONCURRENT = "HINDSIGHT_API_RECALL_MAX_CONCURRENT"
ENV_RECALL_CONNECTION_BUDGET = "HINDSIGHT_API_RECALL_CONNECTION_BUDGET"
ENV_MCP_LOCAL_BANK_ID = "HINDSIGHT_API_MCP_LOCAL_BANK_ID"
ENV_MCP_INSTRUCTIONS = "HINDSIGHT_API_MCP_INSTRUCTIONS"
ENV_MENTAL_MODEL_REFRESH_CONCURRENCY = "HINDSIGHT_API_MENTAL_MODEL_REFRESH_CONCURRENCY"

# Observation settings (consolidated knowledge from facts)
ENV_OBSERVATION_MIN_FACTS = "HINDSIGHT_API_OBSERVATION_MIN_FACTS"
ENV_OBSERVATION_TOP_ENTITIES = "HINDSIGHT_API_OBSERVATION_TOP_ENTITIES"

# Retain settings
ENV_RETAIN_MAX_COMPLETION_TOKENS = "HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS"
ENV_RETAIN_CHUNK_SIZE = "HINDSIGHT_API_RETAIN_CHUNK_SIZE"
ENV_RETAIN_EXTRACT_CAUSAL_LINKS = "HINDSIGHT_API_RETAIN_EXTRACT_CAUSAL_LINKS"
ENV_RETAIN_EXTRACTION_MODE = "HINDSIGHT_API_RETAIN_EXTRACTION_MODE"
ENV_RETAIN_OBSERVATIONS_ASYNC = "HINDSIGHT_API_RETAIN_OBSERVATIONS_ASYNC"

# Observations settings (consolidated knowledge from facts)
ENV_ENABLE_OBSERVATIONS = "HINDSIGHT_API_ENABLE_OBSERVATIONS"
ENV_CONSOLIDATION_SIMILARITY_THRESHOLD = "HINDSIGHT_API_CONSOLIDATION_SIMILARITY_THRESHOLD"
ENV_CONSOLIDATION_BATCH_SIZE = "HINDSIGHT_API_CONSOLIDATION_BATCH_SIZE"

# Optimization flags
ENV_SKIP_LLM_VERIFICATION = "HINDSIGHT_API_SKIP_LLM_VERIFICATION"
ENV_LAZY_RERANKER = "HINDSIGHT_API_LAZY_RERANKER"

# Database migrations
ENV_RUN_MIGRATIONS_ON_STARTUP = "HINDSIGHT_API_RUN_MIGRATIONS_ON_STARTUP"

# Database connection pool
ENV_DB_POOL_MIN_SIZE = "HINDSIGHT_API_DB_POOL_MIN_SIZE"
ENV_DB_POOL_MAX_SIZE = "HINDSIGHT_API_DB_POOL_MAX_SIZE"
ENV_DB_COMMAND_TIMEOUT = "HINDSIGHT_API_DB_COMMAND_TIMEOUT"
ENV_DB_ACQUIRE_TIMEOUT = "HINDSIGHT_API_DB_ACQUIRE_TIMEOUT"

# Worker configuration (distributed task processing)
ENV_WORKER_ENABLED = "HINDSIGHT_API_WORKER_ENABLED"
ENV_WORKER_ID = "HINDSIGHT_API_WORKER_ID"
ENV_WORKER_POLL_INTERVAL_MS = "HINDSIGHT_API_WORKER_POLL_INTERVAL_MS"
ENV_WORKER_MAX_RETRIES = "HINDSIGHT_API_WORKER_MAX_RETRIES"
ENV_WORKER_BATCH_SIZE = "HINDSIGHT_API_WORKER_BATCH_SIZE"
ENV_WORKER_HTTP_PORT = "HINDSIGHT_API_WORKER_HTTP_PORT"

# Reflect agent settings
ENV_REFLECT_MAX_ITERATIONS = "HINDSIGHT_API_REFLECT_MAX_ITERATIONS"

# Default values
DEFAULT_DATABASE_URL = "pg0"
DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_LLM_MODEL = "gpt-5-mini"
DEFAULT_LLM_MAX_CONCURRENT = 32
DEFAULT_LLM_TIMEOUT = 120.0  # seconds

DEFAULT_EMBEDDINGS_PROVIDER = "local"
DEFAULT_EMBEDDINGS_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDINGS_OPENAI_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSION = 384

DEFAULT_RERANKER_PROVIDER = "local"
DEFAULT_RERANKER_LOCAL_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANKER_LOCAL_MAX_CONCURRENT = 4  # Limit concurrent CPU-bound reranking to prevent thrashing
DEFAULT_RERANKER_TEI_BATCH_SIZE = 128
DEFAULT_RERANKER_TEI_MAX_CONCURRENT = 8
DEFAULT_RERANKER_MAX_CANDIDATES = 300
DEFAULT_RERANKER_FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"  # Best balance of speed and quality
DEFAULT_RERANKER_FLASHRANK_CACHE_DIR = None  # Use default cache directory

DEFAULT_EMBEDDINGS_COHERE_MODEL = "embed-english-v3.0"
DEFAULT_RERANKER_COHERE_MODEL = "rerank-english-v3.0"

# LiteLLM defaults
DEFAULT_LITELLM_API_BASE = "http://localhost:4000"
DEFAULT_EMBEDDINGS_LITELLM_MODEL = "text-embedding-3-small"
DEFAULT_RERANKER_LITELLM_MODEL = "cohere/rerank-english-v3.0"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8888
DEFAULT_LOG_LEVEL = "info"
DEFAULT_LOG_FORMAT = "text"  # Options: "text", "json"
DEFAULT_WORKERS = 1
DEFAULT_MCP_ENABLED = True
DEFAULT_GRAPH_RETRIEVER = "link_expansion"  # Options: "link_expansion", "mpfp", "bfs"
DEFAULT_MPFP_TOP_K_NEIGHBORS = 20  # Fan-out limit per node in MPFP graph traversal
DEFAULT_RECALL_MAX_CONCURRENT = 32  # Max concurrent recall operations per worker
DEFAULT_RECALL_CONNECTION_BUDGET = 4  # Max concurrent DB connections per recall operation
DEFAULT_MCP_LOCAL_BANK_ID = "mcp"
DEFAULT_MENTAL_MODEL_REFRESH_CONCURRENCY = 8  # Max concurrent mental model refreshes

# Observation thresholds
DEFAULT_OBSERVATION_MIN_FACTS = 5  # Min facts required to generate entity observations
DEFAULT_OBSERVATION_TOP_ENTITIES = 5  # Max entities to process per retain batch

# Retain settings
DEFAULT_RETAIN_MAX_COMPLETION_TOKENS = 64000  # Max tokens for fact extraction LLM call
DEFAULT_RETAIN_CHUNK_SIZE = 3000  # Max chars per chunk for fact extraction
DEFAULT_RETAIN_EXTRACT_CAUSAL_LINKS = True  # Extract causal links between facts
DEFAULT_RETAIN_EXTRACTION_MODE = "concise"  # Extraction mode: "concise" or "verbose"
RETAIN_EXTRACTION_MODES = ("concise", "verbose")  # Allowed extraction modes
DEFAULT_RETAIN_OBSERVATIONS_ASYNC = False  # Run observation generation async (after retain completes)

# Observations defaults (consolidated knowledge from facts)
DEFAULT_ENABLE_OBSERVATIONS = False  # Observations disabled by default (experimental)
DEFAULT_CONSOLIDATION_SIMILARITY_THRESHOLD = 0.75  # Minimum similarity to consider a learning related
DEFAULT_CONSOLIDATION_BATCH_SIZE = 50  # Memories to load per batch (internal memory optimization)

# Database migrations
DEFAULT_RUN_MIGRATIONS_ON_STARTUP = True

# Database connection pool
DEFAULT_DB_POOL_MIN_SIZE = 5
DEFAULT_DB_POOL_MAX_SIZE = 100
DEFAULT_DB_COMMAND_TIMEOUT = 60  # seconds
DEFAULT_DB_ACQUIRE_TIMEOUT = 30  # seconds

# Worker configuration (distributed task processing)
DEFAULT_WORKER_ENABLED = True  # API runs worker by default (standalone mode)
DEFAULT_WORKER_ID = None  # Will use hostname if not specified
DEFAULT_WORKER_POLL_INTERVAL_MS = 500  # Poll database every 500ms
DEFAULT_WORKER_MAX_RETRIES = 3  # Max retries before marking task failed
DEFAULT_WORKER_BATCH_SIZE = 10  # Tasks to claim per poll cycle
DEFAULT_WORKER_HTTP_PORT = 8889  # HTTP port for worker metrics/health

# Reflect agent settings
DEFAULT_REFLECT_MAX_ITERATIONS = 10  # Max tool call iterations before forcing response

# Default MCP tool descriptions (can be customized via env vars)
DEFAULT_MCP_RETAIN_DESCRIPTION = """Store important information to long-term memory.

Use this tool PROACTIVELY whenever the user shares:
- Personal facts, preferences, or interests
- Important events or milestones
- User history, experiences, or background
- Decisions, opinions, or stated preferences
- Goals, plans, or future intentions
- Relationships or people mentioned
- Work context, projects, or responsibilities"""

DEFAULT_MCP_RECALL_DESCRIPTION = """Search memories to provide personalized, context-aware responses.

Use this tool PROACTIVELY to:
- Check user's preferences before making suggestions
- Recall user's history to provide continuity
- Remember user's goals and context
- Personalize responses based on past interactions"""

# Default embedding dimension (used by initial migration, adjusted at runtime)
EMBEDDING_DIMENSION = DEFAULT_EMBEDDING_DIMENSION


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging.

    Outputs logs in JSON format with a 'severity' field that cloud logging
    systems (GCP, AWS CloudWatch, etc.) can parse to correctly categorize log levels.
    """

    SEVERITY_MAP = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRITICAL",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "severity": self.SEVERITY_MAP.get(record.levelno, "DEFAULT"),
            "message": record.getMessage(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "logger": record.name,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def _validate_extraction_mode(mode: str) -> str:
    """Validate and normalize extraction mode."""
    mode_lower = mode.lower()
    if mode_lower not in RETAIN_EXTRACTION_MODES:
        logger.warning(
            f"Invalid extraction mode '{mode}', must be one of {RETAIN_EXTRACTION_MODES}. "
            f"Defaulting to '{DEFAULT_RETAIN_EXTRACTION_MODE}'."
        )
        return DEFAULT_RETAIN_EXTRACTION_MODE
    return mode_lower


@dataclass
class HindsightConfig:
    """Configuration container for Hindsight API."""

    # Database
    database_url: str

    # LLM (default, used as fallback for per-operation config)
    llm_provider: str
    llm_api_key: str | None
    llm_model: str
    llm_base_url: str | None
    llm_max_concurrent: int
    llm_timeout: float

    # Per-operation LLM configuration (None = use default LLM config)
    retain_llm_provider: str | None
    retain_llm_api_key: str | None
    retain_llm_model: str | None
    retain_llm_base_url: str | None

    reflect_llm_provider: str | None
    reflect_llm_api_key: str | None
    reflect_llm_model: str | None
    reflect_llm_base_url: str | None

    consolidation_llm_provider: str | None
    consolidation_llm_api_key: str | None
    consolidation_llm_model: str | None
    consolidation_llm_base_url: str | None

    # Embeddings
    embeddings_provider: str
    embeddings_local_model: str
    embeddings_tei_url: str | None
    embeddings_openai_base_url: str | None
    embeddings_cohere_base_url: str | None

    # Reranker
    reranker_provider: str
    reranker_local_model: str
    reranker_tei_url: str | None
    reranker_tei_batch_size: int
    reranker_tei_max_concurrent: int
    reranker_max_candidates: int
    reranker_cohere_base_url: str | None

    # Server
    host: str
    port: int
    log_level: str
    log_format: str
    mcp_enabled: bool

    # Recall
    graph_retriever: str
    mpfp_top_k_neighbors: int
    recall_max_concurrent: int
    recall_connection_budget: int
    mental_model_refresh_concurrency: int

    # Observation thresholds
    observation_min_facts: int
    observation_top_entities: int

    # Retain settings
    retain_max_completion_tokens: int
    retain_chunk_size: int
    retain_extract_causal_links: bool
    retain_extraction_mode: str
    retain_observations_async: bool

    # Observations settings (consolidated knowledge from facts)
    enable_observations: bool
    consolidation_similarity_threshold: float
    consolidation_batch_size: int

    # Optimization flags
    skip_llm_verification: bool
    lazy_reranker: bool

    # Database migrations
    run_migrations_on_startup: bool

    # Database connection pool
    db_pool_min_size: int
    db_pool_max_size: int
    db_command_timeout: int
    db_acquire_timeout: int

    # Worker configuration (distributed task processing)
    worker_enabled: bool
    worker_id: str | None
    worker_poll_interval_ms: int
    worker_max_retries: int
    worker_batch_size: int
    worker_http_port: int

    # Reflect agent settings
    reflect_max_iterations: int

    @classmethod
    def from_env(cls) -> "HindsightConfig":
        """Create configuration from environment variables."""
        return cls(
            # Database
            database_url=os.getenv(ENV_DATABASE_URL, DEFAULT_DATABASE_URL),
            # LLM
            llm_provider=os.getenv(ENV_LLM_PROVIDER, DEFAULT_LLM_PROVIDER),
            llm_api_key=os.getenv(ENV_LLM_API_KEY),
            llm_model=os.getenv(ENV_LLM_MODEL, DEFAULT_LLM_MODEL),
            llm_base_url=os.getenv(ENV_LLM_BASE_URL) or None,
            llm_max_concurrent=int(os.getenv(ENV_LLM_MAX_CONCURRENT, str(DEFAULT_LLM_MAX_CONCURRENT))),
            llm_timeout=float(os.getenv(ENV_LLM_TIMEOUT, str(DEFAULT_LLM_TIMEOUT))),
            # Per-operation LLM config (None = use default)
            retain_llm_provider=os.getenv(ENV_RETAIN_LLM_PROVIDER) or None,
            retain_llm_api_key=os.getenv(ENV_RETAIN_LLM_API_KEY) or None,
            retain_llm_model=os.getenv(ENV_RETAIN_LLM_MODEL) or None,
            retain_llm_base_url=os.getenv(ENV_RETAIN_LLM_BASE_URL) or None,
            reflect_llm_provider=os.getenv(ENV_REFLECT_LLM_PROVIDER) or None,
            reflect_llm_api_key=os.getenv(ENV_REFLECT_LLM_API_KEY) or None,
            reflect_llm_model=os.getenv(ENV_REFLECT_LLM_MODEL) or None,
            reflect_llm_base_url=os.getenv(ENV_REFLECT_LLM_BASE_URL) or None,
            consolidation_llm_provider=os.getenv(ENV_CONSOLIDATION_LLM_PROVIDER) or None,
            consolidation_llm_api_key=os.getenv(ENV_CONSOLIDATION_LLM_API_KEY) or None,
            consolidation_llm_model=os.getenv(ENV_CONSOLIDATION_LLM_MODEL) or None,
            consolidation_llm_base_url=os.getenv(ENV_CONSOLIDATION_LLM_BASE_URL) or None,
            # Embeddings
            embeddings_provider=os.getenv(ENV_EMBEDDINGS_PROVIDER, DEFAULT_EMBEDDINGS_PROVIDER),
            embeddings_local_model=os.getenv(ENV_EMBEDDINGS_LOCAL_MODEL, DEFAULT_EMBEDDINGS_LOCAL_MODEL),
            embeddings_tei_url=os.getenv(ENV_EMBEDDINGS_TEI_URL),
            embeddings_openai_base_url=os.getenv(ENV_EMBEDDINGS_OPENAI_BASE_URL) or None,
            embeddings_cohere_base_url=os.getenv(ENV_EMBEDDINGS_COHERE_BASE_URL) or None,
            # Reranker
            reranker_provider=os.getenv(ENV_RERANKER_PROVIDER, DEFAULT_RERANKER_PROVIDER),
            reranker_local_model=os.getenv(ENV_RERANKER_LOCAL_MODEL, DEFAULT_RERANKER_LOCAL_MODEL),
            reranker_tei_url=os.getenv(ENV_RERANKER_TEI_URL),
            reranker_tei_batch_size=int(os.getenv(ENV_RERANKER_TEI_BATCH_SIZE, str(DEFAULT_RERANKER_TEI_BATCH_SIZE))),
            reranker_tei_max_concurrent=int(
                os.getenv(ENV_RERANKER_TEI_MAX_CONCURRENT, str(DEFAULT_RERANKER_TEI_MAX_CONCURRENT))
            ),
            reranker_max_candidates=int(os.getenv(ENV_RERANKER_MAX_CANDIDATES, str(DEFAULT_RERANKER_MAX_CANDIDATES))),
            reranker_cohere_base_url=os.getenv(ENV_RERANKER_COHERE_BASE_URL) or None,
            # Server
            host=os.getenv(ENV_HOST, DEFAULT_HOST),
            port=int(os.getenv(ENV_PORT, DEFAULT_PORT)),
            log_level=os.getenv(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL),
            log_format=os.getenv(ENV_LOG_FORMAT, DEFAULT_LOG_FORMAT).lower(),
            mcp_enabled=os.getenv(ENV_MCP_ENABLED, str(DEFAULT_MCP_ENABLED)).lower() == "true",
            # Recall
            graph_retriever=os.getenv(ENV_GRAPH_RETRIEVER, DEFAULT_GRAPH_RETRIEVER),
            mpfp_top_k_neighbors=int(os.getenv(ENV_MPFP_TOP_K_NEIGHBORS, str(DEFAULT_MPFP_TOP_K_NEIGHBORS))),
            recall_max_concurrent=int(os.getenv(ENV_RECALL_MAX_CONCURRENT, str(DEFAULT_RECALL_MAX_CONCURRENT))),
            recall_connection_budget=int(
                os.getenv(ENV_RECALL_CONNECTION_BUDGET, str(DEFAULT_RECALL_CONNECTION_BUDGET))
            ),
            mental_model_refresh_concurrency=int(
                os.getenv(ENV_MENTAL_MODEL_REFRESH_CONCURRENCY, str(DEFAULT_MENTAL_MODEL_REFRESH_CONCURRENCY))
            ),
            # Optimization flags
            skip_llm_verification=os.getenv(ENV_SKIP_LLM_VERIFICATION, "false").lower() == "true",
            lazy_reranker=os.getenv(ENV_LAZY_RERANKER, "false").lower() == "true",
            # Observation thresholds
            observation_min_facts=int(os.getenv(ENV_OBSERVATION_MIN_FACTS, str(DEFAULT_OBSERVATION_MIN_FACTS))),
            observation_top_entities=int(
                os.getenv(ENV_OBSERVATION_TOP_ENTITIES, str(DEFAULT_OBSERVATION_TOP_ENTITIES))
            ),
            # Retain settings
            retain_max_completion_tokens=int(
                os.getenv(ENV_RETAIN_MAX_COMPLETION_TOKENS, str(DEFAULT_RETAIN_MAX_COMPLETION_TOKENS))
            ),
            retain_chunk_size=int(os.getenv(ENV_RETAIN_CHUNK_SIZE, str(DEFAULT_RETAIN_CHUNK_SIZE))),
            retain_extract_causal_links=os.getenv(
                ENV_RETAIN_EXTRACT_CAUSAL_LINKS, str(DEFAULT_RETAIN_EXTRACT_CAUSAL_LINKS)
            ).lower()
            == "true",
            retain_extraction_mode=_validate_extraction_mode(
                os.getenv(ENV_RETAIN_EXTRACTION_MODE, DEFAULT_RETAIN_EXTRACTION_MODE)
            ),
            retain_observations_async=os.getenv(
                ENV_RETAIN_OBSERVATIONS_ASYNC, str(DEFAULT_RETAIN_OBSERVATIONS_ASYNC)
            ).lower()
            == "true",
            # Observations settings (consolidated knowledge from facts)
            enable_observations=os.getenv(ENV_ENABLE_OBSERVATIONS, str(DEFAULT_ENABLE_OBSERVATIONS)).lower() == "true",
            consolidation_similarity_threshold=float(
                os.getenv(ENV_CONSOLIDATION_SIMILARITY_THRESHOLD, str(DEFAULT_CONSOLIDATION_SIMILARITY_THRESHOLD))
            ),
            consolidation_batch_size=int(
                os.getenv(ENV_CONSOLIDATION_BATCH_SIZE, str(DEFAULT_CONSOLIDATION_BATCH_SIZE))
            ),
            # Database migrations
            run_migrations_on_startup=os.getenv(ENV_RUN_MIGRATIONS_ON_STARTUP, "true").lower() == "true",
            # Database connection pool
            db_pool_min_size=int(os.getenv(ENV_DB_POOL_MIN_SIZE, str(DEFAULT_DB_POOL_MIN_SIZE))),
            db_pool_max_size=int(os.getenv(ENV_DB_POOL_MAX_SIZE, str(DEFAULT_DB_POOL_MAX_SIZE))),
            db_command_timeout=int(os.getenv(ENV_DB_COMMAND_TIMEOUT, str(DEFAULT_DB_COMMAND_TIMEOUT))),
            db_acquire_timeout=int(os.getenv(ENV_DB_ACQUIRE_TIMEOUT, str(DEFAULT_DB_ACQUIRE_TIMEOUT))),
            # Worker configuration
            worker_enabled=os.getenv(ENV_WORKER_ENABLED, str(DEFAULT_WORKER_ENABLED)).lower() == "true",
            worker_id=os.getenv(ENV_WORKER_ID) or DEFAULT_WORKER_ID,
            worker_poll_interval_ms=int(os.getenv(ENV_WORKER_POLL_INTERVAL_MS, str(DEFAULT_WORKER_POLL_INTERVAL_MS))),
            worker_max_retries=int(os.getenv(ENV_WORKER_MAX_RETRIES, str(DEFAULT_WORKER_MAX_RETRIES))),
            worker_batch_size=int(os.getenv(ENV_WORKER_BATCH_SIZE, str(DEFAULT_WORKER_BATCH_SIZE))),
            worker_http_port=int(os.getenv(ENV_WORKER_HTTP_PORT, str(DEFAULT_WORKER_HTTP_PORT))),
            # Reflect agent settings
            reflect_max_iterations=int(os.getenv(ENV_REFLECT_MAX_ITERATIONS, str(DEFAULT_REFLECT_MAX_ITERATIONS))),
        )

    def get_llm_base_url(self) -> str:
        """Get the LLM base URL, with provider-specific defaults."""
        if self.llm_base_url:
            return self.llm_base_url

        provider = self.llm_provider.lower()
        if provider == "groq":
            return "https://api.groq.com/openai/v1"
        elif provider == "ollama":
            return "http://localhost:11434/v1"
        elif provider == "lmstudio":
            return "http://localhost:1234/v1"
        else:
            return ""

    def get_python_log_level(self) -> int:
        """Get the Python logging level from the configured log level string."""
        log_level_map = {
            "critical": logging.CRITICAL,
            "error": logging.ERROR,
            "warning": logging.WARNING,
            "info": logging.INFO,
            "debug": logging.DEBUG,
            "trace": logging.DEBUG,  # Python doesn't have TRACE, use DEBUG
        }
        return log_level_map.get(self.log_level.lower(), logging.INFO)

    def configure_logging(self) -> None:
        """Configure Python logging based on the log level and format.

        When log_format is "json", outputs structured JSON logs with a severity
        field that GCP Cloud Logging can parse for proper log level categorization.
        """
        root_logger = logging.getLogger()
        root_logger.setLevel(self.get_python_log_level())

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create handler writing to stdout (GCP treats stderr as ERROR)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(self.get_python_log_level())

        if self.log_format == "json":
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))

        root_logger.addHandler(handler)

    def log_config(self) -> None:
        """Log the current configuration (without sensitive values)."""
        logger.info(f"Database: {self.database_url}")
        logger.info(f"LLM: provider={self.llm_provider}, model={self.llm_model}")
        if self.retain_llm_provider or self.retain_llm_model:
            retain_provider = self.retain_llm_provider or self.llm_provider
            retain_model = self.retain_llm_model or self.llm_model
            logger.info(f"LLM (retain): provider={retain_provider}, model={retain_model}")
        if self.reflect_llm_provider or self.reflect_llm_model:
            reflect_provider = self.reflect_llm_provider or self.llm_provider
            reflect_model = self.reflect_llm_model or self.llm_model
            logger.info(f"LLM (reflect): provider={reflect_provider}, model={reflect_model}")
        if self.consolidation_llm_provider or self.consolidation_llm_model:
            consolidation_provider = self.consolidation_llm_provider or self.llm_provider
            consolidation_model = self.consolidation_llm_model or self.llm_model
            logger.info(f"LLM (consolidation): provider={consolidation_provider}, model={consolidation_model}")
        logger.info(f"Embeddings: provider={self.embeddings_provider}")
        logger.info(f"Reranker: provider={self.reranker_provider}")
        logger.info(f"Graph retriever: {self.graph_retriever}")


# Cached config instance
_config_cache: HindsightConfig | None = None


def get_config() -> HindsightConfig:
    """Get the cached configuration, loading from environment on first call."""
    global _config_cache
    if _config_cache is None:
        _config_cache = HindsightConfig.from_env()
    return _config_cache


def clear_config_cache() -> None:
    """Clear the config cache. Useful for testing or reloading config."""
    global _config_cache
    _config_cache = None

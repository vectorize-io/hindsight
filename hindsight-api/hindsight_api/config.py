"""
Centralized configuration for Hindsight API.

All environment variables and their defaults are defined here.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from typing import Any

from dotenv import find_dotenv, load_dotenv

# Load .env file, searching current and parent directories (overrides existing env vars)
load_dotenv(find_dotenv(usecwd=True), override=True)

logger = logging.getLogger(__name__)


class ConfigFieldAccessError(AttributeError):
    """Raised when trying to access a bank-configurable field from global config."""

    pass


class StaticConfigProxy:
    """
    Proxy that wraps HindsightConfig and only allows access to static (non-configurable) fields.

    Raises ConfigFieldAccessError when trying to access configurable fields that vary per-bank.
    Forces developers to use get_resolved_config(bank_id, context) for bank-specific settings.
    """

    def __init__(self, config: "HindsightConfig"):
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_configurable_fields", HindsightConfig.get_configurable_fields())

    def __getattribute__(self, name: str):
        if name.startswith("_"):
            return object.__getattribute__(self, name)

        configurable_fields = object.__getattribute__(self, "_configurable_fields")
        if name in configurable_fields:
            raise ConfigFieldAccessError(
                f"Field '{name}' is bank-configurable and cannot be accessed from global config. "
                f"Use ConfigResolver.resolve_full_config(bank_id, context) to get bank-specific config. "
                f"This prevents accidentally using global defaults when bank-specific overrides exist."
            )

        config = object.__getattribute__(self, "_config")
        return getattr(config, name)

    def __setattr__(self, name: str, value):
        raise AttributeError("Config is read-only. Modifications must go through ConfigResolver.")


# Configuration field markers for hierarchical configuration
def hierarchical(default_value):
    """
    Mark a config field as hierarchical (can be overridden per-tenant/bank).

    Hierarchical fields can be customized at the tenant or bank level via database
    configuration. Examples: LLM settings, retention parameters, retrieval settings.
    """
    return field(default=default_value, metadata={"hierarchical": True})


def static(default_value):
    """
    Mark a config field as static (server-level only, cannot be overridden).

    Static fields are infrastructure-level settings that affect the entire server
    and cannot vary per tenant or bank. Examples: database URL, API port, worker settings.
    """
    return field(default=default_value, metadata={"hierarchical": False})


# Configuration key normalization utilities
def normalize_config_key(key: str) -> str:
    """
    Convert environment variable format to Python field name format.

    Examples:
        HINDSIGHT_API_LLM_PROVIDER -> llm_provider
        LLM_MODEL -> llm_model
        llm_model -> llm_model (already normalized)

    Args:
        key: Environment variable name or Python field name

    Returns:
        Normalized Python field name (lowercase snake_case)
    """
    if key.startswith("HINDSIGHT_API_"):
        key = key[len("HINDSIGHT_API_") :]
    return key.lower()


def normalize_config_dict(config: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize all keys in a config dict to Python field names.

    Allows users to provide config overrides in either format:
    - Python field format: {"llm_provider": "openai"}
    - Env var format: {"HINDSIGHT_API_LLM_PROVIDER": "openai"}

    Args:
        config: Dict with env var or Python field names as keys

    Returns:
        Dict with all keys normalized to Python field names
    """
    return {normalize_config_key(k): v for k, v in config.items()}


# Environment variable names
ENV_DATABASE_URL = "HINDSIGHT_API_DATABASE_URL"
ENV_DATABASE_SCHEMA = "HINDSIGHT_API_DATABASE_SCHEMA"
ENV_LLM_PROVIDER = "HINDSIGHT_API_LLM_PROVIDER"
ENV_LLM_API_KEY = "HINDSIGHT_API_LLM_API_KEY"
ENV_LLM_MODEL = "HINDSIGHT_API_LLM_MODEL"
ENV_LLM_BASE_URL = "HINDSIGHT_API_LLM_BASE_URL"
ENV_LLM_MAX_CONCURRENT = "HINDSIGHT_API_LLM_MAX_CONCURRENT"
ENV_LLM_MAX_RETRIES = "HINDSIGHT_API_LLM_MAX_RETRIES"
ENV_LLM_INITIAL_BACKOFF = "HINDSIGHT_API_LLM_INITIAL_BACKOFF"
ENV_LLM_MAX_BACKOFF = "HINDSIGHT_API_LLM_MAX_BACKOFF"
ENV_LLM_TIMEOUT = "HINDSIGHT_API_LLM_TIMEOUT"
ENV_LLM_GROQ_SERVICE_TIER = "HINDSIGHT_API_LLM_GROQ_SERVICE_TIER"
ENV_LLM_OPENAI_SERVICE_TIER = "HINDSIGHT_API_LLM_OPENAI_SERVICE_TIER"

# Defaults for service tiers
DEFAULT_LLM_GROQ_SERVICE_TIER = "auto"  # "on_demand", "flex", or "auto"
DEFAULT_LLM_OPENAI_SERVICE_TIER = None  # None (default) or "flex" (50% cheaper)

# Per-operation LLM configuration (optional, falls back to global LLM config)
ENV_RETAIN_LLM_PROVIDER = "HINDSIGHT_API_RETAIN_LLM_PROVIDER"
ENV_RETAIN_LLM_API_KEY = "HINDSIGHT_API_RETAIN_LLM_API_KEY"
ENV_RETAIN_LLM_MODEL = "HINDSIGHT_API_RETAIN_LLM_MODEL"
ENV_RETAIN_LLM_BASE_URL = "HINDSIGHT_API_RETAIN_LLM_BASE_URL"
ENV_RETAIN_LLM_MAX_CONCURRENT = "HINDSIGHT_API_RETAIN_LLM_MAX_CONCURRENT"
ENV_RETAIN_LLM_MAX_RETRIES = "HINDSIGHT_API_RETAIN_LLM_MAX_RETRIES"
ENV_RETAIN_LLM_INITIAL_BACKOFF = "HINDSIGHT_API_RETAIN_LLM_INITIAL_BACKOFF"
ENV_RETAIN_LLM_MAX_BACKOFF = "HINDSIGHT_API_RETAIN_LLM_MAX_BACKOFF"
ENV_RETAIN_LLM_TIMEOUT = "HINDSIGHT_API_RETAIN_LLM_TIMEOUT"

ENV_REFLECT_LLM_PROVIDER = "HINDSIGHT_API_REFLECT_LLM_PROVIDER"
ENV_REFLECT_LLM_API_KEY = "HINDSIGHT_API_REFLECT_LLM_API_KEY"
ENV_REFLECT_LLM_MODEL = "HINDSIGHT_API_REFLECT_LLM_MODEL"
ENV_REFLECT_LLM_BASE_URL = "HINDSIGHT_API_REFLECT_LLM_BASE_URL"
ENV_REFLECT_LLM_MAX_CONCURRENT = "HINDSIGHT_API_REFLECT_LLM_MAX_CONCURRENT"
ENV_REFLECT_LLM_MAX_RETRIES = "HINDSIGHT_API_REFLECT_LLM_MAX_RETRIES"
ENV_REFLECT_LLM_INITIAL_BACKOFF = "HINDSIGHT_API_REFLECT_LLM_INITIAL_BACKOFF"
ENV_REFLECT_LLM_MAX_BACKOFF = "HINDSIGHT_API_REFLECT_LLM_MAX_BACKOFF"
ENV_REFLECT_LLM_TIMEOUT = "HINDSIGHT_API_REFLECT_LLM_TIMEOUT"

ENV_CONSOLIDATION_LLM_PROVIDER = "HINDSIGHT_API_CONSOLIDATION_LLM_PROVIDER"
ENV_CONSOLIDATION_LLM_API_KEY = "HINDSIGHT_API_CONSOLIDATION_LLM_API_KEY"
ENV_CONSOLIDATION_LLM_MODEL = "HINDSIGHT_API_CONSOLIDATION_LLM_MODEL"
ENV_CONSOLIDATION_LLM_BASE_URL = "HINDSIGHT_API_CONSOLIDATION_LLM_BASE_URL"
ENV_CONSOLIDATION_LLM_MAX_CONCURRENT = "HINDSIGHT_API_CONSOLIDATION_LLM_MAX_CONCURRENT"
ENV_CONSOLIDATION_LLM_MAX_RETRIES = "HINDSIGHT_API_CONSOLIDATION_LLM_MAX_RETRIES"
ENV_CONSOLIDATION_LLM_INITIAL_BACKOFF = "HINDSIGHT_API_CONSOLIDATION_LLM_INITIAL_BACKOFF"
ENV_CONSOLIDATION_LLM_MAX_BACKOFF = "HINDSIGHT_API_CONSOLIDATION_LLM_MAX_BACKOFF"
ENV_CONSOLIDATION_LLM_TIMEOUT = "HINDSIGHT_API_CONSOLIDATION_LLM_TIMEOUT"

ENV_EMBEDDINGS_PROVIDER = "HINDSIGHT_API_EMBEDDINGS_PROVIDER"
ENV_EMBEDDINGS_LOCAL_MODEL = "HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL"
ENV_EMBEDDINGS_LOCAL_FORCE_CPU = "HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU"
ENV_EMBEDDINGS_LOCAL_TRUST_REMOTE_CODE = "HINDSIGHT_API_EMBEDDINGS_LOCAL_TRUST_REMOTE_CODE"
ENV_EMBEDDINGS_TEI_URL = "HINDSIGHT_API_EMBEDDINGS_TEI_URL"
ENV_EMBEDDINGS_OPENAI_API_KEY = "HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY"
ENV_EMBEDDINGS_OPENAI_MODEL = "HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL"
ENV_EMBEDDINGS_OPENAI_BASE_URL = "HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL"

# Cohere configuration (separate for embeddings and reranker)
ENV_EMBEDDINGS_COHERE_API_KEY = "HINDSIGHT_API_EMBEDDINGS_COHERE_API_KEY"
ENV_EMBEDDINGS_COHERE_MODEL = "HINDSIGHT_API_EMBEDDINGS_COHERE_MODEL"
ENV_EMBEDDINGS_COHERE_BASE_URL = "HINDSIGHT_API_EMBEDDINGS_COHERE_BASE_URL"
ENV_RERANKER_COHERE_API_KEY = "HINDSIGHT_API_RERANKER_COHERE_API_KEY"
ENV_RERANKER_COHERE_MODEL = "HINDSIGHT_API_RERANKER_COHERE_MODEL"
ENV_RERANKER_COHERE_BASE_URL = "HINDSIGHT_API_RERANKER_COHERE_BASE_URL"

# Deprecated: Legacy shared Cohere API key (for backward compatibility)
ENV_COHERE_API_KEY = "HINDSIGHT_API_COHERE_API_KEY"

# LiteLLM configuration (separate for embeddings and reranker)
ENV_EMBEDDINGS_LITELLM_API_BASE = "HINDSIGHT_API_EMBEDDINGS_LITELLM_API_BASE"
ENV_EMBEDDINGS_LITELLM_API_KEY = "HINDSIGHT_API_EMBEDDINGS_LITELLM_API_KEY"
ENV_EMBEDDINGS_LITELLM_MODEL = "HINDSIGHT_API_EMBEDDINGS_LITELLM_MODEL"
ENV_RERANKER_LITELLM_API_BASE = "HINDSIGHT_API_RERANKER_LITELLM_API_BASE"
ENV_RERANKER_LITELLM_API_KEY = "HINDSIGHT_API_RERANKER_LITELLM_API_KEY"
ENV_RERANKER_LITELLM_MODEL = "HINDSIGHT_API_RERANKER_LITELLM_MODEL"

# LiteLLM SDK configuration (direct API access, no proxy needed)
ENV_EMBEDDINGS_LITELLM_SDK_API_KEY = "HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_API_KEY"
ENV_EMBEDDINGS_LITELLM_SDK_MODEL = "HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_MODEL"
ENV_EMBEDDINGS_LITELLM_SDK_API_BASE = "HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_API_BASE"
ENV_RERANKER_LITELLM_SDK_API_KEY = "HINDSIGHT_API_RERANKER_LITELLM_SDK_API_KEY"
ENV_RERANKER_LITELLM_SDK_MODEL = "HINDSIGHT_API_RERANKER_LITELLM_SDK_MODEL"
ENV_RERANKER_LITELLM_SDK_API_BASE = "HINDSIGHT_API_RERANKER_LITELLM_SDK_API_BASE"

# Deprecated: Legacy shared LiteLLM config (for backward compatibility)
ENV_LITELLM_API_BASE = "HINDSIGHT_API_LITELLM_API_BASE"
ENV_LITELLM_API_KEY = "HINDSIGHT_API_LITELLM_API_KEY"

ENV_RERANKER_PROVIDER = "HINDSIGHT_API_RERANKER_PROVIDER"
ENV_RERANKER_LOCAL_MODEL = "HINDSIGHT_API_RERANKER_LOCAL_MODEL"
ENV_RERANKER_LOCAL_FORCE_CPU = "HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU"
ENV_RERANKER_LOCAL_MAX_CONCURRENT = "HINDSIGHT_API_RERANKER_LOCAL_MAX_CONCURRENT"
ENV_RERANKER_LOCAL_TRUST_REMOTE_CODE = "HINDSIGHT_API_RERANKER_LOCAL_TRUST_REMOTE_CODE"
ENV_RERANKER_TEI_URL = "HINDSIGHT_API_RERANKER_TEI_URL"
ENV_RERANKER_TEI_BATCH_SIZE = "HINDSIGHT_API_RERANKER_TEI_BATCH_SIZE"
ENV_RERANKER_TEI_MAX_CONCURRENT = "HINDSIGHT_API_RERANKER_TEI_MAX_CONCURRENT"
ENV_RERANKER_MAX_CANDIDATES = "HINDSIGHT_API_RERANKER_MAX_CANDIDATES"
ENV_RERANKER_FLASHRANK_MODEL = "HINDSIGHT_API_RERANKER_FLASHRANK_MODEL"
ENV_RERANKER_FLASHRANK_CACHE_DIR = "HINDSIGHT_API_RERANKER_FLASHRANK_CACHE_DIR"

ENV_VECTOR_EXTENSION = "HINDSIGHT_API_VECTOR_EXTENSION"
ENV_TEXT_SEARCH_EXTENSION = "HINDSIGHT_API_TEXT_SEARCH_EXTENSION"

ENV_HOST = "HINDSIGHT_API_HOST"
ENV_PORT = "HINDSIGHT_API_PORT"
ENV_BASE_PATH = "HINDSIGHT_API_BASE_PATH"
ENV_LOG_LEVEL = "HINDSIGHT_API_LOG_LEVEL"
ENV_LOG_FORMAT = "HINDSIGHT_API_LOG_FORMAT"
ENV_WORKERS = "HINDSIGHT_API_WORKERS"
ENV_MCP_ENABLED = "HINDSIGHT_API_MCP_ENABLED"
ENV_ENABLE_BANK_CONFIG_API = "HINDSIGHT_API_ENABLE_BANK_CONFIG_API"
ENV_GRAPH_RETRIEVER = "HINDSIGHT_API_GRAPH_RETRIEVER"
ENV_MPFP_TOP_K_NEIGHBORS = "HINDSIGHT_API_MPFP_TOP_K_NEIGHBORS"
ENV_RECALL_MAX_CONCURRENT = "HINDSIGHT_API_RECALL_MAX_CONCURRENT"
ENV_RECALL_CONNECTION_BUDGET = "HINDSIGHT_API_RECALL_CONNECTION_BUDGET"
ENV_MENTAL_MODEL_REFRESH_CONCURRENCY = "HINDSIGHT_API_MENTAL_MODEL_REFRESH_CONCURRENCY"

# OpenTelemetry tracing configuration
ENV_OTEL_TRACES_ENABLED = "HINDSIGHT_API_OTEL_TRACES_ENABLED"
ENV_OTEL_EXPORTER_OTLP_ENDPOINT = "HINDSIGHT_API_OTEL_EXPORTER_OTLP_ENDPOINT"
ENV_OTEL_EXPORTER_OTLP_HEADERS = "HINDSIGHT_API_OTEL_EXPORTER_OTLP_HEADERS"
ENV_OTEL_SERVICE_NAME = "HINDSIGHT_API_OTEL_SERVICE_NAME"
ENV_OTEL_DEPLOYMENT_ENVIRONMENT = "HINDSIGHT_API_OTEL_DEPLOYMENT_ENVIRONMENT"

# Vertex AI configuration
ENV_LLM_VERTEXAI_PROJECT_ID = "HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID"
ENV_LLM_VERTEXAI_REGION = "HINDSIGHT_API_LLM_VERTEXAI_REGION"
ENV_LLM_VERTEXAI_SERVICE_ACCOUNT_KEY = "HINDSIGHT_API_LLM_VERTEXAI_SERVICE_ACCOUNT_KEY"

# Retain settings
ENV_RETAIN_MAX_COMPLETION_TOKENS = "HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS"
ENV_RETAIN_CHUNK_SIZE = "HINDSIGHT_API_RETAIN_CHUNK_SIZE"
ENV_RETAIN_EXTRACT_CAUSAL_LINKS = "HINDSIGHT_API_RETAIN_EXTRACT_CAUSAL_LINKS"
ENV_RETAIN_EXTRACTION_MODE = "HINDSIGHT_API_RETAIN_EXTRACTION_MODE"
ENV_RETAIN_CUSTOM_INSTRUCTIONS = "HINDSIGHT_API_RETAIN_CUSTOM_INSTRUCTIONS"
ENV_RETAIN_BATCH_TOKENS = "HINDSIGHT_API_RETAIN_BATCH_TOKENS"
ENV_RETAIN_BATCH_ENABLED = "HINDSIGHT_API_RETAIN_BATCH_ENABLED"
ENV_RETAIN_BATCH_POLL_INTERVAL_SECONDS = "HINDSIGHT_API_RETAIN_BATCH_POLL_INTERVAL_SECONDS"

# File storage configuration
ENV_FILE_STORAGE_TYPE = "HINDSIGHT_API_FILE_STORAGE_TYPE"
ENV_FILE_STORAGE_S3_BUCKET = "HINDSIGHT_API_FILE_STORAGE_S3_BUCKET"
ENV_FILE_STORAGE_S3_REGION = "HINDSIGHT_API_FILE_STORAGE_S3_REGION"
ENV_FILE_STORAGE_S3_ENDPOINT = "HINDSIGHT_API_FILE_STORAGE_S3_ENDPOINT"
ENV_FILE_STORAGE_S3_ACCESS_KEY_ID = "HINDSIGHT_API_FILE_STORAGE_S3_ACCESS_KEY_ID"
ENV_FILE_STORAGE_S3_SECRET_ACCESS_KEY = "HINDSIGHT_API_FILE_STORAGE_S3_SECRET_ACCESS_KEY"
ENV_FILE_STORAGE_GCS_BUCKET = "HINDSIGHT_API_FILE_STORAGE_GCS_BUCKET"
ENV_FILE_STORAGE_GCS_SERVICE_ACCOUNT_KEY = "HINDSIGHT_API_FILE_STORAGE_GCS_SERVICE_ACCOUNT_KEY"
ENV_FILE_STORAGE_AZURE_CONTAINER = "HINDSIGHT_API_FILE_STORAGE_AZURE_CONTAINER"
ENV_FILE_STORAGE_AZURE_ACCOUNT_NAME = "HINDSIGHT_API_FILE_STORAGE_AZURE_ACCOUNT_NAME"
ENV_FILE_STORAGE_AZURE_ACCOUNT_KEY = "HINDSIGHT_API_FILE_STORAGE_AZURE_ACCOUNT_KEY"
ENV_FILE_PARSER = "HINDSIGHT_API_FILE_PARSER"
ENV_FILE_PARSER_IRIS_TOKEN = "HINDSIGHT_API_FILE_PARSER_IRIS_TOKEN"
ENV_FILE_PARSER_IRIS_ORG_ID = "HINDSIGHT_API_FILE_PARSER_IRIS_ORG_ID"
ENV_FILE_CONVERSION_MAX_BATCH_SIZE_MB = "HINDSIGHT_API_FILE_CONVERSION_MAX_BATCH_SIZE_MB"
ENV_FILE_CONVERSION_MAX_BATCH_SIZE = "HINDSIGHT_API_FILE_CONVERSION_MAX_BATCH_SIZE"
ENV_ENABLE_FILE_UPLOAD_API = "HINDSIGHT_API_ENABLE_FILE_UPLOAD_API"
ENV_FILE_DELETE_AFTER_RETAIN = "HINDSIGHT_API_FILE_DELETE_AFTER_RETAIN"

# Observations settings (consolidated knowledge from facts)
ENV_ENABLE_OBSERVATIONS = "HINDSIGHT_API_ENABLE_OBSERVATIONS"
ENV_CONSOLIDATION_BATCH_SIZE = "HINDSIGHT_API_CONSOLIDATION_BATCH_SIZE"
ENV_CONSOLIDATION_MAX_TOKENS = "HINDSIGHT_API_CONSOLIDATION_MAX_TOKENS"

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
ENV_WORKER_HTTP_PORT = "HINDSIGHT_API_WORKER_HTTP_PORT"
ENV_WORKER_MAX_SLOTS = "HINDSIGHT_API_WORKER_MAX_SLOTS"
ENV_WORKER_CONSOLIDATION_MAX_SLOTS = "HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS"

# Reflect agent settings
ENV_REFLECT_MAX_ITERATIONS = "HINDSIGHT_API_REFLECT_MAX_ITERATIONS"

# Default values
DEFAULT_DATABASE_URL = "pg0"
DEFAULT_DATABASE_SCHEMA = "public"
DEFAULT_LLM_PROVIDER = "openai"

# Provider-specific default models
PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "gemini": "gemini-2.5-flash",
    "groq": "openai/gpt-oss-120b",
    "ollama": "gemma3:12b",
    "lmstudio": "local-model",
    "vertexai": "google/gemini-2.5-flash-lite",
    "openai-codex": "gpt-5.2-codex",
    "claude-code": "claude-sonnet-4-5-20250929",
    "mock": "mock-model",
}
DEFAULT_LLM_MODEL = "gpt-4o-mini"  # Fallback if provider not in table
DEFAULT_LLM_MAX_CONCURRENT = 32
DEFAULT_LLM_MAX_RETRIES = 10  # Max retry attempts for LLM API calls
DEFAULT_LLM_INITIAL_BACKOFF = 1.0  # Initial backoff in seconds for retry exponential backoff
DEFAULT_LLM_MAX_BACKOFF = 60.0  # Max backoff cap in seconds for retry exponential backoff
DEFAULT_LLM_TIMEOUT = 120.0  # seconds

# Vertex AI defaults
DEFAULT_LLM_VERTEXAI_PROJECT_ID = None  # Required for Vertex AI
DEFAULT_LLM_VERTEXAI_REGION = "us-central1"
DEFAULT_LLM_VERTEXAI_SERVICE_ACCOUNT_KEY = None  # Optional, uses ADC if not set

DEFAULT_EMBEDDINGS_PROVIDER = "local"
DEFAULT_EMBEDDINGS_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDINGS_LOCAL_FORCE_CPU = False  # Force CPU mode for local embeddings (avoids MPS/XPC issues on macOS)
DEFAULT_EMBEDDINGS_LOCAL_TRUST_REMOTE_CODE = False  # Security: disabled by default, required for some models
DEFAULT_EMBEDDINGS_OPENAI_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSION = 384

DEFAULT_RERANKER_PROVIDER = "local"
DEFAULT_RERANKER_LOCAL_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANKER_LOCAL_FORCE_CPU = False  # Force CPU mode for local reranker (avoids MPS/XPC issues on macOS)
DEFAULT_RERANKER_LOCAL_MAX_CONCURRENT = 4  # Limit concurrent CPU-bound reranking to prevent thrashing
DEFAULT_RERANKER_LOCAL_TRUST_REMOTE_CODE = (
    False  # Security: disabled by default, required for some models like jina-reranker-v2
)
DEFAULT_RERANKER_TEI_BATCH_SIZE = 128
DEFAULT_RERANKER_TEI_MAX_CONCURRENT = 8
DEFAULT_RERANKER_MAX_CANDIDATES = 300
DEFAULT_RERANKER_FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"  # Best balance of speed and quality
DEFAULT_RERANKER_FLASHRANK_CACHE_DIR = None  # Use default cache directory

DEFAULT_EMBEDDINGS_COHERE_MODEL = "embed-english-v3.0"
DEFAULT_RERANKER_COHERE_MODEL = "rerank-english-v3.0"

# Vector extension (pgvector, vchord, or pgvectorscale)
DEFAULT_VECTOR_EXTENSION = "pgvector"  # Options: "pgvector", "vchord", "pgvectorscale"

# Text search extension (native PostgreSQL, vchord BM25, or Timescale pg_textsearch)
DEFAULT_TEXT_SEARCH_EXTENSION = "native"  # Options: "native", "vchord", "pg_textsearch"

# LiteLLM defaults
DEFAULT_LITELLM_API_BASE = "http://localhost:4000"
DEFAULT_EMBEDDINGS_LITELLM_MODEL = "text-embedding-3-small"
DEFAULT_RERANKER_LITELLM_MODEL = "cohere/rerank-english-v3.0"

# LiteLLM SDK defaults
DEFAULT_EMBEDDINGS_LITELLM_SDK_MODEL = "cohere/embed-english-v3.0"
DEFAULT_RERANKER_LITELLM_SDK_MODEL = "cohere/rerank-english-v3.0"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8888
DEFAULT_BASE_PATH = ""  # Empty string = root path
DEFAULT_LOG_LEVEL = "info"
DEFAULT_LOG_FORMAT = "text"  # Options: "text", "json"
DEFAULT_WORKERS = 1
DEFAULT_MCP_ENABLED = True
DEFAULT_ENABLE_BANK_CONFIG_API = False  # Disabled by default for security
DEFAULT_GRAPH_RETRIEVER = "link_expansion"  # Options: "link_expansion", "mpfp", "bfs"
DEFAULT_MPFP_TOP_K_NEIGHBORS = 20  # Fan-out limit per node in MPFP graph traversal
DEFAULT_RECALL_MAX_CONCURRENT = 32  # Max concurrent recall operations per worker
DEFAULT_RECALL_CONNECTION_BUDGET = 4  # Max concurrent DB connections per recall operation
DEFAULT_MENTAL_MODEL_REFRESH_CONCURRENCY = 8  # Max concurrent mental model refreshes

# Retain settings
DEFAULT_RETAIN_MAX_COMPLETION_TOKENS = 64000  # Max tokens for fact extraction LLM call
DEFAULT_RETAIN_CHUNK_SIZE = 3000  # Max chars per chunk for fact extraction
DEFAULT_RETAIN_EXTRACT_CAUSAL_LINKS = True  # Extract causal links between facts
DEFAULT_RETAIN_EXTRACTION_MODE = "concise"  # Extraction mode: "concise", "verbose", or "custom"
RETAIN_EXTRACTION_MODES = ("concise", "verbose", "custom")  # Allowed extraction modes
DEFAULT_RETAIN_CUSTOM_INSTRUCTIONS = None  # Custom extraction guidelines (only used when mode="custom")
DEFAULT_RETAIN_BATCH_TOKENS = 10_000  # ~40KB of text  # Max chars per sub-batch for async retain auto-splitting
DEFAULT_RETAIN_BATCH_ENABLED = False  # Use LLM Batch API for fact extraction (only when async=True)
DEFAULT_RETAIN_BATCH_POLL_INTERVAL_SECONDS = 60  # Batch API polling interval in seconds

# File storage defaults
DEFAULT_FILE_STORAGE_TYPE = "native"  # PostgreSQL BYTEA storage
DEFAULT_FILE_PARSER = "markitdown"  # File parser to use (markitdown is the only supported parser)
DEFAULT_FILE_CONVERSION_MAX_BATCH_SIZE_MB = 100  # Max total batch size in MB (all files combined)
DEFAULT_FILE_CONVERSION_MAX_BATCH_SIZE = 10  # Max files per batch upload
DEFAULT_ENABLE_FILE_UPLOAD_API = True  # Enable file upload endpoint
DEFAULT_FILE_DELETE_AFTER_RETAIN = True  # Delete file bytes after retain (saves storage)

# Observations defaults (consolidated knowledge from facts)
DEFAULT_ENABLE_OBSERVATIONS = True  # Observations enabled by default
DEFAULT_CONSOLIDATION_BATCH_SIZE = 50  # Memories to load per batch (internal memory optimization)
DEFAULT_CONSOLIDATION_MAX_TOKENS = 1024  # Max tokens for recall when finding related observations

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
DEFAULT_WORKER_HTTP_PORT = 8889  # HTTP port for worker metrics/health
DEFAULT_WORKER_MAX_SLOTS = 10  # Total concurrent tasks per worker
DEFAULT_WORKER_CONSOLIDATION_MAX_SLOTS = 2  # Max concurrent consolidation tasks per worker

# Reflect agent settings
DEFAULT_REFLECT_MAX_ITERATIONS = 10  # Max tool call iterations before forcing response

# OpenTelemetry tracing configuration
DEFAULT_OTEL_TRACES_ENABLED = False  # Disabled by default for backward compatibility
DEFAULT_OTEL_SERVICE_NAME = "hindsight-api"
DEFAULT_OTEL_DEPLOYMENT_ENVIRONMENT = "development"

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


def _get_default_model_for_provider(provider: str) -> str:
    """Get the default model for a given provider."""
    return PROVIDER_DEFAULT_MODELS.get(provider.lower(), DEFAULT_LLM_MODEL)


@dataclass
class HindsightConfig:
    """Configuration container for Hindsight API."""

    # Database
    database_url: str
    database_schema: str
    vector_extension: str  # "pgvector" or "vchord"
    text_search_extension: str  # "native" or "vchord"

    # LLM (default, used as fallback for per-operation config)
    llm_provider: str
    llm_api_key: str | None
    llm_model: str
    llm_base_url: str | None
    llm_max_concurrent: int
    llm_max_retries: int
    llm_initial_backoff: float
    llm_max_backoff: float
    llm_timeout: float
    llm_groq_service_tier: str  # Groq: "on_demand", "flex", or "auto"
    llm_openai_service_tier: str | None  # OpenAI: None (default) or "flex" (50% cheaper)

    # Vertex AI configuration
    llm_vertexai_project_id: str | None
    llm_vertexai_region: str
    llm_vertexai_service_account_key: str | None

    # Per-operation LLM configuration (None = use default LLM config)
    retain_llm_provider: str | None
    retain_llm_api_key: str | None
    retain_llm_model: str | None
    retain_llm_base_url: str | None
    retain_llm_max_concurrent: int | None
    retain_llm_max_retries: int | None
    retain_llm_initial_backoff: float | None
    retain_llm_max_backoff: float | None
    retain_llm_timeout: float | None

    reflect_llm_provider: str | None
    reflect_llm_api_key: str | None
    reflect_llm_model: str | None
    reflect_llm_base_url: str | None
    reflect_llm_max_concurrent: int | None
    reflect_llm_max_retries: int | None
    reflect_llm_initial_backoff: float | None
    reflect_llm_max_backoff: float | None
    reflect_llm_timeout: float | None

    consolidation_llm_provider: str | None
    consolidation_llm_api_key: str | None
    consolidation_llm_model: str | None
    consolidation_llm_base_url: str | None
    consolidation_llm_max_concurrent: int | None
    consolidation_llm_max_retries: int | None
    consolidation_llm_initial_backoff: float | None
    consolidation_llm_max_backoff: float | None
    consolidation_llm_timeout: float | None

    # Embeddings
    embeddings_provider: str
    embeddings_local_model: str
    embeddings_local_force_cpu: bool
    embeddings_local_trust_remote_code: bool
    embeddings_tei_url: str | None
    embeddings_openai_base_url: str | None
    embeddings_cohere_api_key: str | None
    embeddings_cohere_model: str
    embeddings_cohere_base_url: str | None
    embeddings_litellm_api_base: str
    embeddings_litellm_api_key: str | None
    embeddings_litellm_model: str
    embeddings_litellm_sdk_api_key: str | None
    embeddings_litellm_sdk_model: str
    embeddings_litellm_sdk_api_base: str | None

    # Reranker
    reranker_provider: str
    reranker_local_model: str
    reranker_local_force_cpu: bool
    reranker_local_max_concurrent: int
    reranker_local_trust_remote_code: bool
    reranker_tei_url: str | None
    reranker_tei_batch_size: int
    reranker_tei_max_concurrent: int
    reranker_max_candidates: int
    reranker_cohere_api_key: str | None
    reranker_cohere_model: str
    reranker_cohere_base_url: str | None
    reranker_litellm_api_base: str
    reranker_litellm_api_key: str | None
    reranker_litellm_model: str
    reranker_litellm_sdk_api_key: str | None
    reranker_litellm_sdk_model: str
    reranker_litellm_sdk_api_base: str | None

    # Server
    host: str
    port: int
    base_path: str
    log_level: str
    log_format: str
    mcp_enabled: bool
    enable_bank_config_api: bool

    # Recall
    graph_retriever: str
    mpfp_top_k_neighbors: int
    recall_max_concurrent: int
    recall_connection_budget: int
    mental_model_refresh_concurrency: int

    # Retain settings
    retain_max_completion_tokens: int
    retain_chunk_size: int
    retain_extract_causal_links: bool
    retain_extraction_mode: str
    retain_custom_instructions: str | None
    retain_batch_tokens: int
    retain_batch_enabled: bool
    retain_batch_poll_interval_seconds: int

    # File storage (static - server-level only)
    file_storage_type: str  # "native" (PostgreSQL) or "s3" (S3-compatible)
    file_storage_s3_bucket: str | None  # S3 bucket name (required for s3 storage)
    file_storage_s3_region: str | None  # S3 region (optional, uses SDK default)
    file_storage_s3_endpoint: str | None  # S3 endpoint URL (for MinIO, R2, etc.)
    file_storage_s3_access_key_id: str | None  # S3 access key (optional, uses env/IAM)
    file_storage_s3_secret_access_key: str | None  # S3 secret key (optional, uses env/IAM)
    file_storage_gcs_bucket: str | None  # GCS bucket name (required for gcs storage)
    file_storage_gcs_service_account_key: str | None  # GCS service account key JSON (optional, uses ADC)
    file_storage_azure_container: str | None  # Azure container name (required for azure storage)
    file_storage_azure_account_name: str | None  # Azure storage account name
    file_storage_azure_account_key: str | None  # Azure storage account key
    file_parser: str  # File parser to use (e.g., "markitdown", "iris")
    file_parser_iris_token: str | None  # Vectorize API token for iris parser (VECTORIZE_TOKEN)
    file_parser_iris_org_id: str | None  # Vectorize org ID for iris parser (VECTORIZE_ORG_ID)
    file_conversion_max_batch_size_mb: int  # Max total batch size in MB (all files combined)
    file_conversion_max_batch_size: int  # Max files per request
    enable_file_upload_api: bool
    file_delete_after_retain: bool

    # Observations settings (consolidated knowledge from facts)
    enable_observations: bool
    consolidation_batch_size: int
    consolidation_max_tokens: int

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
    worker_http_port: int
    worker_max_slots: int
    worker_consolidation_max_slots: int

    # Reflect agent settings
    reflect_max_iterations: int

    # OpenTelemetry tracing configuration
    otel_traces_enabled: bool
    otel_exporter_otlp_endpoint: str | None
    otel_exporter_otlp_headers: str | None
    otel_service_name: str
    otel_deployment_environment: str

    # Class-level sets for configuration categorization

    # CREDENTIAL_FIELDS: Never exposed via API, never configurable per-tenant/bank
    _CREDENTIAL_FIELDS = {
        # API Keys
        "llm_api_key",
        "retain_llm_api_key",
        "reflect_llm_api_key",
        "consolidation_llm_api_key",
        # Base URLs (could expose infrastructure)
        "llm_base_url",
        "retain_llm_base_url",
        "reflect_llm_base_url",
        "consolidation_llm_base_url",
        "embeddings_tei_base_url",
        "reranker_tei_base_url",
        "reranker_cohere_base_url",
        # Service Account Keys
        "llm_vertexai_service_account_key",
        # File storage credentials
        "file_storage_s3_access_key_id",
        "file_storage_s3_secret_access_key",
        "file_storage_gcs_service_account_key",
        "file_storage_azure_account_key",
        # File parser credentials
        "file_parser_iris_token",
    }

    # CONFIGURABLE_FIELDS: Safe behavioral settings that can be customized per-tenant/bank
    # These fields are manually tagged as safe to expose and modify.
    # Excludes credentials, infrastructure config, provider/model selection, and performance tuning.
    _CONFIGURABLE_FIELDS = {
        # Retention settings (behavioral)
        "retain_chunk_size",
        "retain_extraction_mode",
        "retain_custom_instructions",
        # Consolidation settings
        "enable_observations",
    }

    @property
    def file_conversion_max_batch_size_bytes(self) -> int:
        """Get maximum total batch size in bytes."""
        return self.file_conversion_max_batch_size_mb * 1024 * 1024

    @classmethod
    def get_configurable_fields(cls) -> set[str]:
        """
        Get set of field names that are configurable per-tenant/bank via API.

        Configurable fields are manually tagged behavioral settings that are safe
        to expose and modify (e.g., retain_chunk_size, custom_instructions).
        Excludes credentials, infrastructure config, and provider/model selection.

        Returns:
            Set of configurable field names
        """
        return cls._CONFIGURABLE_FIELDS.copy()

    @classmethod
    def get_credential_fields(cls) -> set[str]:
        """
        Get set of field names that are credentials (NEVER exposed via API).

        Credential fields include API keys, base URLs, and service account keys.
        These must never be returned in API responses or accepted in updates.

        Returns:
            Set of credential field names
        """
        return cls._CREDENTIAL_FIELDS.copy()

    @classmethod
    def get_hierarchical_fields(cls) -> set[str]:
        """
        DEPRECATED: Use get_configurable_fields() instead.

        Kept for backward compatibility during migration.
        """
        return cls.get_configurable_fields()

    @classmethod
    def get_static_fields(cls) -> set[str]:
        """
        Get set of field names that are static (server-level only).

        Static fields are infrastructure-level settings that cannot vary
        per tenant or bank. These include database config, API port, worker settings, etc.
        Also includes credential fields which are never configurable.

        Returns:
            Set of static field names
        """
        # Get all field names from dataclass
        all_fields = {f.name for f in fields(cls)}
        # Static fields = all fields - configurable fields
        return all_fields - cls._CONFIGURABLE_FIELDS

    def validate(self) -> None:
        """Validate configuration values and raise errors for invalid combinations."""
        # Validate vector_extension
        valid_extensions = ("pgvector", "vchord", "pgvectorscale")
        if self.vector_extension not in valid_extensions:
            raise ValueError(
                f"Invalid vector_extension: {self.vector_extension}. Must be one of: {', '.join(valid_extensions)}"
            )

        # Validate text_search_extension
        valid_text_search = ("native", "vchord", "pg_textsearch")
        if self.text_search_extension not in valid_text_search:
            raise ValueError(
                f"Invalid text_search_extension: {self.text_search_extension}. Must be one of: {', '.join(valid_text_search)}"
            )

        # RETAIN_MAX_COMPLETION_TOKENS must be greater than RETAIN_CHUNK_SIZE
        # to ensure the LLM has enough output capacity to extract facts from chunks
        if self.retain_max_completion_tokens <= self.retain_chunk_size:
            raise ValueError(
                f"Invalid configuration: HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS "
                f"({self.retain_max_completion_tokens}) must be greater than "
                f"HINDSIGHT_API_RETAIN_CHUNK_SIZE ({self.retain_chunk_size}). "
                f"\n\nYou have two options to fix this:"
                f"\n  1. Increase HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS to a value > {self.retain_chunk_size}"
                f"\n  2. Use a model that supports at least {self.retain_max_completion_tokens} output tokens"
                f"\n     (current model: {self.retain_llm_model or self.llm_model}, "
                f"provider: {self.retain_llm_provider or self.llm_provider})"
            )

    @classmethod
    def from_env(cls) -> "HindsightConfig":
        """Create configuration from environment variables."""
        # Get provider first to determine default model
        llm_provider = os.getenv(ENV_LLM_PROVIDER, DEFAULT_LLM_PROVIDER)
        llm_model = os.getenv(ENV_LLM_MODEL) or _get_default_model_for_provider(llm_provider)

        config = cls(
            # Database
            database_url=os.getenv(ENV_DATABASE_URL, DEFAULT_DATABASE_URL),
            database_schema=os.getenv(ENV_DATABASE_SCHEMA, DEFAULT_DATABASE_SCHEMA),
            vector_extension=os.getenv(ENV_VECTOR_EXTENSION, DEFAULT_VECTOR_EXTENSION).lower(),
            text_search_extension=os.getenv(ENV_TEXT_SEARCH_EXTENSION, DEFAULT_TEXT_SEARCH_EXTENSION).lower(),
            # LLM
            llm_provider=llm_provider,
            llm_api_key=os.getenv(ENV_LLM_API_KEY),
            llm_model=llm_model,
            llm_base_url=os.getenv(ENV_LLM_BASE_URL) or None,
            llm_max_concurrent=int(os.getenv(ENV_LLM_MAX_CONCURRENT, str(DEFAULT_LLM_MAX_CONCURRENT))),
            llm_max_retries=int(os.getenv(ENV_LLM_MAX_RETRIES, str(DEFAULT_LLM_MAX_RETRIES))),
            llm_initial_backoff=float(os.getenv(ENV_LLM_INITIAL_BACKOFF, str(DEFAULT_LLM_INITIAL_BACKOFF))),
            llm_max_backoff=float(os.getenv(ENV_LLM_MAX_BACKOFF, str(DEFAULT_LLM_MAX_BACKOFF))),
            llm_timeout=float(os.getenv(ENV_LLM_TIMEOUT, str(DEFAULT_LLM_TIMEOUT))),
            llm_groq_service_tier=os.getenv(ENV_LLM_GROQ_SERVICE_TIER, DEFAULT_LLM_GROQ_SERVICE_TIER),
            llm_openai_service_tier=os.getenv(ENV_LLM_OPENAI_SERVICE_TIER, DEFAULT_LLM_OPENAI_SERVICE_TIER),
            # Vertex AI
            llm_vertexai_project_id=os.getenv(ENV_LLM_VERTEXAI_PROJECT_ID) or DEFAULT_LLM_VERTEXAI_PROJECT_ID,
            llm_vertexai_region=os.getenv(ENV_LLM_VERTEXAI_REGION, DEFAULT_LLM_VERTEXAI_REGION),
            llm_vertexai_service_account_key=os.getenv(ENV_LLM_VERTEXAI_SERVICE_ACCOUNT_KEY)
            or DEFAULT_LLM_VERTEXAI_SERVICE_ACCOUNT_KEY,
            # Per-operation LLM config (None = use default)
            retain_llm_provider=os.getenv(ENV_RETAIN_LLM_PROVIDER) or None,
            retain_llm_api_key=os.getenv(ENV_RETAIN_LLM_API_KEY) or None,
            retain_llm_model=os.getenv(ENV_RETAIN_LLM_MODEL)
            or (
                _get_default_model_for_provider(os.getenv(ENV_RETAIN_LLM_PROVIDER))
                if os.getenv(ENV_RETAIN_LLM_PROVIDER)
                else None
            ),
            retain_llm_base_url=os.getenv(ENV_RETAIN_LLM_BASE_URL) or None,
            retain_llm_max_concurrent=int(os.getenv(ENV_RETAIN_LLM_MAX_CONCURRENT))
            if os.getenv(ENV_RETAIN_LLM_MAX_CONCURRENT)
            else None,
            retain_llm_max_retries=int(os.getenv(ENV_RETAIN_LLM_MAX_RETRIES))
            if os.getenv(ENV_RETAIN_LLM_MAX_RETRIES)
            else None,
            retain_llm_initial_backoff=float(os.getenv(ENV_RETAIN_LLM_INITIAL_BACKOFF))
            if os.getenv(ENV_RETAIN_LLM_INITIAL_BACKOFF)
            else None,
            retain_llm_max_backoff=float(os.getenv(ENV_RETAIN_LLM_MAX_BACKOFF))
            if os.getenv(ENV_RETAIN_LLM_MAX_BACKOFF)
            else None,
            retain_llm_timeout=float(os.getenv(ENV_RETAIN_LLM_TIMEOUT)) if os.getenv(ENV_RETAIN_LLM_TIMEOUT) else None,
            reflect_llm_provider=os.getenv(ENV_REFLECT_LLM_PROVIDER) or None,
            reflect_llm_api_key=os.getenv(ENV_REFLECT_LLM_API_KEY) or None,
            reflect_llm_model=os.getenv(ENV_REFLECT_LLM_MODEL)
            or (
                _get_default_model_for_provider(os.getenv(ENV_REFLECT_LLM_PROVIDER))
                if os.getenv(ENV_REFLECT_LLM_PROVIDER)
                else None
            ),
            reflect_llm_base_url=os.getenv(ENV_REFLECT_LLM_BASE_URL) or None,
            reflect_llm_max_concurrent=int(os.getenv(ENV_REFLECT_LLM_MAX_CONCURRENT))
            if os.getenv(ENV_REFLECT_LLM_MAX_CONCURRENT)
            else None,
            reflect_llm_max_retries=int(os.getenv(ENV_REFLECT_LLM_MAX_RETRIES))
            if os.getenv(ENV_REFLECT_LLM_MAX_RETRIES)
            else None,
            reflect_llm_initial_backoff=float(os.getenv(ENV_REFLECT_LLM_INITIAL_BACKOFF))
            if os.getenv(ENV_REFLECT_LLM_INITIAL_BACKOFF)
            else None,
            reflect_llm_max_backoff=float(os.getenv(ENV_REFLECT_LLM_MAX_BACKOFF))
            if os.getenv(ENV_REFLECT_LLM_MAX_BACKOFF)
            else None,
            reflect_llm_timeout=float(os.getenv(ENV_REFLECT_LLM_TIMEOUT))
            if os.getenv(ENV_REFLECT_LLM_TIMEOUT)
            else None,
            consolidation_llm_provider=os.getenv(ENV_CONSOLIDATION_LLM_PROVIDER) or None,
            consolidation_llm_api_key=os.getenv(ENV_CONSOLIDATION_LLM_API_KEY) or None,
            consolidation_llm_model=os.getenv(ENV_CONSOLIDATION_LLM_MODEL)
            or (
                _get_default_model_for_provider(os.getenv(ENV_CONSOLIDATION_LLM_PROVIDER))
                if os.getenv(ENV_CONSOLIDATION_LLM_PROVIDER)
                else None
            ),
            consolidation_llm_base_url=os.getenv(ENV_CONSOLIDATION_LLM_BASE_URL) or None,
            consolidation_llm_max_concurrent=int(os.getenv(ENV_CONSOLIDATION_LLM_MAX_CONCURRENT))
            if os.getenv(ENV_CONSOLIDATION_LLM_MAX_CONCURRENT)
            else None,
            consolidation_llm_max_retries=int(os.getenv(ENV_CONSOLIDATION_LLM_MAX_RETRIES))
            if os.getenv(ENV_CONSOLIDATION_LLM_MAX_RETRIES)
            else None,
            consolidation_llm_initial_backoff=float(os.getenv(ENV_CONSOLIDATION_LLM_INITIAL_BACKOFF))
            if os.getenv(ENV_CONSOLIDATION_LLM_INITIAL_BACKOFF)
            else None,
            consolidation_llm_max_backoff=float(os.getenv(ENV_CONSOLIDATION_LLM_MAX_BACKOFF))
            if os.getenv(ENV_CONSOLIDATION_LLM_MAX_BACKOFF)
            else None,
            consolidation_llm_timeout=float(os.getenv(ENV_CONSOLIDATION_LLM_TIMEOUT))
            if os.getenv(ENV_CONSOLIDATION_LLM_TIMEOUT)
            else None,
            # Embeddings
            embeddings_provider=os.getenv(ENV_EMBEDDINGS_PROVIDER, DEFAULT_EMBEDDINGS_PROVIDER),
            embeddings_local_model=os.getenv(ENV_EMBEDDINGS_LOCAL_MODEL, DEFAULT_EMBEDDINGS_LOCAL_MODEL),
            embeddings_local_force_cpu=os.getenv(
                ENV_EMBEDDINGS_LOCAL_FORCE_CPU, str(DEFAULT_EMBEDDINGS_LOCAL_FORCE_CPU)
            ).lower()
            in ("true", "1"),
            embeddings_local_trust_remote_code=os.getenv(
                ENV_EMBEDDINGS_LOCAL_TRUST_REMOTE_CODE, str(DEFAULT_EMBEDDINGS_LOCAL_TRUST_REMOTE_CODE)
            ).lower()
            in ("true", "1"),
            embeddings_tei_url=os.getenv(ENV_EMBEDDINGS_TEI_URL),
            embeddings_openai_base_url=os.getenv(ENV_EMBEDDINGS_OPENAI_BASE_URL) or None,
            # Cohere embeddings (with backward-compatible fallback to shared API key)
            embeddings_cohere_api_key=os.getenv(ENV_EMBEDDINGS_COHERE_API_KEY) or os.getenv(ENV_COHERE_API_KEY),
            embeddings_cohere_model=os.getenv(ENV_EMBEDDINGS_COHERE_MODEL, DEFAULT_EMBEDDINGS_COHERE_MODEL),
            embeddings_cohere_base_url=os.getenv(ENV_EMBEDDINGS_COHERE_BASE_URL) or None,
            # LiteLLM embeddings (with backward-compatible fallback to shared config)
            embeddings_litellm_api_base=os.getenv(ENV_EMBEDDINGS_LITELLM_API_BASE)
            or os.getenv(ENV_LITELLM_API_BASE, DEFAULT_LITELLM_API_BASE),
            embeddings_litellm_api_key=os.getenv(ENV_EMBEDDINGS_LITELLM_API_KEY) or os.getenv(ENV_LITELLM_API_KEY),
            embeddings_litellm_model=os.getenv(ENV_EMBEDDINGS_LITELLM_MODEL, DEFAULT_EMBEDDINGS_LITELLM_MODEL),
            # LiteLLM SDK embeddings (direct API access)
            embeddings_litellm_sdk_api_key=os.getenv(ENV_EMBEDDINGS_LITELLM_SDK_API_KEY),
            embeddings_litellm_sdk_model=os.getenv(
                ENV_EMBEDDINGS_LITELLM_SDK_MODEL, DEFAULT_EMBEDDINGS_LITELLM_SDK_MODEL
            ),
            embeddings_litellm_sdk_api_base=os.getenv(ENV_EMBEDDINGS_LITELLM_SDK_API_BASE) or None,
            # Reranker
            reranker_provider=os.getenv(ENV_RERANKER_PROVIDER, DEFAULT_RERANKER_PROVIDER),
            reranker_local_model=os.getenv(ENV_RERANKER_LOCAL_MODEL, DEFAULT_RERANKER_LOCAL_MODEL),
            reranker_local_force_cpu=os.getenv(
                ENV_RERANKER_LOCAL_FORCE_CPU, str(DEFAULT_RERANKER_LOCAL_FORCE_CPU)
            ).lower()
            in ("true", "1"),
            reranker_local_max_concurrent=int(
                os.getenv(ENV_RERANKER_LOCAL_MAX_CONCURRENT, str(DEFAULT_RERANKER_LOCAL_MAX_CONCURRENT))
            ),
            reranker_local_trust_remote_code=os.getenv(
                ENV_RERANKER_LOCAL_TRUST_REMOTE_CODE, str(DEFAULT_RERANKER_LOCAL_TRUST_REMOTE_CODE)
            ).lower()
            in ("true", "1"),
            reranker_tei_url=os.getenv(ENV_RERANKER_TEI_URL),
            reranker_tei_batch_size=int(os.getenv(ENV_RERANKER_TEI_BATCH_SIZE, str(DEFAULT_RERANKER_TEI_BATCH_SIZE))),
            reranker_tei_max_concurrent=int(
                os.getenv(ENV_RERANKER_TEI_MAX_CONCURRENT, str(DEFAULT_RERANKER_TEI_MAX_CONCURRENT))
            ),
            reranker_max_candidates=int(os.getenv(ENV_RERANKER_MAX_CANDIDATES, str(DEFAULT_RERANKER_MAX_CANDIDATES))),
            # Cohere reranker (with backward-compatible fallback to shared API key)
            reranker_cohere_api_key=os.getenv(ENV_RERANKER_COHERE_API_KEY) or os.getenv(ENV_COHERE_API_KEY),
            reranker_cohere_model=os.getenv(ENV_RERANKER_COHERE_MODEL, DEFAULT_RERANKER_COHERE_MODEL),
            reranker_cohere_base_url=os.getenv(ENV_RERANKER_COHERE_BASE_URL) or None,
            # LiteLLM reranker (with backward-compatible fallback to shared config)
            reranker_litellm_api_base=os.getenv(ENV_RERANKER_LITELLM_API_BASE)
            or os.getenv(ENV_LITELLM_API_BASE, DEFAULT_LITELLM_API_BASE),
            reranker_litellm_api_key=os.getenv(ENV_RERANKER_LITELLM_API_KEY) or os.getenv(ENV_LITELLM_API_KEY),
            reranker_litellm_model=os.getenv(ENV_RERANKER_LITELLM_MODEL, DEFAULT_RERANKER_LITELLM_MODEL),
            # LiteLLM SDK reranker (direct API access)
            reranker_litellm_sdk_api_key=os.getenv(ENV_RERANKER_LITELLM_SDK_API_KEY),
            reranker_litellm_sdk_model=os.getenv(ENV_RERANKER_LITELLM_SDK_MODEL, DEFAULT_RERANKER_LITELLM_SDK_MODEL),
            reranker_litellm_sdk_api_base=os.getenv(ENV_RERANKER_LITELLM_SDK_API_BASE) or None,
            # Server
            host=os.getenv(ENV_HOST, DEFAULT_HOST),
            port=int(os.getenv(ENV_PORT, DEFAULT_PORT)),
            base_path=os.getenv(ENV_BASE_PATH, DEFAULT_BASE_PATH),
            log_level=os.getenv(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL),
            log_format=os.getenv(ENV_LOG_FORMAT, DEFAULT_LOG_FORMAT).lower(),
            mcp_enabled=os.getenv(ENV_MCP_ENABLED, str(DEFAULT_MCP_ENABLED)).lower() == "true",
            enable_bank_config_api=os.getenv(ENV_ENABLE_BANK_CONFIG_API, str(DEFAULT_ENABLE_BANK_CONFIG_API)).lower()
            == "true",
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
            retain_custom_instructions=os.getenv(ENV_RETAIN_CUSTOM_INSTRUCTIONS) or DEFAULT_RETAIN_CUSTOM_INSTRUCTIONS,
            retain_batch_tokens=int(os.getenv(ENV_RETAIN_BATCH_TOKENS, str(DEFAULT_RETAIN_BATCH_TOKENS))),
            retain_batch_enabled=os.getenv(ENV_RETAIN_BATCH_ENABLED, str(DEFAULT_RETAIN_BATCH_ENABLED)).lower()
            == "true",
            retain_batch_poll_interval_seconds=int(
                os.getenv(ENV_RETAIN_BATCH_POLL_INTERVAL_SECONDS, str(DEFAULT_RETAIN_BATCH_POLL_INTERVAL_SECONDS))
            ),
            # File storage
            file_storage_type=os.getenv(ENV_FILE_STORAGE_TYPE, DEFAULT_FILE_STORAGE_TYPE),
            file_storage_s3_bucket=os.getenv(ENV_FILE_STORAGE_S3_BUCKET) or None,
            file_storage_s3_region=os.getenv(ENV_FILE_STORAGE_S3_REGION) or None,
            file_storage_s3_endpoint=os.getenv(ENV_FILE_STORAGE_S3_ENDPOINT) or None,
            file_storage_s3_access_key_id=os.getenv(ENV_FILE_STORAGE_S3_ACCESS_KEY_ID) or None,
            file_storage_s3_secret_access_key=os.getenv(ENV_FILE_STORAGE_S3_SECRET_ACCESS_KEY) or None,
            file_storage_gcs_bucket=os.getenv(ENV_FILE_STORAGE_GCS_BUCKET) or None,
            file_storage_gcs_service_account_key=os.getenv(ENV_FILE_STORAGE_GCS_SERVICE_ACCOUNT_KEY) or None,
            file_storage_azure_container=os.getenv(ENV_FILE_STORAGE_AZURE_CONTAINER) or None,
            file_storage_azure_account_name=os.getenv(ENV_FILE_STORAGE_AZURE_ACCOUNT_NAME) or None,
            file_storage_azure_account_key=os.getenv(ENV_FILE_STORAGE_AZURE_ACCOUNT_KEY) or None,
            file_parser=os.getenv(ENV_FILE_PARSER, DEFAULT_FILE_PARSER),
            file_parser_iris_token=os.getenv(ENV_FILE_PARSER_IRIS_TOKEN) or None,
            file_parser_iris_org_id=os.getenv(ENV_FILE_PARSER_IRIS_ORG_ID) or None,
            file_conversion_max_batch_size_mb=int(
                os.getenv(ENV_FILE_CONVERSION_MAX_BATCH_SIZE_MB, str(DEFAULT_FILE_CONVERSION_MAX_BATCH_SIZE_MB))
            ),
            file_conversion_max_batch_size=int(
                os.getenv(ENV_FILE_CONVERSION_MAX_BATCH_SIZE, str(DEFAULT_FILE_CONVERSION_MAX_BATCH_SIZE))
            ),
            enable_file_upload_api=os.getenv(ENV_ENABLE_FILE_UPLOAD_API, str(DEFAULT_ENABLE_FILE_UPLOAD_API)).lower()
            == "true",
            file_delete_after_retain=os.getenv(
                ENV_FILE_DELETE_AFTER_RETAIN, str(DEFAULT_FILE_DELETE_AFTER_RETAIN)
            ).lower()
            == "true",
            # Observations settings (consolidated knowledge from facts)
            enable_observations=os.getenv(ENV_ENABLE_OBSERVATIONS, str(DEFAULT_ENABLE_OBSERVATIONS)).lower() == "true",
            consolidation_batch_size=int(
                os.getenv(ENV_CONSOLIDATION_BATCH_SIZE, str(DEFAULT_CONSOLIDATION_BATCH_SIZE))
            ),
            consolidation_max_tokens=int(
                os.getenv(ENV_CONSOLIDATION_MAX_TOKENS, str(DEFAULT_CONSOLIDATION_MAX_TOKENS))
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
            worker_http_port=int(os.getenv(ENV_WORKER_HTTP_PORT, str(DEFAULT_WORKER_HTTP_PORT))),
            worker_max_slots=int(os.getenv(ENV_WORKER_MAX_SLOTS, str(DEFAULT_WORKER_MAX_SLOTS))),
            worker_consolidation_max_slots=int(
                os.getenv(ENV_WORKER_CONSOLIDATION_MAX_SLOTS, str(DEFAULT_WORKER_CONSOLIDATION_MAX_SLOTS))
            ),
            # Reflect agent settings
            reflect_max_iterations=int(os.getenv(ENV_REFLECT_MAX_ITERATIONS, str(DEFAULT_REFLECT_MAX_ITERATIONS))),
            # OpenTelemetry tracing configuration
            otel_traces_enabled=os.getenv(ENV_OTEL_TRACES_ENABLED, str(DEFAULT_OTEL_TRACES_ENABLED)).lower()
            in ("true", "1", "yes"),
            otel_exporter_otlp_endpoint=os.getenv(ENV_OTEL_EXPORTER_OTLP_ENDPOINT) or None,
            otel_exporter_otlp_headers=os.getenv(ENV_OTEL_EXPORTER_OTLP_HEADERS) or None,
            otel_service_name=os.getenv(ENV_OTEL_SERVICE_NAME, DEFAULT_OTEL_SERVICE_NAME),
            otel_deployment_environment=os.getenv(ENV_OTEL_DEPLOYMENT_ENVIRONMENT, DEFAULT_OTEL_DEPLOYMENT_ENVIRONMENT),
        )
        config.validate()
        return config

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
        logger.info(f"Database: {self.database_url} (schema: {self.database_schema})")
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


def get_config() -> StaticConfigProxy:
    """
    Get global configuration with ONLY static (non-configurable) fields accessible.

    This returns a proxy that prevents access to bank-configurable fields
    (like enable_observations, retain_chunk_size, etc.).

    For bank-specific configuration, use:
        config_resolver.resolve_full_config(bank_id, context)

    This design prevents accidentally using global defaults when bank-specific
    overrides exist.

    Returns:
        StaticConfigProxy that only exposes static infrastructure fields

    Raises:
        ConfigFieldAccessError: If you try to access a bank-configurable field
    """
    return StaticConfigProxy(_get_raw_config())


def _get_raw_config() -> HindsightConfig:
    """
    Get raw config (internal use only).

    INTERNAL USE ONLY. Do not use this directly in application code.
    Use get_config() for static fields or ConfigResolver.resolve_full_config() for bank-specific config.
    """
    global _config_cache
    if _config_cache is None:
        _config_cache = HindsightConfig.from_env()
    return _config_cache


def clear_config_cache() -> None:
    """Clear the config cache. Useful for testing or reloading config."""
    global _config_cache
    _config_cache = None

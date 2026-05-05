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
from typing import Any, Literal

from dotenv import find_dotenv, load_dotenv

from ._vector_index import validate_extension
from .utils import mask_network_location

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
ENV_DATABASE_BACKEND = "HINDSIGHT_API_DATABASE_BACKEND"
ENV_DATABASE_URL = "HINDSIGHT_API_DATABASE_URL"
ENV_READ_DATABASE_URL = "HINDSIGHT_API_READ_DATABASE_URL"
ENV_READ_DB_POOL_MIN_SIZE = "HINDSIGHT_API_READ_DB_POOL_MIN_SIZE"
ENV_READ_DB_POOL_MAX_SIZE = "HINDSIGHT_API_READ_DB_POOL_MAX_SIZE"
ENV_MIGRATION_DATABASE_URL = "HINDSIGHT_API_MIGRATION_DATABASE_URL"
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
ENV_LLM_EXTRA_BODY = "HINDSIGHT_API_LLM_EXTRA_BODY"
ENV_LLM_DEFAULT_HEADERS = "HINDSIGHT_API_LLM_DEFAULT_HEADERS"

# Defaults for service tiers
DEFAULT_LLM_GROQ_SERVICE_TIER = "auto"  # "on_demand", "flex", or "auto"
DEFAULT_LLM_OPENAI_SERVICE_TIER = None  # None (default) or "flex" (50% cheaper)
DEFAULT_LLM_EXTRA_BODY = None  # None = no extra body params; JSON dict merged into OpenAI extra_body
DEFAULT_LLM_DEFAULT_HEADERS = (
    None  # None = no extra headers; JSON dict passed as default_headers to provider SDK clients
)

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
ENV_EMBEDDINGS_OPENAI_BATCH_SIZE = "HINDSIGHT_API_EMBEDDINGS_OPENAI_BATCH_SIZE"

# Gemini/Vertex AI embeddings configuration
ENV_EMBEDDINGS_GEMINI_API_KEY = "HINDSIGHT_API_EMBEDDINGS_GEMINI_API_KEY"
ENV_EMBEDDINGS_GEMINI_MODEL = "HINDSIGHT_API_EMBEDDINGS_GEMINI_MODEL"
ENV_EMBEDDINGS_GEMINI_OUTPUT_DIMENSIONALITY = "HINDSIGHT_API_EMBEDDINGS_GEMINI_OUTPUT_DIMENSIONALITY"
ENV_EMBEDDINGS_GEMINI_FORCE_IPV4 = "HINDSIGHT_API_EMBEDDINGS_GEMINI_FORCE_IPV4"
ENV_EMBEDDINGS_VERTEXAI_PROJECT_ID = "HINDSIGHT_API_EMBEDDINGS_VERTEXAI_PROJECT_ID"
ENV_EMBEDDINGS_VERTEXAI_REGION = "HINDSIGHT_API_EMBEDDINGS_VERTEXAI_REGION"
ENV_EMBEDDINGS_VERTEXAI_SERVICE_ACCOUNT_KEY = "HINDSIGHT_API_EMBEDDINGS_VERTEXAI_SERVICE_ACCOUNT_KEY"

# Cohere configuration (separate for embeddings and reranker)
ENV_EMBEDDINGS_COHERE_API_KEY = "HINDSIGHT_API_EMBEDDINGS_COHERE_API_KEY"
ENV_EMBEDDINGS_COHERE_MODEL = "HINDSIGHT_API_EMBEDDINGS_COHERE_MODEL"
ENV_EMBEDDINGS_COHERE_BASE_URL = "HINDSIGHT_API_EMBEDDINGS_COHERE_BASE_URL"
ENV_EMBEDDINGS_COHERE_OUTPUT_DIMENSIONS = "HINDSIGHT_API_EMBEDDINGS_COHERE_OUTPUT_DIMENSIONS"
ENV_RERANKER_COHERE_API_KEY = "HINDSIGHT_API_RERANKER_COHERE_API_KEY"
ENV_RERANKER_COHERE_MODEL = "HINDSIGHT_API_RERANKER_COHERE_MODEL"
ENV_RERANKER_COHERE_BASE_URL = "HINDSIGHT_API_RERANKER_COHERE_BASE_URL"

# OpenRouter configuration (embeddings and reranker)
ENV_OPENROUTER_API_KEY = "HINDSIGHT_API_OPENROUTER_API_KEY"
ENV_EMBEDDINGS_OPENROUTER_API_KEY = "HINDSIGHT_API_EMBEDDINGS_OPENROUTER_API_KEY"
ENV_EMBEDDINGS_OPENROUTER_MODEL = "HINDSIGHT_API_EMBEDDINGS_OPENROUTER_MODEL"
ENV_RERANKER_OPENROUTER_API_KEY = "HINDSIGHT_API_RERANKER_OPENROUTER_API_KEY"
ENV_RERANKER_OPENROUTER_MODEL = "HINDSIGHT_API_RERANKER_OPENROUTER_MODEL"

# Deprecated: Legacy shared Cohere API key (for backward compatibility)
ENV_COHERE_API_KEY = "HINDSIGHT_API_COHERE_API_KEY"

# LiteLLM configuration (separate for embeddings and reranker)
ENV_EMBEDDINGS_LITELLM_API_BASE = "HINDSIGHT_API_EMBEDDINGS_LITELLM_API_BASE"
ENV_EMBEDDINGS_LITELLM_API_KEY = "HINDSIGHT_API_EMBEDDINGS_LITELLM_API_KEY"
ENV_EMBEDDINGS_LITELLM_MODEL = "HINDSIGHT_API_EMBEDDINGS_LITELLM_MODEL"
ENV_RERANKER_LITELLM_API_BASE = "HINDSIGHT_API_RERANKER_LITELLM_API_BASE"
ENV_RERANKER_LITELLM_API_KEY = "HINDSIGHT_API_RERANKER_LITELLM_API_KEY"
ENV_RERANKER_LITELLM_MODEL = "HINDSIGHT_API_RERANKER_LITELLM_MODEL"
ENV_RERANKER_LITELLM_MAX_TOKENS_PER_DOC = "HINDSIGHT_API_RERANKER_LITELLM_MAX_TOKENS_PER_DOC"

# LiteLLM SDK configuration (direct API access, no proxy needed)
ENV_EMBEDDINGS_LITELLM_SDK_API_KEY = "HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_API_KEY"
ENV_EMBEDDINGS_LITELLM_SDK_MODEL = "HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_MODEL"
ENV_EMBEDDINGS_LITELLM_SDK_API_BASE = "HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_API_BASE"
ENV_EMBEDDINGS_LITELLM_SDK_OUTPUT_DIMENSIONS = "HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_OUTPUT_DIMENSIONS"
ENV_EMBEDDINGS_LITELLM_SDK_ENCODING_FORMAT = "HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_ENCODING_FORMAT"
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
ENV_RERANKER_LOCAL_FP16 = "HINDSIGHT_API_RERANKER_LOCAL_FP16"
ENV_RERANKER_LOCAL_BUCKET_BATCHING = "HINDSIGHT_API_RERANKER_LOCAL_BUCKET_BATCHING"
ENV_RERANKER_LOCAL_BATCH_SIZE = "HINDSIGHT_API_RERANKER_LOCAL_BATCH_SIZE"
ENV_RERANKER_TEI_URL = "HINDSIGHT_API_RERANKER_TEI_URL"
ENV_RERANKER_TEI_BATCH_SIZE = "HINDSIGHT_API_RERANKER_TEI_BATCH_SIZE"
ENV_RERANKER_TEI_MAX_CONCURRENT = "HINDSIGHT_API_RERANKER_TEI_MAX_CONCURRENT"
ENV_RERANKER_TEI_HTTP_TIMEOUT = "HINDSIGHT_API_RERANKER_TEI_HTTP_TIMEOUT"
ENV_RERANKER_MAX_CANDIDATES = "HINDSIGHT_API_RERANKER_MAX_CANDIDATES"
ENV_RERANKER_FLASHRANK_MODEL = "HINDSIGHT_API_RERANKER_FLASHRANK_MODEL"
ENV_RERANKER_FLASHRANK_CACHE_DIR = "HINDSIGHT_API_RERANKER_FLASHRANK_CACHE_DIR"
ENV_RERANKER_FLASHRANK_CPU_MEM_ARENA = "HINDSIGHT_API_RERANKER_FLASHRANK_CPU_MEM_ARENA"

# ZeroEntropy configuration (reranker only)
ENV_RERANKER_ZEROENTROPY_API_KEY = "HINDSIGHT_API_RERANKER_ZEROENTROPY_API_KEY"
ENV_RERANKER_ZEROENTROPY_MODEL = "HINDSIGHT_API_RERANKER_ZEROENTROPY_MODEL"
ENV_RERANKER_ZEROENTROPY_BASE_URL = "HINDSIGHT_API_RERANKER_ZEROENTROPY_BASE_URL"

# SiliconFlow configuration (reranker only; Cohere-compatible /rerank endpoint)
ENV_RERANKER_SILICONFLOW_API_KEY = "HINDSIGHT_API_RERANKER_SILICONFLOW_API_KEY"
ENV_RERANKER_SILICONFLOW_MODEL = "HINDSIGHT_API_RERANKER_SILICONFLOW_MODEL"
ENV_RERANKER_SILICONFLOW_BASE_URL = "HINDSIGHT_API_RERANKER_SILICONFLOW_BASE_URL"

# Google Discovery Engine reranker configuration
ENV_RERANKER_GOOGLE_MODEL = "HINDSIGHT_API_RERANKER_GOOGLE_MODEL"
ENV_RERANKER_GOOGLE_PROJECT_ID = "HINDSIGHT_API_RERANKER_GOOGLE_PROJECT_ID"
ENV_RERANKER_GOOGLE_SERVICE_ACCOUNT_KEY = "HINDSIGHT_API_RERANKER_GOOGLE_SERVICE_ACCOUNT_KEY"

ENV_VECTOR_EXTENSION = "HINDSIGHT_API_VECTOR_EXTENSION"
ENV_TEXT_SEARCH_EXTENSION = "HINDSIGHT_API_TEXT_SEARCH_EXTENSION"

ENV_HOST = "HINDSIGHT_API_HOST"
ENV_PORT = "HINDSIGHT_API_PORT"
ENV_BASE_PATH = "HINDSIGHT_API_BASE_PATH"
ENV_LOG_LEVEL = "HINDSIGHT_API_LOG_LEVEL"
ENV_LOG_FORMAT = "HINDSIGHT_API_LOG_FORMAT"
ENV_LOG_JSON_FIELDS = "HINDSIGHT_API_LOG_JSON_FIELDS"
ENV_WORKERS = "HINDSIGHT_API_WORKERS"
ENV_MCP_ENABLED = "HINDSIGHT_API_MCP_ENABLED"
ENV_MCP_ENABLED_TOOLS = "HINDSIGHT_API_MCP_ENABLED_TOOLS"
ENV_MCP_STATELESS = "HINDSIGHT_API_MCP_STATELESS"
ENV_ENABLE_BANK_CONFIG_API = "HINDSIGHT_API_ENABLE_BANK_CONFIG_API"
ENV_DEFAULT_BANK_TEMPLATE = "HINDSIGHT_API_DEFAULT_BANK_TEMPLATE"
ENV_GRAPH_RETRIEVER = "HINDSIGHT_API_GRAPH_RETRIEVER"
ENV_RECALL_MAX_CONCURRENT = "HINDSIGHT_API_RECALL_MAX_CONCURRENT"
ENV_RECALL_CONNECTION_BUDGET = "HINDSIGHT_API_RECALL_CONNECTION_BUDGET"
ENV_RECALL_MAX_QUERY_TOKENS = "HINDSIGHT_API_RECALL_MAX_QUERY_TOKENS"
ENV_MENTAL_MODEL_REFRESH_CONCURRENCY = "HINDSIGHT_API_MENTAL_MODEL_REFRESH_CONCURRENCY"
ENV_LINK_EXPANSION_PER_ENTITY_LIMIT = "HINDSIGHT_API_LINK_EXPANSION_PER_ENTITY_LIMIT"
ENV_LINK_EXPANSION_TIMEOUT = "HINDSIGHT_API_LINK_EXPANSION_TIMEOUT"

# OpenTelemetry tracing configuration
ENV_OTEL_TRACES_ENABLED = "HINDSIGHT_API_OTEL_TRACES_ENABLED"
ENV_OTEL_EXPORTER_OTLP_ENDPOINT = "HINDSIGHT_API_OTEL_EXPORTER_OTLP_ENDPOINT"
ENV_OTEL_EXPORTER_OTLP_HEADERS = "HINDSIGHT_API_OTEL_EXPORTER_OTLP_HEADERS"
ENV_OTEL_SERVICE_NAME = "HINDSIGHT_API_OTEL_SERVICE_NAME"
ENV_OTEL_DEPLOYMENT_ENVIRONMENT = "HINDSIGHT_API_OTEL_DEPLOYMENT_ENVIRONMENT"
ENV_METRICS_INCLUDE_BANK_ID = "HINDSIGHT_API_METRICS_INCLUDE_BANK_ID"

# Vertex AI configuration
ENV_LLM_VERTEXAI_PROJECT_ID = "HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID"
ENV_LLM_VERTEXAI_REGION = "HINDSIGHT_API_LLM_VERTEXAI_REGION"
ENV_LLM_VERTEXAI_SERVICE_ACCOUNT_KEY = "HINDSIGHT_API_LLM_VERTEXAI_SERVICE_ACCOUNT_KEY"

# Gemini safety settings
ENV_LLM_GEMINI_SAFETY_SETTINGS = "HINDSIGHT_API_LLM_GEMINI_SAFETY_SETTINGS"

# Retain settings
ENV_RETAIN_MAX_COMPLETION_TOKENS = "HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS"
ENV_RETAIN_CHUNK_SIZE = "HINDSIGHT_API_RETAIN_CHUNK_SIZE"
ENV_RETAIN_EXTRACT_CAUSAL_LINKS = "HINDSIGHT_API_RETAIN_EXTRACT_CAUSAL_LINKS"
ENV_RETAIN_EXTRACTION_MODE = "HINDSIGHT_API_RETAIN_EXTRACTION_MODE"
ENV_RETAIN_MISSION = "HINDSIGHT_API_RETAIN_MISSION"
ENV_RETAIN_CUSTOM_INSTRUCTIONS = "HINDSIGHT_API_RETAIN_CUSTOM_INSTRUCTIONS"
ENV_RETAIN_DEFAULT_STRATEGY = "HINDSIGHT_API_RETAIN_DEFAULT_STRATEGY"
ENV_RETAIN_BATCH_TOKENS = "HINDSIGHT_API_RETAIN_BATCH_TOKENS"
ENV_RETAIN_ENTITY_LOOKUP = "HINDSIGHT_API_RETAIN_ENTITY_LOOKUP"
ENV_RETAIN_BATCH_ENABLED = "HINDSIGHT_API_RETAIN_BATCH_ENABLED"
ENV_RETAIN_BATCH_POLL_INTERVAL_SECONDS = "HINDSIGHT_API_RETAIN_BATCH_POLL_INTERVAL_SECONDS"
ENV_RETAIN_CHUNK_BATCH_SIZE = "HINDSIGHT_API_RETAIN_CHUNK_BATCH_SIZE"

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
ENV_FILE_PARSER_ALLOWLIST = "HINDSIGHT_API_FILE_PARSER_ALLOWLIST"
ENV_FILE_PARSER_IRIS_TOKEN = "HINDSIGHT_API_FILE_PARSER_IRIS_TOKEN"
ENV_FILE_PARSER_IRIS_ORG_ID = "HINDSIGHT_API_FILE_PARSER_IRIS_ORG_ID"
ENV_FILE_PARSER_LLAMA_PARSE_API_KEY = "HINDSIGHT_API_FILE_PARSER_LLAMA_PARSE_API_KEY"
ENV_FILE_CONVERSION_MAX_BATCH_SIZE_MB = "HINDSIGHT_API_FILE_CONVERSION_MAX_BATCH_SIZE_MB"
ENV_FILE_CONVERSION_MAX_BATCH_SIZE = "HINDSIGHT_API_FILE_CONVERSION_MAX_BATCH_SIZE"
ENV_ENABLE_FILE_UPLOAD_API = "HINDSIGHT_API_ENABLE_FILE_UPLOAD_API"
ENV_FILE_DELETE_AFTER_RETAIN = "HINDSIGHT_API_FILE_DELETE_AFTER_RETAIN"

# Observations settings (consolidated knowledge from facts)
ENV_ENABLE_OBSERVATIONS = "HINDSIGHT_API_ENABLE_OBSERVATIONS"
ENV_CONSOLIDATION_BATCH_SIZE = "HINDSIGHT_API_CONSOLIDATION_BATCH_SIZE"
ENV_CONSOLIDATION_MAX_MEMORIES_PER_ROUND = "HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_ROUND"
ENV_CONSOLIDATION_LLM_BATCH_SIZE = "HINDSIGHT_API_CONSOLIDATION_LLM_BATCH_SIZE"
ENV_CONSOLIDATION_MAX_TOKENS = "HINDSIGHT_API_CONSOLIDATION_MAX_TOKENS"
ENV_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS = "HINDSIGHT_API_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS"
ENV_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS_PER_OBSERVATION = (
    "HINDSIGHT_API_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS_PER_OBSERVATION"
)
ENV_CONSOLIDATION_RECALL_BUDGET = "HINDSIGHT_API_CONSOLIDATION_RECALL_BUDGET"
ENV_CONSOLIDATION_MAX_ATTEMPTS = "HINDSIGHT_API_CONSOLIDATION_MAX_ATTEMPTS"
ENV_OBSERVATIONS_MISSION = "HINDSIGHT_API_OBSERVATIONS_MISSION"
ENV_MAX_OBSERVATIONS_PER_SCOPE = "HINDSIGHT_API_MAX_OBSERVATIONS_PER_SCOPE"
ENV_ENABLE_OBSERVATION_HISTORY = "HINDSIGHT_API_ENABLE_OBSERVATION_HISTORY"
ENV_ENABLE_MENTAL_MODEL_HISTORY = "HINDSIGHT_API_ENABLE_MENTAL_MODEL_HISTORY"

# Webhook configuration (global, static - server-level only)
ENV_WEBHOOK_URL = "HINDSIGHT_API_WEBHOOK_URL"
ENV_WEBHOOK_SECRET = "HINDSIGHT_API_WEBHOOK_SECRET"
ENV_WEBHOOK_EVENT_TYPES = "HINDSIGHT_API_WEBHOOK_EVENT_TYPES"
ENV_WEBHOOK_DELIVERY_POLL_INTERVAL_SECONDS = "HINDSIGHT_API_WEBHOOK_DELIVERY_POLL_INTERVAL_SECONDS"

# Built-in llama.cpp configuration (for provider=llamacpp)
ENV_LLAMACPP_MODEL_PATH = "HINDSIGHT_API_LLAMACPP_MODEL_PATH"
ENV_LLAMACPP_GPU_LAYERS = "HINDSIGHT_API_LLAMACPP_GPU_LAYERS"
ENV_LLAMACPP_CONTEXT_SIZE = "HINDSIGHT_API_LLAMACPP_CONTEXT_SIZE"
ENV_LLAMACPP_CHAT_FORMAT = "HINDSIGHT_API_LLAMACPP_CHAT_FORMAT"
ENV_LLAMACPP_NO_GRAMMAR = "HINDSIGHT_API_LLAMACPP_NO_GRAMMAR"
ENV_LLAMACPP_EXTRA_ARGS = "HINDSIGHT_API_LLAMACPP_EXTRA_ARGS"

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
ENV_DB_STATEMENT_TIMEOUT = "HINDSIGHT_API_DB_STATEMENT_TIMEOUT"

# Worker configuration (distributed task processing)
ENV_WORKER_ENABLED = "HINDSIGHT_API_WORKER_ENABLED"
ENV_WORKER_ID = "HINDSIGHT_API_WORKER_ID"
ENV_WORKER_POLL_INTERVAL_MS = "HINDSIGHT_API_WORKER_POLL_INTERVAL_MS"
ENV_WORKER_MAX_RETRIES = "HINDSIGHT_API_WORKER_MAX_RETRIES"
ENV_WORKER_HTTP_PORT = "HINDSIGHT_API_WORKER_HTTP_PORT"
ENV_WORKER_MAX_SLOTS = "HINDSIGHT_API_WORKER_MAX_SLOTS"

# Per-operation-type slot reservations. Each entry maps an operation_type
# (as stored in async_operations.operation_type) to its env var and default.
# Adding a new operation type here is the ONLY change needed to make it
# reservable via env var — config fields, from_env(), and the
# worker_slot_reservations property all derive from this dict.
WORKER_SLOT_RESERVATION_TYPES: dict[str, tuple[str, int]] = {
    "consolidation": ("HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS", 2),
    "retain": ("HINDSIGHT_API_WORKER_RETAIN_MAX_SLOTS", 0),
    "file_convert_retain": ("HINDSIGHT_API_WORKER_FILE_CONVERT_RETAIN_MAX_SLOTS", 0),
    "refresh_mental_model": ("HINDSIGHT_API_WORKER_REFRESH_MENTAL_MODEL_MAX_SLOTS", 0),
}
ENV_RETAIN_MAX_CONCURRENT = "HINDSIGHT_API_RETAIN_MAX_CONCURRENT"

# Reflect agent settings
ENV_REFLECT_MAX_ITERATIONS = "HINDSIGHT_API_REFLECT_MAX_ITERATIONS"
ENV_REFLECT_MAX_CONTEXT_TOKENS = "HINDSIGHT_API_REFLECT_MAX_CONTEXT_TOKENS"
ENV_REFLECT_WALL_TIMEOUT = "HINDSIGHT_API_REFLECT_WALL_TIMEOUT"
ENV_REFLECT_MISSION = "HINDSIGHT_API_REFLECT_MISSION"
ENV_REFLECT_SOURCE_FACTS_MAX_TOKENS = "HINDSIGHT_API_REFLECT_SOURCE_FACTS_MAX_TOKENS"
ENV_RECALL_INCLUDE_CHUNKS = "HINDSIGHT_API_RECALL_INCLUDE_CHUNKS"
ENV_RECALL_MAX_TOKENS = "HINDSIGHT_API_RECALL_MAX_TOKENS"
ENV_RECALL_CHUNKS_MAX_TOKENS = "HINDSIGHT_API_RECALL_CHUNKS_MAX_TOKENS"

# Recall budget mapping (budget enum -> thinking_budget integer)
ENV_RECALL_BUDGET_FUNCTION = "HINDSIGHT_API_RECALL_BUDGET_FUNCTION"
ENV_RECALL_BUDGET_FIXED_LOW = "HINDSIGHT_API_RECALL_BUDGET_FIXED_LOW"
ENV_RECALL_BUDGET_FIXED_MID = "HINDSIGHT_API_RECALL_BUDGET_FIXED_MID"
ENV_RECALL_BUDGET_FIXED_HIGH = "HINDSIGHT_API_RECALL_BUDGET_FIXED_HIGH"
ENV_RECALL_BUDGET_ADAPTIVE_LOW = "HINDSIGHT_API_RECALL_BUDGET_ADAPTIVE_LOW"
ENV_RECALL_BUDGET_ADAPTIVE_MID = "HINDSIGHT_API_RECALL_BUDGET_ADAPTIVE_MID"
ENV_RECALL_BUDGET_ADAPTIVE_HIGH = "HINDSIGHT_API_RECALL_BUDGET_ADAPTIVE_HIGH"
ENV_RECALL_BUDGET_MIN = "HINDSIGHT_API_RECALL_BUDGET_MIN"
ENV_RECALL_BUDGET_MAX = "HINDSIGHT_API_RECALL_BUDGET_MAX"

# Audit log settings
ENV_AUDIT_LOG_ENABLED = "HINDSIGHT_API_AUDIT_LOG_ENABLED"
ENV_AUDIT_LOG_ACTIONS = "HINDSIGHT_API_AUDIT_LOG_ACTIONS"
ENV_AUDIT_LOG_RETENTION_DAYS = "HINDSIGHT_API_AUDIT_LOG_RETENTION_DAYS"

# Disposition settings
ENV_DISPOSITION_SKEPTICISM = "HINDSIGHT_API_DISPOSITION_SKEPTICISM"
ENV_DISPOSITION_LITERALISM = "HINDSIGHT_API_DISPOSITION_LITERALISM"
ENV_DISPOSITION_EMPATHY = "HINDSIGHT_API_DISPOSITION_EMPATHY"

# Default values
DEFAULT_DATABASE_BACKEND = "postgresql"
DEFAULT_DATABASE_URL = "pg0"
DEFAULT_DATABASE_SCHEMA = "public"
DEFAULT_LLM_PROVIDER = "openai"

# Provider-specific default models
PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
    "gemini": "gemini-2.5-flash",
    "groq": "openai/gpt-oss-120b",
    "minimax": "MiniMax-M2.7",
    "deepseek": "deepseek-v4-flash",
    "ollama": "gemma3:12b",
    "llamacpp": "gemma-4-e2b-it",
    "lmstudio": "local-model",
    "vertexai": "google/gemini-2.5-flash-lite",
    "openai-codex": "gpt-5.4-mini",
    "claude-code": "claude-sonnet-4-5-20250929",
    "mock": "mock-model",
    "none": "none",
    "litellm": "gpt-4o-mini",
    "bedrock": "us.amazon.nova-2-lite-v1:0",
    "volcano": "doubao-pro-32k",
    "openrouter": "qwen/qwen3.5-9b",
}
DEFAULT_LLM_MODEL = "gpt-4o-mini"  # Fallback if provider not in table
# Built-in llama.cpp defaults
DEFAULT_LLAMACPP_GPU_LAYERS = -1  # -1 = offload all layers to GPU (Metal/CUDA)
DEFAULT_LLAMACPP_CONTEXT_SIZE = 8192
DEFAULT_LLAMACPP_CHAT_FORMAT = None  # None = auto-detect from GGUF metadata
DEFAULT_LLAMACPP_NO_GRAMMAR = False  # True = disable JSON grammar enforcement (faster but less reliable)
DEFAULT_LLAMACPP_EXTRA_ARGS = None  # Space-separated extra CLI args for llama.cpp server

DEFAULT_LLM_MAX_CONCURRENT = 32
DEFAULT_LLM_MAX_RETRIES = 3  # Max retry attempts for LLM API calls
DEFAULT_LLM_INITIAL_BACKOFF = 1.0  # Initial backoff in seconds for retry exponential backoff
DEFAULT_LLM_MAX_BACKOFF = 60.0  # Max backoff cap in seconds for retry exponential backoff
DEFAULT_LLM_TIMEOUT = 120.0  # seconds

# Vertex AI defaults
DEFAULT_LLM_VERTEXAI_PROJECT_ID = None  # Required for Vertex AI
DEFAULT_LLM_VERTEXAI_REGION = "us-central1"
DEFAULT_LLM_VERTEXAI_SERVICE_ACCOUNT_KEY = None  # Optional, uses ADC if not set

# Gemini safety settings defaults
DEFAULT_LLM_GEMINI_SAFETY_SETTINGS = None  # None = use Gemini default safety settings

DEFAULT_EMBEDDINGS_PROVIDER = "local"
DEFAULT_EMBEDDINGS_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDINGS_LOCAL_FORCE_CPU = False  # Force CPU mode for local embeddings (avoids MPS/XPC issues on macOS)
DEFAULT_EMBEDDINGS_LOCAL_TRUST_REMOTE_CODE = False  # Security: disabled by default, required for some models
DEFAULT_EMBEDDINGS_OPENAI_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDINGS_OPENAI_BATCH_SIZE = 100
DEFAULT_EMBEDDINGS_GEMINI_MODEL = "gemini-embedding-001"
DEFAULT_EMBEDDINGS_GEMINI_OUTPUT_DIMENSIONALITY = 768
DEFAULT_EMBEDDINGS_GEMINI_FORCE_IPV4 = False
DEFAULT_EMBEDDING_DIMENSION = 384

DEFAULT_RERANKER_PROVIDER = "local"
DEFAULT_RERANKER_LOCAL_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANKER_LOCAL_FORCE_CPU = False  # Force CPU mode for local reranker (avoids MPS/XPC issues on macOS)
DEFAULT_RERANKER_LOCAL_MAX_CONCURRENT = 4  # Limit concurrent CPU-bound reranking to prevent thrashing
DEFAULT_RERANKER_LOCAL_TRUST_REMOTE_CODE = (
    False  # Security: disabled by default, required for some models like jina-reranker-v2
)
DEFAULT_RERANKER_LOCAL_FP16 = False  # FP16 inference: opt-in, faster on MPS/CUDA (not CPU)
DEFAULT_RERANKER_LOCAL_BUCKET_BATCHING = False  # Length-sorted bucket batching: opt-in, 36-54% speedup
DEFAULT_RERANKER_LOCAL_BATCH_SIZE = 32  # Batch size for local reranker predict() calls
DEFAULT_RERANKER_TEI_BATCH_SIZE = 128
DEFAULT_RERANKER_TEI_MAX_CONCURRENT = 8
DEFAULT_RERANKER_TEI_HTTP_TIMEOUT = 30.0  # HTTP timeout for TEI reranker requests (seconds)
DEFAULT_RERANKER_MAX_CANDIDATES = 300
DEFAULT_RERANKER_FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"  # Best balance of speed and quality
DEFAULT_RERANKER_FLASHRANK_CACHE_DIR = None  # Use default cache directory
DEFAULT_RERANKER_FLASHRANK_CPU_MEM_ARENA = False  # Disable ONNX CPU memory arena to bound RSS

DEFAULT_EMBEDDINGS_COHERE_MODEL = "embed-english-v3.0"
DEFAULT_RERANKER_COHERE_MODEL = "rerank-english-v3.0"

# OpenRouter defaults
DEFAULT_EMBEDDINGS_OPENROUTER_MODEL = "perplexity/pplx-embed-v1-0.6b"
DEFAULT_RERANKER_OPENROUTER_MODEL = "cohere/rerank-v3.5"

DEFAULT_RERANKER_ZEROENTROPY_MODEL = "zerank-2"

DEFAULT_RERANKER_SILICONFLOW_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANKER_SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

DEFAULT_RERANKER_GOOGLE_MODEL = "semantic-ranker-default-004"

# Vector extension (pgvector, vchord, pgvectorscale, or AlloyDB ScaNN)
DEFAULT_VECTOR_EXTENSION = "pgvector"  # Options: "pgvector", "vchord", "pgvectorscale", "scann"

# Text search extension (native PostgreSQL, vchord BM25, or Timescale pg_textsearch)
DEFAULT_TEXT_SEARCH_EXTENSION = "native"  # Options: "native", "vchord", "pg_textsearch"

# LiteLLM defaults
DEFAULT_LITELLM_API_BASE = "http://localhost:4000"
DEFAULT_EMBEDDINGS_LITELLM_MODEL = "text-embedding-3-small"
DEFAULT_RERANKER_LITELLM_MODEL = "cohere/rerank-english-v3.0"
DEFAULT_RERANKER_LITELLM_MAX_TOKENS_PER_DOC: int | None = None

# LiteLLM SDK defaults
DEFAULT_EMBEDDINGS_LITELLM_SDK_MODEL = "cohere/embed-english-v3.0"
DEFAULT_EMBEDDINGS_LITELLM_SDK_ENCODING_FORMAT = "float"
DEFAULT_RERANKER_LITELLM_SDK_MODEL = "cohere/rerank-english-v3.0"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8888
DEFAULT_BASE_PATH = ""  # Empty string = root path
DEFAULT_LOG_LEVEL = "info"
DEFAULT_LOG_FORMAT = "text"  # Options: "text", "json"
DEFAULT_WORKERS = 1
DEFAULT_MCP_ENABLED = True
DEFAULT_MCP_ENABLED_TOOLS: list[str] | None = None  # None = all tools enabled
DEFAULT_MCP_STATELESS = False  # False = stateful (supports SSE/GET); True = stateless (POST-only)
DEFAULT_ENABLE_BANK_CONFIG_API = True
DEFAULT_DEFAULT_BANK_TEMPLATE: dict | None = None  # BankTemplateManifest dict applied to newly-created banks
DEFAULT_GRAPH_RETRIEVER = "link_expansion"
DEFAULT_RECALL_MAX_CONCURRENT = 32  # Max concurrent recall operations per worker
DEFAULT_RECALL_CONNECTION_BUDGET = 4  # Max concurrent DB connections per recall operation
DEFAULT_RECALL_MAX_QUERY_TOKENS = 500  # Maximum tokens allowed in recall query
DEFAULT_MENTAL_MODEL_REFRESH_CONCURRENCY = 8  # Max concurrent mental model refreshes
DEFAULT_LINK_EXPANSION_PER_ENTITY_LIMIT = 200  # Max target units per entity in graph expansion
DEFAULT_LINK_EXPANSION_TIMEOUT = 10.0  # Timeout (seconds) for entity expansion query

# Retain settings
DEFAULT_RETAIN_MAX_COMPLETION_TOKENS = 64000  # Max tokens for fact extraction LLM call
DEFAULT_RETAIN_CHUNK_SIZE = 3000  # Max chars per chunk for fact extraction
DEFAULT_RETAIN_EXTRACT_CAUSAL_LINKS = True  # Extract causal links between facts
DEFAULT_RETAIN_EXTRACTION_MODE = "concise"  # Extraction mode: "concise", "verbose", or "custom"
RETAIN_EXTRACTION_MODES = ("concise", "verbose", "custom", "verbatim", "chunks")  # Allowed extraction modes
DEFAULT_RETAIN_MISSION = None  # Declarative spec of what to retain (injected into any extraction mode)
DEFAULT_RETAIN_CUSTOM_INSTRUCTIONS = None  # Custom extraction guidelines (only used when mode="custom")
DEFAULT_RETAIN_DEFAULT_STRATEGY = None  # Default strategy name (None = no strategy override)
DEFAULT_RETAIN_STRATEGIES: dict | None = None  # Named retain strategies (dict of name → config overrides)
DEFAULT_RETAIN_CHUNK_BATCH_SIZE = (
    100  # Max chunks per streaming batch. Each chunk produces ~17 facts, so 100 chunks = ~1700 facts/batch.
)
DEFAULT_RETAIN_BATCH_TOKENS = 10_000  # ~40KB of text  # Max chars per sub-batch for async retain auto-splitting
DEFAULT_RETAIN_ENTITY_LOOKUP = "trigram"  # "full" or "trigram"
DEFAULT_RETAIN_BATCH_ENABLED = False  # Use LLM Batch API for fact extraction (only when async=True)
DEFAULT_RETAIN_BATCH_POLL_INTERVAL_SECONDS = 60  # Batch API polling interval in seconds

# File storage defaults
DEFAULT_FILE_STORAGE_TYPE = "native"  # PostgreSQL BYTEA storage
DEFAULT_FILE_PARSER = "markitdown"  # Default parser fallback chain (comma-separated, e.g. "iris,markitdown")
DEFAULT_FILE_PARSER_ALLOWLIST = None  # Allowlist of parsers clients may request (None = all registered parsers)
DEFAULT_FILE_CONVERSION_MAX_BATCH_SIZE_MB = 100  # Max total batch size in MB (all files combined)
DEFAULT_FILE_CONVERSION_MAX_BATCH_SIZE = 10  # Max files per batch upload
DEFAULT_ENABLE_FILE_UPLOAD_API = True  # Enable file upload endpoint
DEFAULT_FILE_DELETE_AFTER_RETAIN = True  # Delete file bytes after retain (saves storage)

# Observations defaults (consolidated knowledge from facts)
DEFAULT_ENABLE_OBSERVATIONS = True  # Observations enabled by default
DEFAULT_ENABLE_OBSERVATION_HISTORY = True  # Observation history tracking enabled by default
DEFAULT_ENABLE_MENTAL_MODEL_HISTORY = True  # Mental model history tracking enabled by default
DEFAULT_CONSOLIDATION_MAX_ATTEMPTS = 3  # Outer retry attempts for consolidation LLM batch calls
DEFAULT_CONSOLIDATION_BATCH_SIZE = 50  # Memories to load per batch (internal memory optimization)
DEFAULT_CONSOLIDATION_MAX_MEMORIES_PER_ROUND = (
    100  # Max memories per consolidation round (0 = unlimited). Limits how long one bank holds a worker slot.
)
DEFAULT_CONSOLIDATION_LLM_BATCH_SIZE = 8  # Facts per LLM call (1 = no batching; >1 = batch mode)
DEFAULT_CONSOLIDATION_MAX_TOKENS = 512  # Max tokens for recall when finding related observations
DEFAULT_CONSOLIDATION_RECALL_BUDGET = "low"  # Budget level for consolidation recall (low/mid/high)
DEFAULT_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS = (
    4096  # Total token budget for source facts in consolidation recall (-1 = unlimited)
)
DEFAULT_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS_PER_OBSERVATION = (
    256  # Max tokens of source facts per observation in consolidation prompt (-1 = unlimited)
)
DEFAULT_OBSERVATIONS_MISSION = None  # Declarative spec of what observations are for this bank
DEFAULT_MAX_OBSERVATIONS_PER_SCOPE = -1  # Max observations per tag scope (-1 = unlimited)

# Database migrations
DEFAULT_RUN_MIGRATIONS_ON_STARTUP = True

# Database connection pool
DEFAULT_DB_POOL_MIN_SIZE = 5
DEFAULT_DB_POOL_MAX_SIZE = 100
DEFAULT_DB_COMMAND_TIMEOUT = 60  # seconds
DEFAULT_DB_ACQUIRE_TIMEOUT = 30  # seconds
DEFAULT_DB_STATEMENT_TIMEOUT = 600  # seconds (Postgres statement_timeout applied on every pool connection; 0 disables)

# Worker configuration (distributed task processing)
DEFAULT_WORKER_ENABLED = True  # API runs worker by default (standalone mode)
DEFAULT_WORKER_ID = None  # Will use hostname if not specified
DEFAULT_WORKER_POLL_INTERVAL_MS = 500  # Poll database every 500ms
DEFAULT_WORKER_MAX_RETRIES = 3  # Max retries before marking task failed
DEFAULT_WORKER_HTTP_PORT = 8889  # HTTP port for worker metrics/health
DEFAULT_WORKER_MAX_SLOTS = 10  # Total concurrent tasks per worker
DEFAULT_RETAIN_MAX_CONCURRENT = 4  # Max concurrent retain DB phases (HNSW reads + writes). Limits I/O contention.

# Reflect agent settings
DEFAULT_REFLECT_MAX_ITERATIONS = 10  # Max tool call iterations before forcing response
DEFAULT_REFLECT_MAX_CONTEXT_TOKENS = 100_000  # Max accumulated context tokens before forcing final prompt
DEFAULT_REFLECT_WALL_TIMEOUT = 300  # Wall-clock timeout in seconds for the entire reflect operation (5 minutes)
DEFAULT_REFLECT_SOURCE_FACTS_MAX_TOKENS = -1  # Token budget for source facts in search_observations (-1 = disabled)
DEFAULT_RECALL_INCLUDE_CHUNKS = True  # Whether internal recall (e.g. mental model refresh) returns raw chunks
DEFAULT_RECALL_MAX_TOKENS = 2048  # Token budget for facts returned by internal recall
DEFAULT_RECALL_CHUNKS_MAX_TOKENS = 1000  # Token budget for raw chunks returned by internal recall

# Recall budget mapping
# "fixed": thinking_budget = recall_budget_fixed_<level> (preserves legacy behavior)
# "adaptive": thinking_budget = round(max_tokens * recall_budget_adaptive_<level>),
#             clamped to [recall_budget_min, recall_budget_max]
RECALL_BUDGET_FUNCTIONS = ("fixed", "adaptive")
DEFAULT_RECALL_BUDGET_FUNCTION = "fixed"
DEFAULT_RECALL_BUDGET_FIXED_LOW = 100
DEFAULT_RECALL_BUDGET_FIXED_MID = 300
DEFAULT_RECALL_BUDGET_FIXED_HIGH = 1000
# Adaptive defaults chosen to roughly match fixed defaults at max_tokens=4096
DEFAULT_RECALL_BUDGET_ADAPTIVE_LOW = 0.025
DEFAULT_RECALL_BUDGET_ADAPTIVE_MID = 0.075
DEFAULT_RECALL_BUDGET_ADAPTIVE_HIGH = 0.25
DEFAULT_RECALL_BUDGET_MIN = 20  # Floor for the adaptive function
DEFAULT_RECALL_BUDGET_MAX = 2000  # Ceiling for the adaptive function

# Disposition defaults (None = not set, fall back to bank DB value or 3)
DEFAULT_DISPOSITION_SKEPTICISM = None
DEFAULT_DISPOSITION_LITERALISM = None
DEFAULT_DISPOSITION_EMPATHY = None

# OpenTelemetry tracing configuration
DEFAULT_OTEL_TRACES_ENABLED = False  # Disabled by default for backward compatibility
DEFAULT_OTEL_SERVICE_NAME = "hindsight-api"
DEFAULT_OTEL_DEPLOYMENT_ENVIRONMENT = "development"
DEFAULT_METRICS_INCLUDE_BANK_ID = False  # Disabled by default to avoid high-cardinality OTel metric growth

# Audit log defaults
DEFAULT_AUDIT_LOG_ENABLED = False  # Disabled by default
DEFAULT_AUDIT_LOG_ACTIONS = ""  # Empty = audit all eligible actions
DEFAULT_AUDIT_LOG_RETENTION_DAYS = -1  # -1 = keep forever

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

# Webhook configuration defaults
DEFAULT_WEBHOOK_URL = None  # None = no global webhook configured
DEFAULT_WEBHOOK_SECRET = None  # None = no signing
DEFAULT_WEBHOOK_EVENT_TYPES = "consolidation.completed"  # Comma-separated; default = all supported events
DEFAULT_WEBHOOK_DELIVERY_POLL_INTERVAL_SECONDS = 30  # How often to poll for pending deliveries


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

    def __init__(self, allowed_fields: frozenset[str] | None = None):
        super().__init__()
        self._allowed_fields = allowed_fields

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "severity": self.SEVERITY_MAP.get(record.levelno, "DEFAULT"),
            "message": record.getMessage(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "logger": record.name,
        }

        # Lazy import to avoid circular dependency (engine imports from config).
        from hindsight_api.engine.memory_engine import _current_schema

        tenant = _current_schema.get()
        if tenant:
            log_entry["tenant"] = tenant

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        if self._allowed_fields is not None:
            log_entry = {k: v for k, v in log_entry.items() if k in self._allowed_fields}

        return json.dumps(log_entry)


def _parse_str_list(value: str) -> list[str]:
    """Parse a comma-separated string into a non-empty list of stripped tokens."""
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_positive_int(name: str, raw: str | None, default: int) -> int:
    """
    Parse an env var that must be a positive integer (>= 1).

    Falls back to ``default`` when unset/empty. Raises ValueError on non-integer
    or non-positive values so misconfiguration fails fast instead of triggering
    infinite loops or zero-step range() calls downstream.
    """
    if raw is None or raw == "":
        return default
    try:
        parsed = int(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from e
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1, got {parsed}")
    return parsed


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


def _validate_recall_budget_function(function: str) -> str:
    """Validate and normalize recall budget function."""
    function_lower = function.lower()
    if function_lower not in RECALL_BUDGET_FUNCTIONS:
        logger.warning(
            f"Invalid recall budget function '{function}', must be one of {RECALL_BUDGET_FUNCTIONS}. "
            f"Defaulting to '{DEFAULT_RECALL_BUDGET_FUNCTION}'."
        )
        return DEFAULT_RECALL_BUDGET_FUNCTION
    return function_lower


def _get_default_model_for_provider(provider: str) -> str:
    """Get the default model for a given provider."""
    return PROVIDER_DEFAULT_MODELS.get(provider.lower(), DEFAULT_LLM_MODEL)


def _parse_default_bank_template(raw: str | None) -> dict | None:
    """
    Parse HINDSIGHT_API_DEFAULT_BANK_TEMPLATE as JSON.

    The env var holds a BankTemplateManifest (JSON object) applied verbatim to
    every newly-created bank. Full Pydantic validation is deferred to bank
    creation time (to avoid pulling API models into config.py), but we fail
    fast here if the value is not valid JSON or not a JSON object.
    """
    if raw is None or raw.strip() == "":
        return DEFAULT_DEFAULT_BANK_TEMPLATE
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid {ENV_DEFAULT_BANK_TEMPLATE}: expected a JSON object, got invalid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise ValueError(f"Invalid {ENV_DEFAULT_BANK_TEMPLATE}: expected a JSON object, got {type(parsed).__name__}")
    return parsed


@dataclass
class HindsightConfig:
    """Configuration container for Hindsight API."""

    # Database
    database_backend: Literal["postgresql", "oracle"]
    database_url: str
    # Optional read-replica URL for recall queries. When set, the engine opens
    # a second pool and routes recall SELECTs through it.
    read_database_url: str | None
    read_db_pool_min_size: int
    read_db_pool_max_size: int
    migration_database_url: str | None
    database_schema: str
    vector_extension: str  # "pgvector", "vchord", "pgvectorscale", or "scann"
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
    llm_extra_body: (
        dict | None
    )  # Extra body params merged into OpenAI-compatible API calls (e.g. {"chat_template_kwargs": {"enable_thinking": true}})
    llm_default_headers: (
        dict | None
    )  # Custom headers passed as default_headers to provider SDK clients (e.g. {"X-Component-Id": "hindsight"} for proxies / request tracing)

    # Vertex AI configuration
    llm_vertexai_project_id: str | None
    llm_vertexai_region: str
    llm_vertexai_service_account_key: str | None

    # Gemini safety settings (None = use Gemini defaults; list of dicts with category/threshold)
    llm_gemini_safety_settings: list | None

    # Built-in llama.cpp configuration (for provider=llamacpp)
    llamacpp_model_path: str | None  # Path to GGUF file (None = auto-download default)
    llamacpp_gpu_layers: int  # -1 = all layers on GPU, 0 = CPU only
    llamacpp_context_size: int  # Context window size
    llamacpp_chat_format: str | None  # Chat template format (None = auto-detect from GGUF)
    llamacpp_no_grammar: bool  # Disable JSON grammar enforcement (faster, less reliable)
    llamacpp_extra_args: str | None  # Space-separated extra CLI args for llama.cpp server

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
    embeddings_cohere_output_dimensions: int | None
    embeddings_openrouter_api_key: str | None
    embeddings_openrouter_model: str
    embeddings_litellm_api_base: str
    embeddings_litellm_api_key: str | None
    embeddings_litellm_model: str
    embeddings_litellm_sdk_api_key: str | None
    embeddings_litellm_sdk_model: str
    embeddings_litellm_sdk_api_base: str | None
    embeddings_litellm_sdk_output_dimensions: int | None
    embeddings_litellm_sdk_encoding_format: str | None
    # Gemini/Vertex AI embeddings
    embeddings_gemini_api_key: str | None
    embeddings_gemini_model: str
    embeddings_gemini_output_dimensionality: int | None
    embeddings_gemini_force_ipv4: bool
    embeddings_vertexai_project_id: str | None
    embeddings_vertexai_region: str | None
    embeddings_vertexai_service_account_key: str | None

    # Reranker
    reranker_provider: str
    reranker_local_model: str
    reranker_local_force_cpu: bool
    reranker_local_max_concurrent: int
    reranker_local_trust_remote_code: bool
    reranker_local_fp16: bool
    reranker_local_bucket_batching: bool
    reranker_local_batch_size: int
    reranker_tei_url: str | None
    reranker_tei_batch_size: int
    reranker_tei_max_concurrent: int
    reranker_tei_http_timeout: float
    reranker_max_candidates: int
    reranker_cohere_api_key: str | None
    reranker_cohere_model: str
    reranker_cohere_base_url: str | None
    reranker_openrouter_api_key: str | None
    reranker_openrouter_model: str
    reranker_litellm_api_base: str
    reranker_litellm_api_key: str | None
    reranker_litellm_model: str
    reranker_litellm_max_tokens_per_doc: int | None
    reranker_litellm_sdk_api_key: str | None
    reranker_litellm_sdk_model: str
    reranker_litellm_sdk_api_base: str | None
    reranker_zeroentropy_api_key: str | None
    reranker_zeroentropy_model: str
    reranker_zeroentropy_base_url: str | None
    reranker_siliconflow_api_key: str | None
    reranker_siliconflow_model: str
    reranker_siliconflow_base_url: str
    reranker_google_model: str
    reranker_google_project_id: str | None
    reranker_google_service_account_key: str | None

    # Server
    host: str
    port: int
    base_path: str
    log_level: str
    log_format: str
    log_json_fields: list[str] | None  # None = all fields; explicit list = allowlist
    mcp_enabled: bool
    mcp_enabled_tools: list[str] | None  # None = all tools; explicit list = allowlist
    mcp_stateless: bool  # True = stateless HTTP (POST-only); False = stateful (supports GET/SSE)
    enable_bank_config_api: bool
    # Default bank template (static, server-level only). When set, the manifest is applied
    # to every newly-created bank, overriding the env/config defaults for any fields it sets.
    default_bank_template: dict | None

    # Recall
    graph_retriever: str
    recall_max_concurrent: int
    recall_connection_budget: int
    recall_max_query_tokens: int
    mental_model_refresh_concurrency: int
    link_expansion_per_entity_limit: int
    link_expansion_timeout: float

    # Retain settings
    retain_max_completion_tokens: int
    retain_chunk_size: int
    retain_extract_causal_links: bool
    retain_extraction_mode: str
    retain_mission: str | None
    retain_custom_instructions: str | None
    retain_default_strategy: str | None
    retain_strategies: dict | None
    retain_batch_tokens: int
    retain_batch_enabled: bool
    retain_batch_poll_interval_seconds: int
    retain_entity_lookup: str  # "full" or "trigram"
    retain_chunk_batch_size: int  # Max chunks per streaming batch (0 = disabled)

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
    file_parser: list[str]  # Ordered fallback chain of parsers (e.g. ["iris", "markitdown"])
    file_parser_allowlist: list[str] | None  # Parsers clients may request (None = all registered)
    file_parser_iris_token: str | None  # Vectorize API token for iris parser (VECTORIZE_TOKEN)
    file_parser_iris_org_id: str | None  # Vectorize org ID for iris parser (VECTORIZE_ORG_ID)
    file_parser_llama_parse_api_key: str | None  # LlamaCloud API key for llama_parse parser
    file_conversion_max_batch_size_mb: int  # Max total batch size in MB (all files combined)
    file_conversion_max_batch_size: int  # Max files per request
    enable_file_upload_api: bool
    file_delete_after_retain: bool

    # Observations settings (consolidated knowledge from facts)
    enable_observations: bool
    enable_observation_history: bool
    enable_mental_model_history: bool
    consolidation_batch_size: int
    consolidation_max_memories_per_round: int
    consolidation_llm_batch_size: int
    consolidation_max_tokens: int
    consolidation_recall_budget: str
    consolidation_source_facts_max_tokens: int
    consolidation_source_facts_max_tokens_per_observation: int
    consolidation_max_attempts: int
    observations_mission: str | None
    max_observations_per_scope: int

    # Entity labels (controlled vocabulary of key:value classification labels extracted at retain time)
    # List of label group dicts: [{key, description, type, optional, values: [{value, description}]}]
    entity_labels: list | None
    # Whether to extract regular named entities alongside entity labels (default: True)
    # When False: only label entities are extracted (or no entities at all if no labels configured)
    entities_allow_free_form: bool

    # Reflect agent settings
    reflect_mission: str | None
    reflect_source_facts_max_tokens: int

    # Recall settings (used by internal recall, e.g. during mental model refresh)
    recall_include_chunks: bool
    recall_max_tokens: int
    recall_chunks_max_tokens: int

    # Recall budget mapping: how the Budget enum (LOW/MID/HIGH) maps to thinking_budget integer.
    # function="fixed": use the recall_budget_fixed_* values directly (legacy behavior).
    # function="adaptive": compute round(max_tokens * recall_budget_adaptive_*),
    #                      clamped to [recall_budget_min, recall_budget_max].
    recall_budget_function: str
    recall_budget_fixed_low: int
    recall_budget_fixed_mid: int
    recall_budget_fixed_high: int
    recall_budget_adaptive_low: float
    recall_budget_adaptive_mid: float
    recall_budget_adaptive_high: float
    recall_budget_min: int
    recall_budget_max: int

    # Disposition settings (hierarchical - can be overridden per bank; None = fall back to DB)
    disposition_skepticism: int | None
    disposition_literalism: int | None
    disposition_empathy: int | None

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
    db_statement_timeout: int

    # Worker configuration (distributed task processing)
    worker_enabled: bool
    worker_id: str | None
    worker_poll_interval_ms: int
    worker_max_retries: int
    worker_http_port: int
    worker_max_slots: int
    worker_slot_reservations: dict[str, int]
    retain_max_concurrent: int

    # Reflect agent settings
    reflect_max_iterations: int
    reflect_max_context_tokens: int
    reflect_wall_timeout: int

    # OpenTelemetry tracing configuration
    otel_traces_enabled: bool
    otel_exporter_otlp_endpoint: str | None
    otel_exporter_otlp_headers: str | None
    otel_service_name: str
    otel_deployment_environment: str
    metrics_include_bank_id: bool

    # Audit log configuration (static - server-level only)
    audit_log_enabled: bool  # Master switch for audit logging
    audit_log_actions: list[str]  # Allowlist of action types (empty = all)
    audit_log_retention_days: int  # -1 = keep forever, >0 = delete after N days

    # Webhook configuration (static - server-level only, not per-bank)
    webhook_url: str | None  # Global webhook URL (None = disabled)
    webhook_secret: str | None  # HMAC signing secret (None = unsigned)
    webhook_event_types: list[str]  # Event types to deliver globally
    webhook_delivery_poll_interval_seconds: int  # How often the delivery worker polls

    # Defaulted fields (source-compatible additions — existing direct constructor callers keep working).
    # Keep at the end of the dataclass; Python forbids non-default fields after default fields.
    embeddings_openai_batch_size: int = DEFAULT_EMBEDDINGS_OPENAI_BATCH_SIZE

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
        "reranker_zeroentropy_base_url",
        "reranker_siliconflow_base_url",
        # Service Account Keys
        "llm_vertexai_service_account_key",
        "embeddings_vertexai_service_account_key",
        "reranker_google_service_account_key",
        # Embeddings API keys
        "embeddings_gemini_api_key",
        # File storage credentials
        "file_storage_s3_access_key_id",
        "file_storage_s3_secret_access_key",
        "file_storage_gcs_service_account_key",
        "file_storage_azure_account_key",
        # File parser credentials
        "file_parser_iris_token",
        "file_parser_llama_parse_api_key",
    }

    # CONFIGURABLE_FIELDS: Safe behavioral settings that can be customized per-tenant/bank
    # These fields are manually tagged as safe to expose and modify.
    # Excludes credentials, infrastructure config, provider/model selection, and performance tuning.
    _CONFIGURABLE_FIELDS = {
        # MCP tool access control
        "mcp_enabled_tools",
        # Retention settings (behavioral)
        "retain_chunk_size",
        "retain_extraction_mode",
        "retain_mission",
        "retain_custom_instructions",
        "retain_default_strategy",
        "retain_strategies",
        "retain_chunk_batch_size",
        # Entity labels (controlled vocabulary for entity classification)
        "entity_labels",
        "entities_allow_free_form",
        # Consolidation settings
        "enable_observations",
        "consolidation_llm_batch_size",
        "consolidation_max_memories_per_round",
        "consolidation_source_facts_max_tokens",
        "consolidation_source_facts_max_tokens_per_observation",
        "observations_mission",
        "max_observations_per_scope",
        # Reflect settings
        "reflect_mission",
        "reflect_source_facts_max_tokens",
        # Recall settings (used by internal recall, e.g. mental model refresh)
        "recall_include_chunks",
        "recall_max_tokens",
        "recall_chunks_max_tokens",
        # Recall budget mapping (Budget enum -> thinking_budget integer)
        "recall_budget_function",
        "recall_budget_fixed_low",
        "recall_budget_fixed_mid",
        "recall_budget_fixed_high",
        "recall_budget_adaptive_low",
        "recall_budget_adaptive_mid",
        "recall_budget_adaptive_high",
        "recall_budget_min",
        "recall_budget_max",
        # Disposition settings
        "disposition_skepticism",
        "disposition_literalism",
        "disposition_empathy",
        # Gemini safety settings (controls content filtering for Gemini/VertexAI providers)
        "llm_gemini_safety_settings",
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
        validate_extension(self.vector_extension)

        # Validate text_search_extension
        valid_text_search = ("native", "vchord", "pg_textsearch")
        if self.text_search_extension not in valid_text_search:
            raise ValueError(
                f"Invalid text_search_extension: {self.text_search_extension}. Must be one of: {', '.join(valid_text_search)}"
            )

        # When LLM provider is "none", force chunks-only mode and disable LLM-dependent features
        if self.llm_provider == "none":
            self.retain_extraction_mode = "chunks"
            self.enable_observations = False
            logger.info(
                "LLM provider set to 'none': forcing retain_extraction_mode='chunks', "
                "disabling observations/consolidation. Reflect will return HTTP 400."
            )

        # RETAIN_MAX_COMPLETION_TOKENS must be greater than RETAIN_CHUNK_SIZE
        # to ensure the LLM has enough output capacity to extract facts from chunks
        # (not applicable when provider is "none" since no LLM calls are made)
        if self.llm_provider != "none" and self.retain_max_completion_tokens <= self.retain_chunk_size:
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

        # Warn if local ML dependencies are missing when configured.
        # Don't hard-fail here — the actual ImportError fires at model init time
        # with a clear message. This early warning catches it before startup proceeds.
        if self.embeddings_provider == "local" or self.reranker_provider == "local":
            try:
                import importlib

                importlib.import_module("sentence_transformers")
            except ImportError:
                missing = []
                if self.embeddings_provider == "local":
                    missing.append("embeddings")
                if self.reranker_provider == "local":
                    missing.append("reranker")
                logger.warning(
                    "Local ML provider configured for %s, but 'sentence-transformers' "
                    "is not installed. The API will fail at startup. Either:\n"
                    "  1. Install local ML deps: pip install hindsight-api[local-ml]\n"
                    "  2. Use a remote provider instead:\n"
                    "     HINDSIGHT_API_EMBEDDINGS_PROVIDER=openai (or gemini, tei)\n"
                    "     HINDSIGHT_API_RERANKER_PROVIDER=none (or tei)",
                    " and ".join(missing),
                )

        # Validate that sum of per-operation slot reservations does not exceed max_slots
        total_reserved = sum(self.worker_slot_reservations.values())
        if total_reserved > self.worker_max_slots:
            reservation_details = ", ".join(f"{k}={v}" for k, v in self.worker_slot_reservations.items() if v > 0)
            raise ValueError(
                f"Sum of per-operation slot reservations ({total_reserved}: {reservation_details}) "
                f"exceeds worker_max_slots ({self.worker_max_slots}). "
                f"Reduce reservations or increase HINDSIGHT_API_WORKER_MAX_SLOTS."
            )

    @classmethod
    def from_env(cls) -> "HindsightConfig":
        """Create configuration from environment variables."""
        # Get provider first to determine default model
        llm_provider = os.getenv(ENV_LLM_PROVIDER, DEFAULT_LLM_PROVIDER)
        llm_model = os.getenv(ENV_LLM_MODEL) or _get_default_model_for_provider(llm_provider)

        config = cls(
            # Database
            database_backend=os.getenv(ENV_DATABASE_BACKEND, DEFAULT_DATABASE_BACKEND).lower(),
            database_url=os.getenv(ENV_DATABASE_URL, DEFAULT_DATABASE_URL),
            read_database_url=os.getenv(ENV_READ_DATABASE_URL) or None,
            read_db_pool_min_size=int(os.getenv(ENV_READ_DB_POOL_MIN_SIZE, str(DEFAULT_DB_POOL_MIN_SIZE))),
            read_db_pool_max_size=int(os.getenv(ENV_READ_DB_POOL_MAX_SIZE, str(DEFAULT_DB_POOL_MAX_SIZE))),
            migration_database_url=os.getenv(ENV_MIGRATION_DATABASE_URL) or None,
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
            llm_extra_body=json.loads(os.getenv(ENV_LLM_EXTRA_BODY, "null")),
            llm_default_headers=json.loads(os.getenv(ENV_LLM_DEFAULT_HEADERS, "null")),
            # Vertex AI
            llm_vertexai_project_id=os.getenv(ENV_LLM_VERTEXAI_PROJECT_ID) or DEFAULT_LLM_VERTEXAI_PROJECT_ID,
            llm_vertexai_region=os.getenv(ENV_LLM_VERTEXAI_REGION, DEFAULT_LLM_VERTEXAI_REGION),
            llm_vertexai_service_account_key=os.getenv(ENV_LLM_VERTEXAI_SERVICE_ACCOUNT_KEY)
            or DEFAULT_LLM_VERTEXAI_SERVICE_ACCOUNT_KEY,
            # Gemini safety settings (JSON-encoded list of {category, threshold} dicts)
            llm_gemini_safety_settings=json.loads(os.getenv(ENV_LLM_GEMINI_SAFETY_SETTINGS, "null")),
            # Built-in llama.cpp configuration
            llamacpp_model_path=os.getenv(ENV_LLAMACPP_MODEL_PATH) or None,
            llamacpp_gpu_layers=int(os.getenv(ENV_LLAMACPP_GPU_LAYERS, str(DEFAULT_LLAMACPP_GPU_LAYERS))),
            llamacpp_context_size=int(os.getenv(ENV_LLAMACPP_CONTEXT_SIZE, str(DEFAULT_LLAMACPP_CONTEXT_SIZE))),
            llamacpp_chat_format=os.getenv(ENV_LLAMACPP_CHAT_FORMAT) or DEFAULT_LLAMACPP_CHAT_FORMAT,
            llamacpp_no_grammar=os.getenv(ENV_LLAMACPP_NO_GRAMMAR, str(DEFAULT_LLAMACPP_NO_GRAMMAR)).lower()
            in ("true", "1"),
            llamacpp_extra_args=os.getenv(ENV_LLAMACPP_EXTRA_ARGS) or DEFAULT_LLAMACPP_EXTRA_ARGS,
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
            embeddings_openai_batch_size=_parse_positive_int(
                ENV_EMBEDDINGS_OPENAI_BATCH_SIZE,
                os.getenv(ENV_EMBEDDINGS_OPENAI_BATCH_SIZE),
                DEFAULT_EMBEDDINGS_OPENAI_BATCH_SIZE,
            ),
            # Cohere embeddings (with backward-compatible fallback to shared API key)
            embeddings_cohere_api_key=os.getenv(ENV_EMBEDDINGS_COHERE_API_KEY) or os.getenv(ENV_COHERE_API_KEY),
            embeddings_cohere_model=os.getenv(ENV_EMBEDDINGS_COHERE_MODEL, DEFAULT_EMBEDDINGS_COHERE_MODEL),
            embeddings_cohere_base_url=os.getenv(ENV_EMBEDDINGS_COHERE_BASE_URL) or None,
            embeddings_cohere_output_dimensions=int(v)
            if (v := os.getenv(ENV_EMBEDDINGS_COHERE_OUTPUT_DIMENSIONS))
            else None,
            # OpenRouter embeddings (with fallback to shared OpenRouter key, then LLM key)
            embeddings_openrouter_api_key=os.getenv(ENV_EMBEDDINGS_OPENROUTER_API_KEY)
            or os.getenv(ENV_OPENROUTER_API_KEY)
            or os.getenv(ENV_LLM_API_KEY),
            embeddings_openrouter_model=os.getenv(ENV_EMBEDDINGS_OPENROUTER_MODEL, DEFAULT_EMBEDDINGS_OPENROUTER_MODEL),
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
            embeddings_litellm_sdk_output_dimensions=int(v)
            if (v := os.getenv(ENV_EMBEDDINGS_LITELLM_SDK_OUTPUT_DIMENSIONS))
            else None,
            embeddings_litellm_sdk_encoding_format=os.getenv(
                ENV_EMBEDDINGS_LITELLM_SDK_ENCODING_FORMAT, DEFAULT_EMBEDDINGS_LITELLM_SDK_ENCODING_FORMAT
            ),
            # Gemini/Vertex AI embeddings (with fallback to LLM keys)
            embeddings_gemini_api_key=os.getenv(ENV_EMBEDDINGS_GEMINI_API_KEY) or os.getenv(ENV_LLM_API_KEY),
            embeddings_gemini_model=os.getenv(ENV_EMBEDDINGS_GEMINI_MODEL, DEFAULT_EMBEDDINGS_GEMINI_MODEL),
            embeddings_gemini_output_dimensionality=int(
                os.getenv(
                    ENV_EMBEDDINGS_GEMINI_OUTPUT_DIMENSIONALITY,
                    str(DEFAULT_EMBEDDINGS_GEMINI_OUTPUT_DIMENSIONALITY),
                )
            ),
            embeddings_gemini_force_ipv4=os.getenv(
                ENV_EMBEDDINGS_GEMINI_FORCE_IPV4,
                str(DEFAULT_EMBEDDINGS_GEMINI_FORCE_IPV4),
            ).lower()
            in ("true", "1"),
            embeddings_vertexai_project_id=os.getenv(ENV_EMBEDDINGS_VERTEXAI_PROJECT_ID)
            or os.getenv(ENV_LLM_VERTEXAI_PROJECT_ID),
            embeddings_vertexai_region=os.getenv(ENV_EMBEDDINGS_VERTEXAI_REGION) or os.getenv(ENV_LLM_VERTEXAI_REGION),
            embeddings_vertexai_service_account_key=os.getenv(ENV_EMBEDDINGS_VERTEXAI_SERVICE_ACCOUNT_KEY)
            or os.getenv(ENV_LLM_VERTEXAI_SERVICE_ACCOUNT_KEY),
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
            reranker_local_fp16=os.getenv(ENV_RERANKER_LOCAL_FP16, str(DEFAULT_RERANKER_LOCAL_FP16)).lower()
            in ("true", "1"),
            reranker_local_bucket_batching=os.getenv(
                ENV_RERANKER_LOCAL_BUCKET_BATCHING, str(DEFAULT_RERANKER_LOCAL_BUCKET_BATCHING)
            ).lower()
            in ("true", "1"),
            reranker_local_batch_size=int(
                os.getenv(ENV_RERANKER_LOCAL_BATCH_SIZE, str(DEFAULT_RERANKER_LOCAL_BATCH_SIZE))
            ),
            reranker_tei_url=os.getenv(ENV_RERANKER_TEI_URL),
            reranker_tei_batch_size=int(os.getenv(ENV_RERANKER_TEI_BATCH_SIZE, str(DEFAULT_RERANKER_TEI_BATCH_SIZE))),
            reranker_tei_max_concurrent=int(
                os.getenv(ENV_RERANKER_TEI_MAX_CONCURRENT, str(DEFAULT_RERANKER_TEI_MAX_CONCURRENT))
            ),
            reranker_tei_http_timeout=float(
                os.getenv(ENV_RERANKER_TEI_HTTP_TIMEOUT, str(DEFAULT_RERANKER_TEI_HTTP_TIMEOUT))
            ),
            reranker_max_candidates=int(os.getenv(ENV_RERANKER_MAX_CANDIDATES, str(DEFAULT_RERANKER_MAX_CANDIDATES))),
            # Cohere reranker (with backward-compatible fallback to shared API key)
            reranker_cohere_api_key=os.getenv(ENV_RERANKER_COHERE_API_KEY) or os.getenv(ENV_COHERE_API_KEY),
            reranker_cohere_model=os.getenv(ENV_RERANKER_COHERE_MODEL, DEFAULT_RERANKER_COHERE_MODEL),
            reranker_cohere_base_url=os.getenv(ENV_RERANKER_COHERE_BASE_URL) or None,
            # OpenRouter reranker (with fallback to shared OpenRouter key, then LLM key)
            reranker_openrouter_api_key=os.getenv(ENV_RERANKER_OPENROUTER_API_KEY)
            or os.getenv(ENV_OPENROUTER_API_KEY)
            or os.getenv(ENV_LLM_API_KEY),
            reranker_openrouter_model=os.getenv(ENV_RERANKER_OPENROUTER_MODEL, DEFAULT_RERANKER_OPENROUTER_MODEL),
            # LiteLLM reranker (with backward-compatible fallback to shared config)
            reranker_litellm_api_base=os.getenv(ENV_RERANKER_LITELLM_API_BASE)
            or os.getenv(ENV_LITELLM_API_BASE, DEFAULT_LITELLM_API_BASE),
            reranker_litellm_api_key=os.getenv(ENV_RERANKER_LITELLM_API_KEY) or os.getenv(ENV_LITELLM_API_KEY),
            reranker_litellm_model=os.getenv(ENV_RERANKER_LITELLM_MODEL, DEFAULT_RERANKER_LITELLM_MODEL),
            reranker_litellm_max_tokens_per_doc=int(v)
            if (v := os.getenv(ENV_RERANKER_LITELLM_MAX_TOKENS_PER_DOC))
            else DEFAULT_RERANKER_LITELLM_MAX_TOKENS_PER_DOC,
            # LiteLLM SDK reranker (direct API access)
            reranker_litellm_sdk_api_key=os.getenv(ENV_RERANKER_LITELLM_SDK_API_KEY),
            reranker_litellm_sdk_model=os.getenv(ENV_RERANKER_LITELLM_SDK_MODEL, DEFAULT_RERANKER_LITELLM_SDK_MODEL),
            reranker_litellm_sdk_api_base=os.getenv(ENV_RERANKER_LITELLM_SDK_API_BASE) or None,
            # ZeroEntropy reranker
            reranker_zeroentropy_api_key=os.getenv(ENV_RERANKER_ZEROENTROPY_API_KEY),
            reranker_zeroentropy_model=os.getenv(ENV_RERANKER_ZEROENTROPY_MODEL, DEFAULT_RERANKER_ZEROENTROPY_MODEL),
            reranker_zeroentropy_base_url=os.getenv(ENV_RERANKER_ZEROENTROPY_BASE_URL) or None,
            # SiliconFlow reranker (Cohere-compatible /rerank endpoint)
            reranker_siliconflow_api_key=os.getenv(ENV_RERANKER_SILICONFLOW_API_KEY),
            reranker_siliconflow_model=os.getenv(ENV_RERANKER_SILICONFLOW_MODEL, DEFAULT_RERANKER_SILICONFLOW_MODEL),
            reranker_siliconflow_base_url=os.getenv(
                ENV_RERANKER_SILICONFLOW_BASE_URL, DEFAULT_RERANKER_SILICONFLOW_BASE_URL
            ),
            # Google Discovery Engine reranker (with fallback to LLM Vertex AI keys)
            reranker_google_model=os.getenv(ENV_RERANKER_GOOGLE_MODEL, DEFAULT_RERANKER_GOOGLE_MODEL),
            reranker_google_project_id=os.getenv(ENV_RERANKER_GOOGLE_PROJECT_ID)
            or os.getenv(ENV_LLM_VERTEXAI_PROJECT_ID),
            reranker_google_service_account_key=os.getenv(ENV_RERANKER_GOOGLE_SERVICE_ACCOUNT_KEY)
            or os.getenv(ENV_LLM_VERTEXAI_SERVICE_ACCOUNT_KEY),
            # Server
            host=os.getenv(ENV_HOST, DEFAULT_HOST),
            port=int(os.getenv(ENV_PORT, DEFAULT_PORT)),
            base_path=os.getenv(ENV_BASE_PATH, DEFAULT_BASE_PATH),
            log_level=os.getenv(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL),
            log_format=os.getenv(ENV_LOG_FORMAT, DEFAULT_LOG_FORMAT).lower(),
            log_json_fields=_parse_str_list(os.getenv(ENV_LOG_JSON_FIELDS, "")) or None,
            mcp_enabled=os.getenv(ENV_MCP_ENABLED, str(DEFAULT_MCP_ENABLED)).lower() == "true",
            mcp_enabled_tools=[t.strip() for t in os.getenv(ENV_MCP_ENABLED_TOOLS).split(",") if t.strip()]
            if os.getenv(ENV_MCP_ENABLED_TOOLS)
            else DEFAULT_MCP_ENABLED_TOOLS,
            mcp_stateless=os.getenv(ENV_MCP_STATELESS, str(DEFAULT_MCP_STATELESS)).lower() == "true",
            enable_bank_config_api=os.getenv(ENV_ENABLE_BANK_CONFIG_API, str(DEFAULT_ENABLE_BANK_CONFIG_API)).lower()
            == "true",
            default_bank_template=_parse_default_bank_template(os.getenv(ENV_DEFAULT_BANK_TEMPLATE)),
            # Recall
            graph_retriever=os.getenv(ENV_GRAPH_RETRIEVER, DEFAULT_GRAPH_RETRIEVER),
            recall_max_concurrent=int(os.getenv(ENV_RECALL_MAX_CONCURRENT, str(DEFAULT_RECALL_MAX_CONCURRENT))),
            recall_connection_budget=int(
                os.getenv(ENV_RECALL_CONNECTION_BUDGET, str(DEFAULT_RECALL_CONNECTION_BUDGET))
            ),
            recall_max_query_tokens=int(os.getenv(ENV_RECALL_MAX_QUERY_TOKENS, str(DEFAULT_RECALL_MAX_QUERY_TOKENS))),
            mental_model_refresh_concurrency=int(
                os.getenv(ENV_MENTAL_MODEL_REFRESH_CONCURRENCY, str(DEFAULT_MENTAL_MODEL_REFRESH_CONCURRENCY))
            ),
            link_expansion_per_entity_limit=int(
                os.getenv(ENV_LINK_EXPANSION_PER_ENTITY_LIMIT, str(DEFAULT_LINK_EXPANSION_PER_ENTITY_LIMIT))
            ),
            link_expansion_timeout=float(os.getenv(ENV_LINK_EXPANSION_TIMEOUT, str(DEFAULT_LINK_EXPANSION_TIMEOUT))),
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
            retain_mission=os.getenv(ENV_RETAIN_MISSION) or DEFAULT_RETAIN_MISSION,
            retain_custom_instructions=os.getenv(ENV_RETAIN_CUSTOM_INSTRUCTIONS) or DEFAULT_RETAIN_CUSTOM_INSTRUCTIONS,
            retain_default_strategy=os.getenv(ENV_RETAIN_DEFAULT_STRATEGY) or DEFAULT_RETAIN_DEFAULT_STRATEGY,
            retain_strategies=DEFAULT_RETAIN_STRATEGIES,
            retain_batch_tokens=int(os.getenv(ENV_RETAIN_BATCH_TOKENS, str(DEFAULT_RETAIN_BATCH_TOKENS))),
            retain_entity_lookup=os.getenv(ENV_RETAIN_ENTITY_LOOKUP, DEFAULT_RETAIN_ENTITY_LOOKUP),
            retain_batch_enabled=os.getenv(ENV_RETAIN_BATCH_ENABLED, str(DEFAULT_RETAIN_BATCH_ENABLED)).lower()
            == "true",
            retain_batch_poll_interval_seconds=int(
                os.getenv(ENV_RETAIN_BATCH_POLL_INTERVAL_SECONDS, str(DEFAULT_RETAIN_BATCH_POLL_INTERVAL_SECONDS))
            ),
            retain_chunk_batch_size=int(os.getenv(ENV_RETAIN_CHUNK_BATCH_SIZE, str(DEFAULT_RETAIN_CHUNK_BATCH_SIZE))),
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
            file_parser=_parse_str_list(os.getenv(ENV_FILE_PARSER, DEFAULT_FILE_PARSER)),
            file_parser_allowlist=_parse_str_list(os.getenv(ENV_FILE_PARSER_ALLOWLIST))
            if os.getenv(ENV_FILE_PARSER_ALLOWLIST)
            else None,
            file_parser_iris_token=os.getenv(ENV_FILE_PARSER_IRIS_TOKEN) or None,
            file_parser_iris_org_id=os.getenv(ENV_FILE_PARSER_IRIS_ORG_ID) or None,
            file_parser_llama_parse_api_key=os.getenv(ENV_FILE_PARSER_LLAMA_PARSE_API_KEY) or None,
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
            enable_observation_history=os.getenv(
                ENV_ENABLE_OBSERVATION_HISTORY, str(DEFAULT_ENABLE_OBSERVATION_HISTORY)
            ).lower()
            == "true",
            enable_mental_model_history=os.getenv(
                ENV_ENABLE_MENTAL_MODEL_HISTORY, str(DEFAULT_ENABLE_MENTAL_MODEL_HISTORY)
            ).lower()
            == "true",
            consolidation_batch_size=int(
                os.getenv(ENV_CONSOLIDATION_BATCH_SIZE, str(DEFAULT_CONSOLIDATION_BATCH_SIZE))
            ),
            consolidation_max_memories_per_round=int(
                os.getenv(
                    ENV_CONSOLIDATION_MAX_MEMORIES_PER_ROUND,
                    str(DEFAULT_CONSOLIDATION_MAX_MEMORIES_PER_ROUND),
                )
            ),
            consolidation_llm_batch_size=int(
                os.getenv(ENV_CONSOLIDATION_LLM_BATCH_SIZE, str(DEFAULT_CONSOLIDATION_LLM_BATCH_SIZE))
            ),
            consolidation_max_tokens=int(
                os.getenv(ENV_CONSOLIDATION_MAX_TOKENS, str(DEFAULT_CONSOLIDATION_MAX_TOKENS))
            ),
            consolidation_recall_budget=os.getenv(ENV_CONSOLIDATION_RECALL_BUDGET, DEFAULT_CONSOLIDATION_RECALL_BUDGET),
            consolidation_source_facts_max_tokens=int(
                os.getenv(ENV_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS, str(DEFAULT_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS))
            ),
            consolidation_source_facts_max_tokens_per_observation=int(
                os.getenv(
                    ENV_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS_PER_OBSERVATION,
                    str(DEFAULT_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS_PER_OBSERVATION),
                )
            ),
            consolidation_max_attempts=int(
                os.getenv(ENV_CONSOLIDATION_MAX_ATTEMPTS, str(DEFAULT_CONSOLIDATION_MAX_ATTEMPTS))
            ),
            observations_mission=os.getenv(ENV_OBSERVATIONS_MISSION) or DEFAULT_OBSERVATIONS_MISSION,
            max_observations_per_scope=int(
                os.getenv(ENV_MAX_OBSERVATIONS_PER_SCOPE, str(DEFAULT_MAX_OBSERVATIONS_PER_SCOPE))
            ),
            entity_labels=None,
            entities_allow_free_form=True,
            # Database migrations
            run_migrations_on_startup=os.getenv(ENV_RUN_MIGRATIONS_ON_STARTUP, "true").lower() == "true",
            # Database connection pool
            db_pool_min_size=int(os.getenv(ENV_DB_POOL_MIN_SIZE, str(DEFAULT_DB_POOL_MIN_SIZE))),
            db_pool_max_size=int(os.getenv(ENV_DB_POOL_MAX_SIZE, str(DEFAULT_DB_POOL_MAX_SIZE))),
            db_command_timeout=int(os.getenv(ENV_DB_COMMAND_TIMEOUT, str(DEFAULT_DB_COMMAND_TIMEOUT))),
            db_acquire_timeout=int(os.getenv(ENV_DB_ACQUIRE_TIMEOUT, str(DEFAULT_DB_ACQUIRE_TIMEOUT))),
            db_statement_timeout=int(os.getenv(ENV_DB_STATEMENT_TIMEOUT, str(DEFAULT_DB_STATEMENT_TIMEOUT))),
            # Worker configuration
            worker_enabled=os.getenv(ENV_WORKER_ENABLED, str(DEFAULT_WORKER_ENABLED)).lower() == "true",
            worker_id=os.getenv(ENV_WORKER_ID) or DEFAULT_WORKER_ID,
            worker_poll_interval_ms=int(os.getenv(ENV_WORKER_POLL_INTERVAL_MS, str(DEFAULT_WORKER_POLL_INTERVAL_MS))),
            worker_max_retries=int(os.getenv(ENV_WORKER_MAX_RETRIES, str(DEFAULT_WORKER_MAX_RETRIES))),
            worker_http_port=int(os.getenv(ENV_WORKER_HTTP_PORT, str(DEFAULT_WORKER_HTTP_PORT))),
            worker_max_slots=int(os.getenv(ENV_WORKER_MAX_SLOTS, str(DEFAULT_WORKER_MAX_SLOTS))),
            worker_slot_reservations={
                op_type: int(os.getenv(env_var, str(default)))
                for op_type, (env_var, default) in WORKER_SLOT_RESERVATION_TYPES.items()
                if int(os.getenv(env_var, str(default))) > 0
            },
            retain_max_concurrent=int(os.getenv(ENV_RETAIN_MAX_CONCURRENT, str(DEFAULT_RETAIN_MAX_CONCURRENT))),
            # Reflect agent settings
            reflect_max_iterations=int(os.getenv(ENV_REFLECT_MAX_ITERATIONS, str(DEFAULT_REFLECT_MAX_ITERATIONS))),
            reflect_max_context_tokens=int(
                os.getenv(ENV_REFLECT_MAX_CONTEXT_TOKENS, str(DEFAULT_REFLECT_MAX_CONTEXT_TOKENS))
            ),
            reflect_wall_timeout=int(os.getenv(ENV_REFLECT_WALL_TIMEOUT, str(DEFAULT_REFLECT_WALL_TIMEOUT))),
            reflect_mission=os.getenv(ENV_REFLECT_MISSION) or None,
            reflect_source_facts_max_tokens=int(
                os.getenv(ENV_REFLECT_SOURCE_FACTS_MAX_TOKENS, str(DEFAULT_REFLECT_SOURCE_FACTS_MAX_TOKENS))
            ),
            recall_include_chunks=os.getenv(ENV_RECALL_INCLUDE_CHUNKS, str(DEFAULT_RECALL_INCLUDE_CHUNKS)).lower()
            in ("true", "1", "yes"),
            recall_max_tokens=int(os.getenv(ENV_RECALL_MAX_TOKENS, str(DEFAULT_RECALL_MAX_TOKENS))),
            recall_chunks_max_tokens=int(
                os.getenv(ENV_RECALL_CHUNKS_MAX_TOKENS, str(DEFAULT_RECALL_CHUNKS_MAX_TOKENS))
            ),
            recall_budget_function=_validate_recall_budget_function(
                os.getenv(ENV_RECALL_BUDGET_FUNCTION, DEFAULT_RECALL_BUDGET_FUNCTION)
            ),
            recall_budget_fixed_low=int(os.getenv(ENV_RECALL_BUDGET_FIXED_LOW, str(DEFAULT_RECALL_BUDGET_FIXED_LOW))),
            recall_budget_fixed_mid=int(os.getenv(ENV_RECALL_BUDGET_FIXED_MID, str(DEFAULT_RECALL_BUDGET_FIXED_MID))),
            recall_budget_fixed_high=int(
                os.getenv(ENV_RECALL_BUDGET_FIXED_HIGH, str(DEFAULT_RECALL_BUDGET_FIXED_HIGH))
            ),
            recall_budget_adaptive_low=float(
                os.getenv(ENV_RECALL_BUDGET_ADAPTIVE_LOW, str(DEFAULT_RECALL_BUDGET_ADAPTIVE_LOW))
            ),
            recall_budget_adaptive_mid=float(
                os.getenv(ENV_RECALL_BUDGET_ADAPTIVE_MID, str(DEFAULT_RECALL_BUDGET_ADAPTIVE_MID))
            ),
            recall_budget_adaptive_high=float(
                os.getenv(ENV_RECALL_BUDGET_ADAPTIVE_HIGH, str(DEFAULT_RECALL_BUDGET_ADAPTIVE_HIGH))
            ),
            recall_budget_min=int(os.getenv(ENV_RECALL_BUDGET_MIN, str(DEFAULT_RECALL_BUDGET_MIN))),
            recall_budget_max=int(os.getenv(ENV_RECALL_BUDGET_MAX, str(DEFAULT_RECALL_BUDGET_MAX))),
            # Disposition settings (None = fall back to DB value)
            disposition_skepticism=int(os.getenv(ENV_DISPOSITION_SKEPTICISM))
            if os.getenv(ENV_DISPOSITION_SKEPTICISM)
            else DEFAULT_DISPOSITION_SKEPTICISM,
            disposition_literalism=int(os.getenv(ENV_DISPOSITION_LITERALISM))
            if os.getenv(ENV_DISPOSITION_LITERALISM)
            else DEFAULT_DISPOSITION_LITERALISM,
            disposition_empathy=int(os.getenv(ENV_DISPOSITION_EMPATHY))
            if os.getenv(ENV_DISPOSITION_EMPATHY)
            else DEFAULT_DISPOSITION_EMPATHY,
            # OpenTelemetry tracing configuration
            otel_traces_enabled=os.getenv(ENV_OTEL_TRACES_ENABLED, str(DEFAULT_OTEL_TRACES_ENABLED)).lower()
            in ("true", "1", "yes"),
            otel_exporter_otlp_endpoint=os.getenv(ENV_OTEL_EXPORTER_OTLP_ENDPOINT) or None,
            otel_exporter_otlp_headers=os.getenv(ENV_OTEL_EXPORTER_OTLP_HEADERS) or None,
            otel_service_name=os.getenv(ENV_OTEL_SERVICE_NAME, DEFAULT_OTEL_SERVICE_NAME),
            otel_deployment_environment=os.getenv(ENV_OTEL_DEPLOYMENT_ENVIRONMENT, DEFAULT_OTEL_DEPLOYMENT_ENVIRONMENT),
            metrics_include_bank_id=os.getenv(ENV_METRICS_INCLUDE_BANK_ID, str(DEFAULT_METRICS_INCLUDE_BANK_ID)).lower()
            in ("true", "1", "yes"),
            # Audit log configuration (static, server-level only)
            audit_log_enabled=os.getenv(ENV_AUDIT_LOG_ENABLED, str(DEFAULT_AUDIT_LOG_ENABLED)).lower() == "true",
            audit_log_actions=[
                a.strip() for a in os.getenv(ENV_AUDIT_LOG_ACTIONS, DEFAULT_AUDIT_LOG_ACTIONS).split(",") if a.strip()
            ],
            audit_log_retention_days=int(
                os.getenv(ENV_AUDIT_LOG_RETENTION_DAYS, str(DEFAULT_AUDIT_LOG_RETENTION_DAYS))
            ),
            # Webhook configuration (static, server-level only)
            webhook_url=os.getenv(ENV_WEBHOOK_URL) or DEFAULT_WEBHOOK_URL,
            webhook_secret=os.getenv(ENV_WEBHOOK_SECRET) or DEFAULT_WEBHOOK_SECRET,
            webhook_event_types=[
                t.strip()
                for t in os.getenv(ENV_WEBHOOK_EVENT_TYPES, DEFAULT_WEBHOOK_EVENT_TYPES).split(",")
                if t.strip()
            ],
            webhook_delivery_poll_interval_seconds=int(
                os.getenv(
                    ENV_WEBHOOK_DELIVERY_POLL_INTERVAL_SECONDS,
                    str(DEFAULT_WEBHOOK_DELIVERY_POLL_INTERVAL_SECONDS),
                )
            ),
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
            allowed = frozenset(self.log_json_fields) if self.log_json_fields else None
            handler.setFormatter(JsonFormatter(allowed_fields=allowed))
        else:
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))

        root_logger.addHandler(handler)

        # Silence noisy third-party loggers
        logging.getLogger("google_genai.models").setLevel(logging.WARNING)

    def log_config(self) -> None:
        """Log the current configuration (without sensitive values)."""
        logger.info(f"Database: {mask_network_location(self.database_url)} (schema: {self.database_schema})")
        if self.read_database_url:
            logger.info(f"Read database (recall queries only): {mask_network_location(self.read_database_url)}")
        if self.migration_database_url:
            logger.info(f"Migration database: {mask_network_location(self.migration_database_url)}")
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

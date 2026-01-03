"""
Centralized configuration for Hindsight API.

All environment variables and their defaults are defined here.
"""

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Environment variable names
ENV_DATABASE_URL = "HINDSIGHT_API_DATABASE_URL"
ENV_LLM_PROVIDER = "HINDSIGHT_API_LLM_PROVIDER"
ENV_LLM_API_KEY = "HINDSIGHT_API_LLM_API_KEY"
ENV_LLM_MODEL = "HINDSIGHT_API_LLM_MODEL"
ENV_LLM_BASE_URL = "HINDSIGHT_API_LLM_BASE_URL"
ENV_LLM_MAX_CONCURRENT = "HINDSIGHT_API_LLM_MAX_CONCURRENT"
ENV_LLM_TIMEOUT = "HINDSIGHT_API_LLM_TIMEOUT"
ENV_LLM_STRIP_THINKING = "HINDSIGHT_API_LLM_STRIP_THINKING"

ENV_EMBEDDINGS_PROVIDER = "HINDSIGHT_API_EMBEDDINGS_PROVIDER"
ENV_EMBEDDINGS_LOCAL_MODEL = "HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL"
ENV_EMBEDDINGS_TEI_URL = "HINDSIGHT_API_EMBEDDINGS_TEI_URL"

ENV_RERANKER_PROVIDER = "HINDSIGHT_API_RERANKER_PROVIDER"
ENV_RERANKER_LOCAL_MODEL = "HINDSIGHT_API_RERANKER_LOCAL_MODEL"
ENV_RERANKER_TEI_URL = "HINDSIGHT_API_RERANKER_TEI_URL"

ENV_HOST = "HINDSIGHT_API_HOST"
ENV_PORT = "HINDSIGHT_API_PORT"
ENV_LOG_LEVEL = "HINDSIGHT_API_LOG_LEVEL"
ENV_MCP_ENABLED = "HINDSIGHT_API_MCP_ENABLED"
ENV_GRAPH_RETRIEVER = "HINDSIGHT_API_GRAPH_RETRIEVER"
ENV_MCP_LOCAL_BANK_ID = "HINDSIGHT_API_MCP_LOCAL_BANK_ID"
ENV_MCP_INSTRUCTIONS = "HINDSIGHT_API_MCP_INSTRUCTIONS"

# Observation thresholds
ENV_OBSERVATION_MIN_FACTS = "HINDSIGHT_API_OBSERVATION_MIN_FACTS"
ENV_OBSERVATION_TOP_ENTITIES = "HINDSIGHT_API_OBSERVATION_TOP_ENTITIES"

# Optimization flags
ENV_SKIP_LLM_VERIFICATION = "HINDSIGHT_API_SKIP_LLM_VERIFICATION"
ENV_LAZY_RERANKER = "HINDSIGHT_API_LAZY_RERANKER"

# Default values
DEFAULT_DATABASE_URL = "pg0"
DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_LLM_MODEL = "gpt-5-mini"
DEFAULT_LLM_MAX_CONCURRENT = 32
DEFAULT_LLM_TIMEOUT = 120.0  # seconds

DEFAULT_EMBEDDINGS_PROVIDER = "local"
DEFAULT_EMBEDDINGS_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"

DEFAULT_RERANKER_PROVIDER = "local"
DEFAULT_RERANKER_LOCAL_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8888
DEFAULT_LOG_LEVEL = "info"
DEFAULT_MCP_ENABLED = True
DEFAULT_GRAPH_RETRIEVER = "bfs"  # Options: "bfs", "mpfp"
DEFAULT_MCP_LOCAL_BANK_ID = "mcp"

# Observation thresholds
DEFAULT_OBSERVATION_MIN_FACTS = 5  # Min facts required to generate entity observations
DEFAULT_OBSERVATION_TOP_ENTITIES = 5  # Max entities to process per retain batch

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

# Required embedding dimension for database schema
EMBEDDING_DIMENSION = 384


@dataclass
class HindsightConfig:
    """Configuration container for Hindsight API."""

    # Database
    database_url: str

    # LLM
    llm_provider: str
    llm_api_key: str | None
    llm_model: str
    llm_base_url: str | None
    llm_max_concurrent: int
    llm_timeout: float

    # Embeddings
    embeddings_provider: str
    embeddings_local_model: str
    embeddings_tei_url: str | None

    # Reranker
    reranker_provider: str
    reranker_local_model: str
    reranker_tei_url: str | None

    # Server
    host: str
    port: int
    log_level: str
    mcp_enabled: bool

    # Recall
    graph_retriever: str

    # Observation thresholds
    observation_min_facts: int
    observation_top_entities: int

    # Optimization flags
    skip_llm_verification: bool
    lazy_reranker: bool

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
            # Embeddings
            embeddings_provider=os.getenv(ENV_EMBEDDINGS_PROVIDER, DEFAULT_EMBEDDINGS_PROVIDER),
            embeddings_local_model=os.getenv(ENV_EMBEDDINGS_LOCAL_MODEL, DEFAULT_EMBEDDINGS_LOCAL_MODEL),
            embeddings_tei_url=os.getenv(ENV_EMBEDDINGS_TEI_URL),
            # Reranker
            reranker_provider=os.getenv(ENV_RERANKER_PROVIDER, DEFAULT_RERANKER_PROVIDER),
            reranker_local_model=os.getenv(ENV_RERANKER_LOCAL_MODEL, DEFAULT_RERANKER_LOCAL_MODEL),
            reranker_tei_url=os.getenv(ENV_RERANKER_TEI_URL),
            # Server
            host=os.getenv(ENV_HOST, DEFAULT_HOST),
            port=int(os.getenv(ENV_PORT, DEFAULT_PORT)),
            log_level=os.getenv(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL),
            mcp_enabled=os.getenv(ENV_MCP_ENABLED, str(DEFAULT_MCP_ENABLED)).lower() == "true",
            # Recall
            graph_retriever=os.getenv(ENV_GRAPH_RETRIEVER, DEFAULT_GRAPH_RETRIEVER),
            # Optimization flags
            skip_llm_verification=os.getenv(ENV_SKIP_LLM_VERIFICATION, "false").lower() == "true",
            lazy_reranker=os.getenv(ENV_LAZY_RERANKER, "false").lower() == "true",
            # Observation thresholds
            observation_min_facts=int(os.getenv(ENV_OBSERVATION_MIN_FACTS, str(DEFAULT_OBSERVATION_MIN_FACTS))),
            observation_top_entities=int(
                os.getenv(ENV_OBSERVATION_TOP_ENTITIES, str(DEFAULT_OBSERVATION_TOP_ENTITIES))
            ),
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
        """Configure Python logging based on the log level."""
        logging.basicConfig(
            level=self.get_python_log_level(),
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            force=True,  # Override any existing configuration
        )

    def log_config(self) -> None:
        """Log the current configuration (without sensitive values)."""
        logger.info(f"Database: {self.database_url}")
        logger.info(f"LLM: provider={self.llm_provider}, model={self.llm_model}")
        logger.info(f"Embeddings: provider={self.embeddings_provider}")
        logger.info(f"Reranker: provider={self.reranker_provider}")
        logger.info(f"Graph retriever: {self.graph_retriever}")


def get_config() -> HindsightConfig:
    """Get the current configuration from environment variables."""
    return HindsightConfig.from_env()

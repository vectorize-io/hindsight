"""Central API configuration.

Config fields only — no real connections are opened at import time. Engine URLs
and Authentik settings establish the module boundary for later wiring.
"""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CENTRAL_API_", env_file=".env", extra="ignore")

    # Runtime
    env: str = "development"
    debug: bool = True

    # Security
    internal_context_secret: SecretStr = SecretStr("dev-secret-do-not-use-in-prod")

    # Auth (module boundary — no JWKS calls are made in the scaffold)
    authentik_issuer: str = "https://auth.example.internal/application/o/collabmind/"
    authentik_jwks_url: str = (
        "https://auth.example.internal/application/o/collabmind/.well-known/jwks.json"
    )
    jwt_algorithms: tuple[str, ...] = ("RS256",)

    # Engine endpoints (internal-only)
    memory_controller_url: str = "http://memory-controller:3020"
    memlord_url: str = "http://memlord.internal:8000"
    memlord_api_key: str = ""  # mlk_… ; when set, MemlordAdapter makes real calls
    hindsight_url: str = ""  # Hindsight MCP server HTTP endpoint
    hindsight_api_key: str = ""  # Optional API key for Hindsight
    hindsight_bank_id: str = "collabmind-control-plane"  # Default bank for CollabMind memories
    coderag_url: str = "http://coderag.internal:8000"
    openmemory_url: str = "http://openmemory.internal:8080"

    # Storage / control-plane database (source of truth for governance).
    # async driver URLs: postgresql+asyncpg://… (prod) or sqlite+aiosqlite://… (dev/test)
    database_url: str = ""
    qdrant_url: str = ""

    # AI Gateway
    litellm_url: str = "http://litellm.internal:4000"
    litellm_api_key: str = ""  # Bearer key for the LiteLLM proxy
    ollama_url: str = "http://ollama.internal:11434"

    # LocalAI provider (first governed provider backend)
    localai_base_url: str = "https://localai.collabmind.dev"
    localai_api_key: str = ""  # Optional Bearer key for LocalAI

    # Cloud / enterprise provider keys (optional; stored in env only, never committed)
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    xai_api_key: str = ""
    aws_profile: str = ""
    vertex_project_id: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""

    # Cloud / enterprise provider keys (optional; stored in env only, never committed)
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    xai_api_key: str = ""
    aws_profile: str = ""
    vertex_project_id: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""

    # Google Drive connector (read-only). Secrets come from the environment ONLY;
    # never commit real values — see .env.example. Empty client_id => connector
    # reports "not_configured" and refuses to start an OAuth flow.
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://localhost:8000/connectors/google-drive/callback"
    # Minimal read-only scopes — write/delete/admin scopes are forbidden by policy.
    google_scopes: tuple[str, ...] = (
        "https://www.googleapis.com/auth/drive.metadata.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    )

    @property
    def is_dev(self) -> bool:
        return self.env != "production"

    @property
    def effective_database_url(self) -> str:
        """DB URL with a safe dev/test default so nothing connects to prod by accident."""
        if self.database_url:
            return self.database_url
        return "sqlite+aiosqlite:///./central_api_dev.db"

    @property
    def google_configured(self) -> bool:
        return bool(self.google_client_id and self.google_client_secret)


settings = Settings()

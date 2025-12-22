"""
Centralized configuration module for Hindsight Foundry Agent.

Loads configuration from Azure App Configuration with fallback to environment variables.
"""
import os
from dataclasses import dataclass
from typing import Optional

# Try to import Azure App Configuration provider
try:
    from azure.appconfiguration.provider import load
    # DefaultAzureCredential unused but keeping if needed for future AAD auth
    HAS_APP_CONFIG = True
except ImportError:
    HAS_APP_CONFIG = False


@dataclass
class HindsightConfig:
    """Configuration for the Hindsight Foundry Agent."""
    project_endpoint: str
    model_deployment_name: str
    mcp_base_url: str
    default_bank_id: str
    agent_name: str
    
    @property
    def mcp_url(self) -> str:
        """Full MCP URL including bank_id path."""
        base = self.mcp_base_url.rstrip('/')
        return f"{base}/mcp/{self.default_bank_id}/"


# Azure App Configuration endpoint and connection string
APP_CONFIG_ENDPOINT = "https://hindsightapp.azconfig.io"
# Use environment variable for secrets - never hardcode credentials
APP_CONFIG_CONNECTION_STRING = os.environ.get(
    "AZURE_APP_CONFIG_CONNECTION_STRING",
    ""  # Must be set in environment for App Config to work
)

# Prefix for configuration keys in App Configuration
CONFIG_PREFIX = "Hindsight:"


def _get_from_app_config() -> Optional[dict]:
    """Load configuration from Azure App Configuration."""
    if not HAS_APP_CONFIG:
        print("Azure App Configuration SDK not installed")
        return None
    
    try:
        from azure.appconfiguration.provider import SettingSelector
        
        # Load using connection string (access key auth)
        if not APP_CONFIG_CONNECTION_STRING:
            print("WARNING: AZURE_APP_CONFIG_CONNECTION_STRING is empty. App Config will fail.")
            
        config = load(
            connection_string=APP_CONFIG_CONNECTION_STRING,
            selects=[SettingSelector(key_filter=f"{CONFIG_PREFIX}*")],
            trim_prefixes=[CONFIG_PREFIX],
        )
        
        if config:
            print("Loaded configuration from Azure App Configuration")
            return dict(config)
        
    except Exception as e:
        print(f"WARNING: Could not load from App Configuration: {e}")
    
    return None


def _get_from_env() -> dict:
    """Load configuration from environment variables."""
    return {
        "ProjectEndpoint": os.environ.get(
            "HINDSIGHT_PROJECT_ENDPOINT",
            "https://jacob-1216-resource.services.ai.azure.com/api/projects/jacob-1216"
        ),
        # Use gpt-4o as default - verified working deployment
        # Set HINDSIGHT_MODEL_DEPLOYMENT_NAME to override
        "ModelDeploymentName": os.environ.get(
            "HINDSIGHT_MODEL_DEPLOYMENT_NAME",
            "gpt-5.2-chat"
        ),
        "McpBaseUrl": os.environ.get(
            "HINDSIGHT_MCP_BASE_URL",
            "https://hindsight-api.politebay-1635b4f9.centralus.azurecontainerapps.io"
        ),
        "DefaultBankId": os.environ.get(
            "HINDSIGHT_DEFAULT_BANK_ID",
            "hindsight_agent_bank"
        ),
        "AgentName": os.environ.get(
            "HINDSIGHT_AGENT_NAME",
            "Hindsight"
        ),
    }


def get_config() -> HindsightConfig:
    """
    Get configuration from Azure App Configuration or environment variables.
    
    Priority:
    1. Azure App Configuration (if available)
    2. Environment variables
    3. Hardcoded defaults
    """
    # Try App Configuration first
    config_dict = _get_from_app_config()
    
    # Allow environment variable overrides
    if config_dict:
        if "HINDSIGHT_MODEL_DEPLOYMENT_NAME" in os.environ:
            print(f"ℹ Overriding model with: {os.environ['HINDSIGHT_MODEL_DEPLOYMENT_NAME']}")
            config_dict["ModelDeploymentName"] = os.environ["HINDSIGHT_MODEL_DEPLOYMENT_NAME"]
    
    # Fall back to environment variables
    if not config_dict:
        print("Using environment/default configuration")
        config_dict = _get_from_env()
    
    return HindsightConfig(
        project_endpoint=config_dict.get("ProjectEndpoint", ""),
        model_deployment_name=config_dict.get("ModelDeploymentName", "gpt-5.2-chat"),
        mcp_base_url=config_dict.get("McpBaseUrl", ""),
        default_bank_id=config_dict.get("DefaultBankId", "hindsight_agent_bank"),
        agent_name=config_dict.get("AgentName", "Hindsight"),
    )


if __name__ == "__main__":
    # Test configuration loading
    print("Testing configuration loading...")
    config = get_config()
    print(f"\nConfiguration loaded:")
    print(f"  Project Endpoint: {config.project_endpoint}")
    print(f"  Model: {config.model_deployment_name}")
    print(f"  MCP Base URL: {config.mcp_base_url}")
    print(f"  Default Bank ID: {config.default_bank_id}")
    print(f"  Agent Name: {config.agent_name}")
    print(f"  Full MCP URL: {config.mcp_url}")

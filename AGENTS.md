# Hindsight Agent: Azure AI Foundry Operations Manual

This document serves as the **Operational Manual** for the Hindsight Agent deployed on Azure AI Foundry. It covers cloud resources, identity management, deployment workflows, API patterns, and troubleshooting.

---

## 📋 Table of Contents

1. [Resource Inventory](#-resource-inventory)
2. [Architecture Overview](#-architecture-overview)
3. [Remote Agent API](#-remote-agent-api)
4. [Local Development](#-local-development)
5. [Deployment Guide](#-deployment-guide)
6. [Authentication & Identity](#-authentication--identity)
7. [Configuration Management](#-configuration-management)
8. [Agent Modification Workflow](#-agent-modification-workflow)
9. [API Patterns & SDK Usage](#-api-patterns--sdk-usage)
10. [Memory Banks](#-memory-banks)
11. [Troubleshooting](#-troubleshooting)

---

## ☁️ Resource Inventory

### Core Azure Resources

| Component | Azure Resource Name / ID | Purpose |
|-----------|-------------------------|---------|
| **AI Project** | `jacob-1216` | Container for the Agent Service in Azure AI Foundry |
| **Project Endpoint** | `https://jacob-1216-resource.services.ai.azure.com/api/projects/jacob-1216` | Data Plane URL for SDK connectivity |
| **AI Resource** | `jacob-1216-resource` | Cognitive Services resource hosting models |
| **Managed Identity** | `267bc722-69a7-4bca-9196-9e8133094d37` | System identity for agent services |
| **Location** | `centralus` | Primary Azure region |

### Compute & Container Resources

| Component | Resource Name | URL | Purpose |
|-----------|--------------|-----|---------|
| **Memory API** | `hindsight-api` | `https://hindsight-api.politebay-1635b4f9.centralus.azurecontainerapps.io` | Core memory storage/retrieval (retain, recall, reflect) |
| **Agent API** | `hindsight-agent-api` | `https://hindsight-agent-api.jollyforest-7224b47b.centralus.azurecontainerapps.io` | Remote HTTP access to Hindsight agent |
| **Container Registry** | `hindsightacr9631` | `hindsightacr9631.azurecr.io` | Docker image storage |
| **Container Environment** | `hindsight-env` | - | Container Apps managed environment |
| **Log Analytics** | `hindsight-logs` | - | Centralized logging |

### Model Deployments

| Deployment Name | Model | Status | Notes |
|-----------------|-------|--------|-------|
| `gpt-4o` | GPT-4o | ✅ Active | Primary model for Hindsight agent |
| `gpt-5.2-chat` | GPT-5.2 | ⚠️ Available | Post-cutoff model, available for use |

### Configuration Store

| Component | Endpoint | Auth Method |
|-----------|----------|-------------|
| **App Configuration** | `https://hindsightapp.azconfig.io` | Entra ID (AAD) or Connection String |

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Azure AI Foundry                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Project: jacob-1216                                         │    │
│  │  ┌─────────────────┐    ┌─────────────────────────────────┐ │    │
│  │  │  Agent:         │    │  Model Deployment:              │ │    │
│  │  │  Hindsight-v2   │───▶│  gpt-4o                         │ │    │
│  │  │  (agent_ref)    │    │                                 │ │    │
│  │  └─────────────────┘    └─────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ Responses API
                              │ (agent_reference pattern)
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Azure Container Apps                              │
│                                                                      │
│  ┌────────────────────────┐      ┌────────────────────────────┐    │
│  │ hindsight-agent-api    │      │ hindsight-api              │    │
│  │ (FastAPI)              │─────▶│ (Memory System)            │    │
│  │                        │      │                            │    │
│  │ POST /chat             │      │ POST /memories             │    │
│  │ GET  /health           │      │ POST /memories/recall      │    │
│  └────────────────────────┘      │ POST /memories/reflect     │    │
│                                  └────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Client** → `POST /chat` → **hindsight-agent-api**
2. **Agent API** → `responses.create(agent_reference)` → **Azure AI Foundry**
3. **Foundry** returns tool calls (retain/recall/reflect)
4. **Agent API** → executes tools via **HindsightClient** → **hindsight-api**
5. **Agent API** → submits tool outputs → **Foundry** → generates response
6. **Client** ← receives response with tool call details

---

## 🤖 Remote Agent API

The `hindsight-agent-api` provides remote HTTP access to the Hindsight agent.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Send a message, get agent response with tool calls |
| `GET` | `/health` | Health check with status and configuration |
| `GET` | `/docs` | Interactive OpenAPI documentation |
| `GET` | `/` | API info and available endpoints |

### Request/Response Format

**POST /chat**
```json
// Request
{
  "message": "What do you know about me?",
  "conversation_id": null
}

// Response
{
  "response": "Based on my memories, your name is Jacob. You live in Seattle and work at Microsoft...",
  "conversation_id": null,
  "tool_calls": [
    {
      "name": "recall",
      "arguments": {"query": "user profile preferences", "bank_id": "user_preferences"},
      "result_preview": "{\"results\": [{\"id\": \"...\", \"text\": \"User's name is Jacob...\"}]}"
    }
  ]
}
```

**GET /health**
```json
{
  "status": "healthy",
  "agent": "Hindsight-v2",
  "project_endpoint": "https://jacob-1216-resource.services.ai.azure.com/api/projects/jacob-1216"
}
```

### Quick Test (PowerShell)

```powershell
# Health check
Invoke-RestMethod -Uri "https://hindsight-agent-api.jollyforest-7224b47b.centralus.azurecontainerapps.io/health"

# Chat
$body = @{message = "What do you know about me?"} | ConvertTo-Json
Invoke-RestMethod -Uri "https://hindsight-agent-api.jollyforest-7224b47b.centralus.azurecontainerapps.io/chat" `
  -Method POST -Body $body -ContentType "application/json"
```

---

## 💻 Local Development

### Prerequisites

- Python 3.12+
- Azure CLI (`az login` required)
- Access to `jacob-1216` AI Project

### Setup

```powershell
# Clone and setup
cd hindsight
pip install -r requirements-agent-api.txt

# Login to Azure (required for local credential)
az login
az account set --subscription "<your-subscription-id>"

# Run locally
python hindsight_agent_api.py
```

### Local Testing

```powershell
# Test local health
Invoke-RestMethod -Uri "http://localhost:8080/health"

# Test local chat
$body = @{message = "Hello!"} | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8080/chat" -Method POST -Body $body -ContentType "application/json"
```

### Interactive CLI Mode

```powershell
# Run interactive agent locally
python hindsight_agent.py --interactive

# Single message
python hindsight_agent.py --message "What do you remember about me?"
```

---

## 🚀 Deployment Guide

### Prerequisites

- Azure CLI authenticated (`az login`)
- Contributor role on `hindsight-rg` resource group
- Docker (optional, for local builds)

### Deploy with Bicep (Recommended)

```powershell
# Full deployment (creates/updates all resources)
.\deploy-bicep.ps1 -ResourceGroup hindsight-rg -Location centralus

# Deploy with specific image tag
.\deploy-bicep.ps1 -ResourceGroup hindsight-rg -ImageTag v1.0.0
```

### What the Deployment Does

1. **Creates/Updates Infrastructure** (via Bicep):
   - Log Analytics Workspace
   - Container Registry (ACR)
   - Container Apps Environment
   - Container App with managed identity

2. **Builds Container Image**:
   - Uses ACR Tasks (`az acr build`)
   - Pushes to `hindsightacr9631.azurecr.io/hindsight-agent-api:latest`

3. **Configures RBAC**:
   - Assigns `Cognitive Services User` role to Container App's managed identity
   - Scoped to `jacob-1216-resource` AI resource

### Manual Container Build (Alternative)

```powershell
# Build locally
docker build -t hindsight-agent-api:latest -f Dockerfile.agent-api .

# Tag for ACR
docker tag hindsight-agent-api:latest hindsightacr9631.azurecr.io/hindsight-agent-api:latest

# Push to ACR
az acr login --name hindsightacr9631
docker push hindsightacr9631.azurecr.io/hindsight-agent-api:latest

# Update Container App
az containerapp update --name hindsight-agent-api --resource-group hindsight-rg `
  --image hindsightacr9631.azurecr.io/hindsight-agent-api:latest
```

### Verify Deployment

```powershell
# Check container status
az containerapp show --name hindsight-agent-api --resource-group hindsight-rg `
  --query "{fqdn:properties.configuration.ingress.fqdn, status:properties.provisioningState, revision:properties.latestRevisionName}"

# View logs
az containerapp logs show --name hindsight-agent-api --resource-group hindsight-rg --type console --tail 50

# Test deployed API
Invoke-RestMethod -Uri "https://hindsight-agent-api.jollyforest-7224b47b.centralus.azurecontainerapps.io/health"
```

---

## 🔐 Authentication & Identity

### Authentication Model by Environment

| Environment | Credential Type | How to Setup |
|-------------|----------------|--------------|
| **Local Development** | `AzureCliCredential` | Run `az login` and select correct subscription |
| **Azure Container Apps** | `ManagedIdentityCredential` | Automatic via system-assigned identity |
| **CI/CD Pipelines** | `DefaultAzureCredential` | Use federated credentials or service principal |

### Required RBAC Roles

| Identity | Role | Scope | Purpose |
|----------|------|-------|---------|
| Your User | `Azure AI Developer` | `jacob-1216` project | Local development, agent creation |
| Container App MI | `Cognitive Services User` | `jacob-1216-resource` | Production API calls |

### Credential Selection Logic

```python
def get_credential():
    """Get the appropriate credential based on environment."""
    is_azure = any([
        os.environ.get("AZURE_FUNCTIONS_ENVIRONMENT"),
        os.environ.get("CONTAINER_APP_NAME"),
        os.environ.get("MSI_ENDPOINT"),
        os.environ.get("IDENTITY_ENDPOINT")
    ])
    
    if is_azure:
        return ManagedIdentityCredential()
    
    # Local: prefer CLI credential (has your user permissions)
    return AzureCliCredential()
```

### Verify Permissions

```powershell
# Check your role assignments on AI Project
az role assignment list --assignee $(az ad signed-in-user show --query id -o tsv) `
  --scope "/subscriptions/<sub-id>/resourceGroups/jacob-1216-resource/providers/Microsoft.CognitiveServices/accounts/jacob-1216-resource"

# Check Container App managed identity
az containerapp identity show --name hindsight-agent-api --resource-group hindsight-rg
```

---

## ⚙️ Configuration Management

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HINDSIGHT_PROJECT_ENDPOINT` | `https://jacob-1216-resource...` | Foundry project data plane URL |
| `HINDSIGHT_MODEL_DEPLOYMENT_NAME` | `gpt-4o` | Model deployment to use |
| `HINDSIGHT_MCP_BASE_URL` | `https://hindsight-api...` | Memory API base URL |
| `HINDSIGHT_DEFAULT_BANK_ID` | `hindsight_agent_bank` | Default memory bank |
| `HINDSIGHT_AGENT_NAME` | `Hindsight` | Agent display name |

### Azure App Configuration

Keys stored in App Configuration (prefix: `Hindsight:`):

- `Hindsight:ProjectEndpoint`
- `Hindsight:ModelDeploymentName`
- `Hindsight:McpBaseUrl`
- `Hindsight:DefaultBankId`
- `Hindsight:AgentName`

### Configuration Priority

1. Environment variables (highest priority)
2. Azure App Configuration
3. Hardcoded defaults

### Override Model at Runtime

```powershell
# Local override
$env:HINDSIGHT_MODEL_DEPLOYMENT_NAME = "gpt-5.2-chat"
python hindsight_agent.py --interactive

# Container App override
az containerapp update --name hindsight-agent-api --resource-group hindsight-rg `
  --set-env-vars "HINDSIGHT_MODEL_DEPLOYMENT_NAME=gpt-5.2-chat"
```

---

## 🔧 Agent Modification Workflow

### Updating System Prompt

1. **Edit Instructions**: Modify `AGENT_INSTRUCTIONS` in `hindsight_agent.py`
2. **Redeploy**: Run `deploy-bicep.ps1` or manually update container
3. **Test**: Verify via `/chat` endpoint

### Switching Models

1. **Verify Deployment Exists**: Check Azure AI Studio for available model deployments
2. **Update Configuration**:
   - App Configuration: `Hindsight:ModelDeploymentName`
   - Or environment variable: `HINDSIGHT_MODEL_DEPLOYMENT_NAME`
3. **Restart Container App** (if needed):
   ```powershell
   az containerapp revision restart --name hindsight-agent-api --resource-group hindsight-rg `
     --revision $(az containerapp show --name hindsight-agent-api --resource-group hindsight-rg --query properties.latestRevisionName -o tsv)
   ```

### Adding New Tools

1. **Define Tool Schema** in `HINDSIGHT_TOOLS` array
2. **Implement Handler** in `handle_tool_call()` function
3. **Add Client Method** in `HindsightClient` if calling external API
4. **Redeploy**

---

## 🔌 API Patterns & SDK Usage

### Foundry Responses API (Correct Pattern)

The agent uses the **Responses API** with `agent_reference` pattern:

```python
from azure.ai.projects import AIProjectClient

client = AIProjectClient(credential=credential, endpoint=project_endpoint)
openai_client = client.get_openai_client()

# Create response using agent reference
response = openai_client.responses.create(
    extra_body={"agent": {"name": "Hindsight-v2", "type": "agent_reference"}},
    input=user_input,
)

# Handle tool calls (always execute client-side)
for tool_call in response.output:
    if tool_call.type == 'function_call':
        result = execute_tool(tool_call.name, tool_call.arguments)
        tool_outputs.append({
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": result
        })

# Continue with tool outputs
response = openai_client.responses.create(
    extra_body={"agent": {"name": "Hindsight-v2", "type": "agent_reference"}},
    input=tool_outputs,
    previous_response_id=response.id,
)
```

### Key API Insights

| Pattern | Status | Notes |
|---------|--------|-------|
| `responses.create()` with `agent_reference` | ✅ Working | Correct pattern for Foundry agents |
| `chat.completions.create()` with `model=` | ❌ Fails | Returns 404 for deployment names |
| Tools in `extra_body` with `agent_reference` | ❌ Fails | "tools not allowed when agent is specified" |
| Client-side tool execution | ✅ Required | Server may return `completed` but still needs output |

---

## 🧠 Memory Banks

### Available Banks

| Bank ID | Purpose | Usage |
|---------|---------|-------|
| `hindsight_agent_bank` | Default bank for agent memories | General storage |
| `user_preferences` | User preferences and profile | Name, location, settings |
| `project_context` | Project-related information | Work, tasks, milestones |
| `knowledge_base` | Facts and knowledge | Technical info, references |

### HindsightClient Operations

```python
from hindsight_client import HindsightClient

client = HindsightClient(base_url, default_bank_id)

# Store a memory
client.retain(content="User prefers Python", context="preferences")

# Search memories
results = client.recall(query="user preferences", max_tokens=4096, budget="mid")

# Synthesize/reflect
reflection = client.reflect(query="What do I know about this user?")

# Query specific bank
client.recall(query="projects", bank_id="project_context")
```

---

## 🐞 Troubleshooting

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `401 Unauthorized` | Missing/invalid credentials | Local: `az login`. Cloud: Check MI role assignment |
| `404 DeploymentNotFound` | Using `model=` instead of `agent_reference` | Use `extra_body={"agent": {...}}` pattern |
| `tools not allowed when agent is specified` | Providing tools with agent_reference | Remove tools from API call; agent has tools defined in Foundry |
| Tool call completes but no response | Not providing tool outputs to API | Always submit `function_call_output` even if status="completed" |
| Container timeout | hindsight-api scaled to zero | Wake the API first with health check |

### Debug Commands

```powershell
# Check container logs
az containerapp logs show --name hindsight-agent-api --resource-group hindsight-rg --type console --tail 100

# Check container status
az containerapp show --name hindsight-agent-api --resource-group hindsight-rg --query properties.runningStatus

# List revisions
az containerapp revision list --name hindsight-agent-api --resource-group hindsight-rg --query "[].{name:name, active:properties.active, created:properties.createdTime}"

# Check role assignments
az role assignment list --scope "/subscriptions/<sub-id>/resourceGroups/jacob-1216-resource/providers/Microsoft.CognitiveServices/accounts/jacob-1216-resource" --output table
```

### Health Checks

```powershell
# Memory API health
Invoke-RestMethod -Uri "https://hindsight-api.politebay-1635b4f9.centralus.azurecontainerapps.io/health"

# Agent API health
Invoke-RestMethod -Uri "https://hindsight-agent-api.jollyforest-7224b47b.centralus.azurecontainerapps.io/health"

# Full chat test
$body = @{message = "Test: What do you know about me?"} | ConvertTo-Json
Invoke-RestMethod -Uri "https://hindsight-agent-api.jollyforest-7224b47b.centralus.azurecontainerapps.io/chat" -Method POST -Body $body -ContentType "application/json"
```

---

## 📁 File Structure

```
hindsight/
├── AGENTS.md                    # This file - operations manual
├── hindsight_agent.py           # Main agent script (local/interactive)
├── hindsight_agent_api.py       # FastAPI wrapper for remote access
├── hindsight_client.py          # Memory API client (retain/recall/reflect)
├── config.py                    # Configuration management
├── Dockerfile.agent-api         # Container definition
├── requirements-agent-api.txt   # Python dependencies
├── deploy-bicep.ps1             # Deployment script
└── infra/
    └── agent-api.bicep          # Infrastructure as Code
```

---

## 📝 Changelog

- **2024-12-19**: Initial comprehensive documentation
- **2024-12-19**: Deployed Agent API to Azure Container Apps
- **2024-12-19**: Optimized agent to use Responses API with client-side tool execution

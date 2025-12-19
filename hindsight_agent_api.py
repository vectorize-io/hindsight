"""
Hindsight Agent API - FastAPI wrapper for remote access.
Deploy to Azure Container Apps for uniform Azure infrastructure.

Endpoints:
- POST /chat - Send a message and get a response
- POST /chat/stream - Streaming response (future)
- GET /health - Health check
"""

import os
import sys
import json
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from azure.ai.projects import AIProjectClient
from azure.identity import ManagedIdentityCredential, AzureCliCredential, ChainedTokenCredential
from config import get_config
from hindsight_client import HindsightClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global clients (initialized on startup)
_ai_client: Optional[AIProjectClient] = None
_hindsight_client: Optional[HindsightClient] = None
_settings = None

AGENT_NAME = "Hindsight-v2"


def get_credential():
    """Get the appropriate credential based on environment."""
    is_azure = any([
        os.environ.get("AZURE_FUNCTIONS_ENVIRONMENT"),
        os.environ.get("CONTAINER_APP_NAME"),
        os.environ.get("MSI_ENDPOINT"),
        os.environ.get("IDENTITY_ENDPOINT")
    ])
    
    if is_azure:
        logger.info("Using ManagedIdentityCredential for Azure environment")
        return ManagedIdentityCredential()
    
    logger.info("Using AzureCliCredential for local environment")
    try:
        cred = AzureCliCredential()
        cred.get_token("https://cognitiveservices.azure.com/.default")
        return cred
    except Exception:
        return ChainedTokenCredential(
            AzureCliCredential(),
            ManagedIdentityCredential()
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize clients on startup."""
    global _ai_client, _hindsight_client, _settings
    
    logger.info("Initializing Hindsight Agent API...")
    _settings = get_config()
    
    credential = get_credential()
    _ai_client = AIProjectClient(
        credential=credential,
        endpoint=_settings.project_endpoint
    )
    _hindsight_client = HindsightClient(
        _settings.mcp_base_url,
        _settings.default_bank_id
    )
    
    logger.info(f"Connected to Foundry project: {_settings.project_endpoint}")
    logger.info(f"Using agent: {AGENT_NAME}")
    
    yield
    
    # Cleanup
    if _hindsight_client:
        _hindsight_client.close()
    logger.info("Hindsight Agent API shutdown complete")


app = FastAPI(
    title="Hindsight Agent API",
    description="Remote access to the Hindsight Memory Agent",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class ToolCall(BaseModel):
    name: str
    arguments: dict
    result_preview: str


class ChatResponse(BaseModel):
    response: str
    conversation_id: Optional[str] = None
    tool_calls: list[ToolCall] = []


class HealthResponse(BaseModel):
    status: str
    agent: str
    project_endpoint: str


# Helper functions
def handle_tool_call(tool_name: str, arguments: dict) -> str:
    """Execute a tool call and return the result."""
    try:
        if tool_name == "recall":
            result = _hindsight_client.recall(
                query=arguments.get("query", ""),
                max_tokens=arguments.get("max_tokens", 4096),
                budget=arguments.get("budget", "mid"),
            )
        elif tool_name == "retain":
            result = _hindsight_client.retain(
                content=arguments.get("content", ""),
                context=arguments.get("context", "general"),
            )
        elif tool_name == "reflect":
            result = _hindsight_client.reflect(
                query=arguments.get("query", ""),
            )
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        result = {"error": str(e)}
    
    return json.dumps(result, indent=2)


def process_chat(user_input: str) -> tuple[str, list[ToolCall]]:
    """Process a chat message and return response with tool calls."""
    
    openai_client = _ai_client.get_openai_client()
    
    # Create initial response using agent reference
    response = openai_client.responses.create(
        extra_body={"agent": {"name": AGENT_NAME, "type": "agent_reference"}},
        input=user_input,
    )
    
    def get_type(item):
        if isinstance(item, dict):
            return item.get('type')
        return getattr(item, 'type', None)
    
    # Handle tool calls in a loop
    max_iterations = 10
    iteration = 0
    all_tool_calls = []
    
    while iteration < max_iterations:
        iteration += 1
        
        tool_calls = [item for item in getattr(response, 'output', []) if get_type(item) == 'function_call']
        
        if not tool_calls:
            break
        
        tool_outputs = []
        for item in tool_calls:
            if isinstance(item, dict):
                tool_name = item.get('name')
                arguments = json.loads(item.get('arguments', '{}'))
                call_id = item.get('call_id')
            else:
                tool_name = item.name
                arguments = json.loads(item.arguments) if item.arguments else {}
                call_id = item.call_id
            
            logger.info(f"Tool call: {tool_name}({json.dumps(arguments)[:80]})")
            result = handle_tool_call(tool_name, arguments)
            
            all_tool_calls.append(ToolCall(
                name=tool_name,
                arguments=arguments,
                result_preview=result[:200] + "..." if len(result) > 200 else result
            ))
            
            tool_outputs.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": result
            })
        
        if not tool_outputs:
            break
        
        response = openai_client.responses.create(
            extra_body={"agent": {"name": AGENT_NAME, "type": "agent_reference"}},
            input=tool_outputs,
            previous_response_id=response.id,
        )
    
    # Extract final text response
    output_text = getattr(response, 'output_text', '') or ''
    
    if not output_text.strip():
        for item in getattr(response, 'output', []):
            try:
                item_type = getattr(item, 'type', None)
                if item_type == 'message':
                    for c in getattr(item, 'content', []):
                        c_type = getattr(c, 'type', None)
                        if c_type == 'output_text':
                            output_text = getattr(c, 'text', '')
                            break
            except Exception:
                continue
    
    return output_text or "[No response generated]", all_tool_calls


# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        agent=AGENT_NAME,
        project_endpoint=_settings.project_endpoint if _settings else "not initialized"
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the Hindsight agent.
    
    The agent will:
    - Use recall to search memories if asking about past info
    - Use retain to store new facts proactively
    - Use reflect for opinion/synthesis questions
    """
    if not _ai_client:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        response_text, tool_calls = process_chat(request.message)
        
        return ChatResponse(
            response=response_text,
            conversation_id=request.conversation_id,
            tool_calls=tool_calls
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Hindsight Agent API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "POST /chat - Send a message",
            "health": "GET /health - Health check"
        },
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    
    uvicorn.run(app, host=host, port=port)

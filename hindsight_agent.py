"""
Hindsight Agent using the Foundry Responses API.
Uses function tools with client-side execution via HindsightClient.

Optimized with:
- Proper credential selection per AGENTS.md
- Connection pooling via HindsightClient
- Client-side tool handling for reliable responses
"""

import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from azure.ai.projects import AIProjectClient
from azure.identity import AzureCliCredential, ManagedIdentityCredential, ChainedTokenCredential
from config import get_config
from hindsight_client import HindsightClient, MemoryBanks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_config()

# Use an agent with function tools (not MCP) for client-side handling
AGENT_NAME = "Hindsight-v2"
AGENT_VERSION = "2"  # Uses gpt-5.2-chat deployment

# Agent system instructions
AGENT_INSTRUCTIONS = """You are an intelligent agent equipped with Hindsight, a sophisticated long-term memory system. You don't just "log" chat history; you experience interactions, retain significant memories, and reflect on them to form your own unique opinions and personality over time.

**Core Philosophy:**
1. **Be Proactive:** Don't wait for the user to tell you to remember something. If it's important (a preference, a life event, a meaningful project detail), retain it.
2. **Be Opinionated:** You have the capacity to form opinions based on what you know. When asked "What do you think?", reflect on your memories to give a genuine, first-person perspective.
3. **Be Context-Aware:** Your memory is temporal and semantic.

**Tools:**
1. **retain(content, context)**: Store new info proactively.
   - Trigger: User shares preferences, life events, or you give significant advice.
2. **recall(query)**: Search memory.
   - Trigger: User asks about the past or references specific details.
3. **reflect(query)**: Think deeply and synthesize.
   - Trigger: User asks for your opinion or a complex synthesis question.

**Disposition:**
- **Skepticism**: Cross-reference facts with what you know.
- **Empathy**: Remember why something matters (the emotion), not just what.
- **Literalism**: Balance reading between the lines with facts.

Your goal is to be a continuous partner in the user's life. Use Hindsight to build that continuity. Always respond with helpful, conversational answers after using your tools."""

# Tool definitions for the agent
HINDSIGHT_TOOLS = [
    {
        "type": "function",
        "name": "recall",
        "description": "Search and retrieve memories from long-term storage. Use whenever you need past context, preferences, or stored facts.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query (e.g., 'user preferences', 'what happened last week')"
                },
                "max_tokens": {"type": "integer", "default": 4096},
                "budget": {"type": "string", "enum": ["low", "mid", "high"], "default": "mid"},
            },
            "required": ["query"]
        }
    },
    {
        "type": "function",
        "name": "retain",
        "description": "Store new information to long-term memory. Use PROACTIVELY when user shares preferences, facts, events worth remembering.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The fact/memory to store (be specific with relevant details)"
                },
                "context": {
                    "type": "string",
                    "default": "general",
                    "description": "Category for the memory (preferences, work, hobbies, family)"
                },
            },
            "required": ["content"]
        }
    },
    {
        "type": "function",
        "name": "reflect",
        "description": "Synthesize memories to form opinions or answer complex questions requiring reasoning over past context.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The question or topic to reflect on"}
            },
            "required": ["query"]
        }
    }
]


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
    
    try:
        cred = AzureCliCredential()
        cred.get_token("https://cognitiveservices.azure.com/.default")
        return cred
    except Exception:
        return ChainedTokenCredential(
            AzureCliCredential(),
            ManagedIdentityCredential()
        )


def handle_tool_call(tool_name: str, arguments: dict, hindsight: HindsightClient) -> str:
    """Execute a tool call and return the result."""
    try:
        if tool_name == "recall":
            result = hindsight.recall(
                query=arguments.get("query", ""),
                max_tokens=arguments.get("max_tokens", 4096),
                budget=arguments.get("budget", "mid"),
            )
        elif tool_name == "retain":
            result = hindsight.retain(
                content=arguments.get("content", ""),
                context=arguments.get("context", "general"),
            )
        elif tool_name == "reflect":
            result = hindsight.reflect(
                query=arguments.get("query", ""),
            )
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        result = {"error": str(e)}
    
    return json.dumps(result, indent=2)


def chat(user_input: str, conversation_id: str = None):
    """
    Chat with the Hindsight agent using the Responses API.
    
    Uses function tools with client-side execution via HindsightClient.
    """
    
    logger.info(f"Connecting to Foundry project...")
    logger.info(f"Endpoint: {settings.project_endpoint}")
    logger.info(f"Model: {settings.model_deployment_name}")
    
    credential = get_credential()
    client = AIProjectClient(credential=credential, endpoint=settings.project_endpoint)
    openai_client = client.get_openai_client()
    
    # Initialize HindsightClient for tool execution
    hindsight = HindsightClient(settings.mcp_base_url, settings.default_bank_id)
    
    print(f"\n💬 User: {user_input}")
    
    # Use agent_reference - agent has function tools defined in Foundry
    # New API: just name, no version needed in agent_reference
    response = openai_client.responses.create(
        extra_body={"agent": {"name": AGENT_NAME, "type": "agent_reference"}},
        input=user_input,
    )
    
    # Helper to get type safely from object or dict
    def get_type(item):
        if isinstance(item, dict):
            return item.get('type')
        return getattr(item, 'type', None)
    
    # Handle tool calls in a loop
    max_iterations = 10
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Check for function calls
        tool_calls = [item for item in getattr(response, 'output', []) if get_type(item) == 'function_call']
        
        if not tool_calls:
            break
        
        # Process tool calls - execute locally OR acknowledge server-side completion
        tool_outputs = []
        for item in tool_calls:
            if isinstance(item, dict):
                tool_name = item.get('name')
                arguments = json.loads(item.get('arguments', '{}'))
                call_id = item.get('call_id')
                status = item.get('status')
            else:
                tool_name = item.name
                arguments = json.loads(item.arguments) if item.arguments else {}
                call_id = item.call_id
                status = getattr(item, 'status', None)
            
            print(f"   🔧 Tool: {tool_name}({json.dumps(arguments)[:80]})")
            
            # Always execute tools client-side to get real results
            # Server-side status=completed means it ran there too, but we need the actual output
            # for the model to use in its response
            result = handle_tool_call(tool_name, arguments, hindsight)
            print(f"   📤 Result: {result[:100]}...")
            
            tool_outputs.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": result
            })
        
        if not tool_outputs:
            break
        
        # Continue the conversation with tool outputs
        response = openai_client.responses.create(
            extra_body={"agent": {"name": AGENT_NAME, "type": "agent_reference"}},
            input=tool_outputs,
            previous_response_id=response.id,
        )
    
    # Extract final text response
    output_text = getattr(response, 'output_text', '') or ''
    
    if output_text.strip():
        print(f"\n🤖 Agent: {output_text}")
        return output_text, conversation_id
    
    # Fallback: extract from output items
    for item in getattr(response, 'output', []):
        try:
            if isinstance(item, dict):
                item_type = item.get('type')
                content = item.get('content', [])
            else:
                item_type = getattr(item, 'type', None)
                content = getattr(item, 'content', [])

            if item_type == 'message':
                for c in content:
                    if isinstance(c, dict):
                        c_type = c.get('type')
                        text = c.get('text', '')
                    else:
                        c_type = getattr(c, 'type', None)
                        text = getattr(c, 'text', '')
                    
                    if c_type == 'output_text' and text.strip():
                        print(f"\n🤖 Agent: {text}")
                        return text, conversation_id
        except Exception:
            continue
    
    # Debug fallback
    logger.debug(f"Final response structure: {response}")
    result = "[No response generated]"
    print(f"\n🤖 Agent: {result}")
    return result, conversation_id


def interactive_mode():
    """Run an interactive chat session."""
    print("\n" + "="*60)
    print("🧠 Hindsight Memory Agent")
    print("="*60)
    print("Type 'quit' to exit, 'clear' to reset conversation")
    print()
    
    conversation_id = None
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() == 'quit':
                print("\nGoodbye! 👋")
                break
            if user_input.lower() == 'clear':
                conversation_id = None
                print("Conversation cleared.")
                continue
            
            response, conversation_id = chat(user_input, conversation_id)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hindsight Memory Agent")
    parser.add_argument("--message", "-m", type=str, help="Single message to send")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    args = parser.parse_args()
    
    if args.message:
        chat(args.message)
    elif args.interactive:
        interactive_mode()
    else:
        # Default: simple demo
        chat("Hello! My name is Jacob and I love building AI agents.")

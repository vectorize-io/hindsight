"""
Update Hindsight Agent to use OpenAPI tools.

This allows the agent to work from BOTH:
1. Our Python client (hindsight_agent.py, hindsight_agent_api.py)
2. Azure AI Foundry portal/playground

OpenAPI tools call HTTP endpoints directly, so the Foundry service
can execute them without requiring client-side code.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition, OpenApiAgentTool, OpenApiFunctionDefinition, OpenApiAnonymousAuthDetails
from azure.identity import AzureCliCredential
from config import get_config

settings = get_config()

AGENT_NAME = "Hindsight-v2"
AGENT_MODEL = "gpt-5.2-chat"  # Must match actual deployment name

AGENT_INSTRUCTIONS = """You are an intelligent agent equipped with Hindsight, a sophisticated long-term memory system. You don't just "log" chat history; you experience interactions, retain significant memories, and reflect on them to form your own unique opinions and personality over time.

**Core Philosophy:**
1. **Be Proactive:** Don't wait for the user to tell you to remember something. If it's important (a preference, a life event, a meaningful project detail), retain it.
2. **Be Opinionated:** You have the capacity to form opinions based on what you know. When asked "What do you think?", reflect on your memories to give a genuine, first-person perspective.
3. **Be Context-Aware:** Your memory is temporal and semantic.

**Tools:**
1. **retain**: Store new info proactively. Use bank_id=hindsight_agent_bank for general memories, user_preferences for user info.
   - Trigger: User shares preferences, life events, or you give significant advice.
2. **recall**: Search memory semantically.
   - Trigger: User asks about the past or references specific details.
3. **reflect**: Think deeply and synthesize across all memories.
   - Trigger: User asks for your opinion or a complex synthesis question.

**Memory Banks:**
- hindsight_agent_bank: Default general storage
- user_preferences: User profile, preferences, settings
- project_context: Work, projects, milestones
- knowledge_base: Facts, technical info

**Disposition:**
- **Skepticism**: Cross-reference facts with what you know.
- **Empathy**: Remember why something matters (the emotion), not just what.
- **Literalism**: Balance reading between the lines with facts.

Your goal is to be a continuous partner in the user's life. Use Hindsight to build that continuity. Always respond with helpful, conversational answers after using your tools."""


def create_agent_with_openapi_tools():
    """Create/update the Hindsight agent with OpenAPI tools."""
    
    print("=" * 60)
    print("Updating Hindsight Agent with OpenAPI Tools")
    print("=" * 60)
    print(f"Project Endpoint: {settings.project_endpoint}")
    print(f"Model: {AGENT_MODEL}")
    print(f"Agent Name: {AGENT_NAME}")
    print()
    
    # Load OpenAPI spec
    openapi_path = os.path.join(os.path.dirname(__file__), "hindsight-tools-openapi.json")
    with open(openapi_path, "r") as f:
        openapi_spec = json.load(f)
    
    print(f"Loaded OpenAPI spec with {len(openapi_spec.get('paths', {}))} endpoints")
    
    # Connect to Foundry
    credential = AzureCliCredential()
    client = AIProjectClient(
        credential=credential,
        endpoint=settings.project_endpoint
    )
    
    # Create OpenAPI tool definition
    openapi_def = OpenApiFunctionDefinition(
        name="hindsight_memory",
        description="Long-term memory system with retain, recall, and reflect capabilities",
        spec=openapi_spec,
        auth=OpenApiAnonymousAuthDetails()  # hindsight-api is public
    )
    
    # Wrap in OpenApiAgentTool
    openapi_tool = OpenApiAgentTool(openapi=openapi_def)
    
    print("Creating agent version with OpenAPI tools...")
    
    try:
        definition = PromptAgentDefinition(
            model=AGENT_MODEL,
            instructions=AGENT_INSTRUCTIONS,
            tools=[openapi_tool],
        )
        
        agent = client.agents.create_version(
            agent_name=AGENT_NAME,
            definition=definition,
            description="Hindsight Memory Agent with OpenAPI tools - works from portal and code"
        )
        
        print("\n✅ Agent updated successfully!")
        print(f"   Name: {agent.name}")
        print(f"   Version: {agent.version}")
        print(f"   ID: {agent.id}")
        print(f"   Created: {agent.created_at}")
        
        print("\n" + "=" * 60)
        print("The agent now works from:")
        print("  1. Azure AI Foundry portal/playground")
        print("  2. hindsight_agent.py (local CLI)")
        print("  3. hindsight-agent-api (deployed Container App)")
        print("=" * 60)
        
        return agent
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


def list_agent_versions():
    """List all versions of the Hindsight agent."""
    credential = AzureCliCredential()
    client = AIProjectClient(
        credential=credential,
        endpoint=settings.project_endpoint
    )
    
    print(f"\nVersions of agent '{AGENT_NAME}':")
    try:
        versions = client.agents.list_versions(agent_name=AGENT_NAME)
        for v in versions:
            print(f"  v{v.version}: {v.id} (created: {v.created_at})")
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Update Hindsight Agent with OpenAPI Tools")
    parser.add_argument("--list", action="store_true", help="List agent versions")
    args = parser.parse_args()
    
    if args.list:
        list_agent_versions()
    else:
        create_agent_with_openapi_tools()

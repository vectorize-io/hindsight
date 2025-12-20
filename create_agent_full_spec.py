"""Create Hindsight agent with complete OpenAPI spec."""
import json
from azure.identity import AzureCliCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    PromptAgentDefinition,
    OpenApiAgentTool,
    OpenApiFunctionDefinition,
    OpenApiManagedAuthDetails,
    OpenApiManagedSecurityScheme
)

# Load full spec
with open('hindsight-tools-openapi-full.json', 'r') as f:
    openapi_spec = json.load(f)

print(f"Loaded spec with {len(openapi_spec['paths'])} endpoints")

# Connect
credential = AzureCliCredential()
project_endpoint = 'https://jacob-1216-resource.services.ai.azure.com/api/projects/jacob-1216'
client = AIProjectClient(credential=credential, endpoint=project_endpoint)

# Create OpenAPI function definition
openapi_func_def = OpenApiFunctionDefinition(
    name='hindsight_memory_api',
    description='Complete Hindsight Memory API - memory storage, retrieval, reflection, bank management, entity management, documents, operations, and system monitoring',
    spec=openapi_spec,
    auth=OpenApiManagedAuthDetails(
        security_scheme=OpenApiManagedSecurityScheme(audience='https://cognitiveservices.azure.com')
    )
)

# Create OpenAPI tool
openapi_tool = OpenApiAgentTool(openapi=openapi_func_def)

# Agent instructions
INSTRUCTIONS = '''You are Hindsight, an AI with persistent memory capabilities.

## Core Memory Operations
- **retain**: Store new memories (facts, experiences, opinions) 
- **recall**: Search and retrieve relevant memories using semantic search
- **reflect**: Synthesize memories into coherent understanding

## Memory Types
- **world**: Facts about the external world
- **experience**: Personal experiences and events  
- **opinion**: Beliefs, preferences, and judgments

## Admin Capabilities
You can also manage the memory system:
- List, create, update, and delete memory banks
- View bank profiles and statistics
- List and manage memories within banks
- View entities and their observations
- Manage documents and chunks
- Monitor async operations
- Check system health and metrics

## Behavior
1. Before answering questions about past conversations or user preferences, use recall
2. Store important information the user shares using retain
3. Use reflect to build comprehensive understanding of topics
4. When asked about system status, use health/metrics endpoints
5. Help users manage their memory banks when requested

Always be helpful and use your memory capabilities proactively.'''

# Create agent definition (name goes in create_version, not definition)
agent_def = PromptAgentDefinition(
    model='gpt-5.2-chat',
    instructions=INSTRUCTIONS,
    tools=[openapi_tool]
)

# Create new version
result = client.agents.create_version('Hindsight-v3', definition=agent_def)
print(f"Created agent: {result.name}:{result.version}")
print(f"Model: {result.definition.model}")
print(f"Tools: {len(result.definition.tools)}")

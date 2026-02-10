# Extensions

Extensions allow you to customize and extend Hindsight behavior without modifying core code. They enable multi-tenancy, custom authentication, additional HTTP endpoints, and operation hooks.

---

## Available Extensions

### TenantExtension

Handles multi-tenancy and API key authentication. Validates incoming requests and determines which PostgreSQL schema to use for database operations, enabling tenant isolation at the database level.

**Built-in: ApiKeyTenantExtension**

A simple implementation that validates API keys against an environment variable and uses the `public` schema for all authenticated requests.

```bash
HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.tenant:ApiKeyTenantExtension
HINDSIGHT_API_TENANT_API_KEY=your-secret-key
```

**Built-in: SupabaseTenantExtension**

Validates [Supabase](https://supabase.com) JWTs and provides multi-tenant memory isolation. Each authenticated user gets their own PostgreSQL schema (`{prefix}_{user_id}`), ensuring complete data separation. Performs local JWT verification using JWKS for optimal performance (no network call per request).

```bash
HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.supabase_tenant:SupabaseTenantExtension
HINDSIGHT_API_TENANT_SUPABASE_URL=https://your-project.supabase.co
# Optional - only needed for legacy HS256 projects or health check
HINDSIGHT_API_TENANT_SUPABASE_SERVICE_KEY=your-service-role-key
```

See the [source code](https://github.com/vectorize-io/hindsight/blob/main/hindsight-api/hindsight_api/extensions/builtin/supabase_tenant.py) for complete configuration options and implementation details.

For other multi-tenant setups with separate schemas per tenant (e.g., custom JWT-based auth), implement a custom `TenantExtension`.

---

### HttpExtension

Adds custom HTTP endpoints under the `/ext/` path prefix. Useful for adding domain-specific APIs that integrate with Hindsight's memory engine.

**No built-in implementation** - implement your own to add custom endpoints.

```bash
HINDSIGHT_API_HTTP_EXTENSION=mypackage.ext:MyHttpExtension
```

---

### OperationValidatorExtension

Hooks into retain/recall/reflect operations for validation and monitoring. Use cases include:
- Rate limiting and quota enforcement
- Permission checks and content filtering
- Audit logging and usage tracking
- Custom metrics collection

**No built-in implementation** - implement your own based on your requirements.

```bash
HINDSIGHT_API_OPERATION_VALIDATOR_EXTENSION=mypackage.validators:MyValidator
```

---

### MCPExtension

Registers additional MCP (Model Context Protocol) tools on the Hindsight MCP server. Enables external packages to add custom tools without modifying core code.

**No built-in implementation** - implement your own to add custom MCP tools.

```bash
HINDSIGHT_API_MCP_EXTENSION=mypackage.mcp:MyMCPExtension
```

---

## Writing Custom Extensions

### Extension Basics

Extensions are Python classes loaded via environment variables:

```bash
HINDSIGHT_API_<TYPE>_EXTENSION=mypackage.module:MyExtensionClass
```

Configuration is passed via prefixed environment variables:

```bash
HINDSIGHT_API_<TYPE>_SOME_CONFIG=value
# Extension receives: {"some_config": "value"}
```

All extensions support lifecycle hooks:
- `on_startup()` - Called when the application starts
- `on_shutdown()` - Called when the application shuts down

Extensions have access to an `ExtensionContext` that provides:
- `run_migration(schema)` - Run database migrations for a schema
- `get_memory_engine()` - Get the MemoryEngine interface

### Example: Custom TenantExtension with JWT

```python
import jwt
from hindsight_api.extensions import TenantExtension, TenantContext, AuthenticationError

class JwtTenantExtension(TenantExtension):
    def __init__(self, config: dict[str, str]):
        super().__init__(config)
        self.jwt_secret = config.get("jwt_secret")
        if not self.jwt_secret:
            raise ValueError("HINDSIGHT_API_TENANT_JWT_SECRET is required")

    async def authenticate(self, context: RequestContext) -> TenantContext:
        token = context.api_key
        if not token:
            raise AuthenticationError("Bearer token required")

        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            tenant_id = payload.get("tenant_id")
            if not tenant_id:
                raise AuthenticationError("Missing tenant_id in token")
            return TenantContext(schema_name=f"tenant_{tenant_id}")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(str(e))
```

### Example: Custom HttpExtension

```python
from fastapi import APIRouter
from hindsight_api.extensions import HttpExtension

class MyHttpExtension(HttpExtension):
    def get_router(self, memory: MemoryEngine) -> APIRouter:
        router = APIRouter()

        @router.get("/hello")
        async def hello():
            return {"message": "Hello from extension!"}

        @router.post("/custom/{bank_id}/action")
        async def custom_action(bank_id: str):
            # Access memory engine for database operations
            pool = await memory._get_pool()
            # ... custom logic
            return {"status": "ok"}

        return router
```

Routes are available at `/ext/hello`, `/ext/custom/{bank_id}/action`, etc.

### Example: Custom OperationValidatorExtension

```python
from hindsight_api.extensions import (
    OperationValidatorExtension,
    ValidationResult,
    RetainContext,
    RecallContext,
    ReflectContext,
    RetainResult,
)

class MyValidator(OperationValidatorExtension):
    # Pre-operation validation (required)
    async def validate_retain(self, ctx: RetainContext) -> ValidationResult:
        # Implement your validation logic
        return ValidationResult.accept()
        # Or reject: return ValidationResult.reject("Reason")

    async def validate_recall(self, ctx: RecallContext) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_reflect(self, ctx: ReflectContext) -> ValidationResult:
        return ValidationResult.accept()

    # Post-operation hooks (optional)
    async def on_retain_complete(self, result: RetainResult) -> None:
        # Log usage, update metrics, send notifications, etc.
        pass
```

### Example: Custom MCPExtension

```python
from mcp.server.fastmcp import FastMCP
from hindsight_api.extensions import MCPExtension
from hindsight_api.engine import MemoryEngine

class MyMCPExtension(MCPExtension):
    async def register_tools(self, mcp: FastMCP, memory: MemoryEngine) -> None:
        @mcp.tool()
        async def custom_search(query: str) -> str:
            """Custom MCP tool for specialized search."""
            # Access memory engine for operations
            pool = await memory._get_pool()
            # ... custom logic
            return f"Results for: {query}"
```

---

## Deploying Custom Extensions

### With Docker

Mount your extension package as a volume and set the environment variable:

```yaml
# docker-compose.yml
services:
  hindsight-api:
    image: vectorize/hindsight-api:latest
    volumes:
      - ./my_extensions:/app/my_extensions
    environment:
      - HINDSIGHT_API_TENANT_EXTENSION=my_extensions.auth:JwtTenantExtension
      - HINDSIGHT_API_TENANT_JWT_SECRET=${JWT_SECRET}
      - PYTHONPATH=/app
```

Or build a custom image with your extensions:

```dockerfile
FROM vectorize/hindsight-api:latest
COPY my_extensions /app/my_extensions
ENV PYTHONPATH=/app
```

### Bare Metal

Install your extension package in the same Python environment as Hindsight:

```bash
# Install Hindsight
pip install hindsight-api

# Install your extension package
pip install ./my-extensions
# or
pip install my-extensions-package

# Configure
export HINDSIGHT_API_TENANT_EXTENSION=my_extensions.auth:JwtTenantExtension
export HINDSIGHT_API_TENANT_JWT_SECRET=your-secret

# Run
hindsight-api
```

---

## Contributing Extensions

Custom extensions that solve common use cases are welcome contributions to the Hindsight project. If you've built an extension for:

- Authentication providers (OAuth, SAML, API gateways)
- Rate limiting or quota management
- Audit logging integrations
- Metrics exporters (Datadog, New Relic, etc.)
- Custom HTTP endpoints for specific platforms

Consider contributing it to the `hindsight_api.extensions.builtin` package. Open an issue or pull request on [GitHub](https://github.com/vectorize-io/hindsight) to discuss your extension.

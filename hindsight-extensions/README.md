# Hindsight Extensions

Extensions customise the Hindsight API server without forking it: multi-tenancy and
authentication, extra HTTP endpoints, extra MCP tools, and hooks around
retain/recall/reflect. They are ordinary Python packages that the server imports by
path at startup.

This directory is the **registry**. Each subdirectory is an extension that lives
outside the server, so installing Hindsight does not drag in a third-party vendor's
client library and changing an extension does not require a Hindsight release.

Extensions here are not published to PyPI. You ship one by building an image on top of
Hindsight that copies the extension in — see [Packaging](#packaging-an-extension).

## Registry

| Extension | Slot | What it does |
| --- | --- | --- |
| [`supabase-tenant`](./supabase-tenant) | `TENANT` | Validates [Supabase](https://supabase.com) Auth JWTs and gives each user their own Postgres schema |

Extensions maintained outside this repository can be listed here too — open a PR
adding a row that links to yours.

### What stays in the server

Two extensions ship with `hindsight-api-slim` because they add no dependencies and
any deployment may want them:

- `hindsight_api.extensions.builtin.tenant:ApiKeyTenantExtension` — single shared API
  key, single schema.
- `hindsight_api.extensions.builtin.memory_defense_regex:MemoryDefenseRegexExtension`
  — regex-based memory defense policies.

Anything that talks to a specific vendor, identity provider, or deployment style
belongs here instead.

---

## Extension slots

The server loads at most one extension per slot, each from its own environment
variable in `module.path:ClassName` form:

| Slot | Environment variable | Base class |
| --- | --- | --- |
| Tenancy / auth | `HINDSIGHT_API_TENANT_EXTENSION` | `TenantExtension` |
| HTTP endpoints | `HINDSIGHT_API_HTTP_EXTENSION` | `HttpExtension` |
| MCP tools | `HINDSIGHT_API_MCP_EXTENSION` | `MCPExtension` |
| Operation hooks | `HINDSIGHT_API_OPERATION_VALIDATOR_EXTENSION` | `OperationValidatorExtension` |
| Memory defense | `HINDSIGHT_API_MEMORY_DEFENSE_EXTENSION` | `MemoryDefenseExtension` |

Every other environment variable sharing the slot's prefix becomes the extension's
config, lowercased and with the prefix stripped. For the `TENANT` slot:

```bash
HINDSIGHT_API_TENANT_EXTENSION=hindsight_ext_supabase_tenant:SupabaseTenantExtension
HINDSIGHT_API_TENANT_SUPABASE_URL=https://xxx.supabase.co   # -> config["supabase_url"]
HINDSIGHT_API_TENANT_SCHEMA_PREFIX=user                     # -> config["schema_prefix"]
```

There is nothing to register: if the class is importable in the server's Python
environment and subclasses the slot's base class, it loads.

---

## Writing an extension

```python
# hindsight_ext_myauth/extension.py
from hindsight_api.extensions.tenant import AuthenticationError, Tenant, TenantContext, TenantExtension
from hindsight_api.models import RequestContext


class MyTenantExtension(TenantExtension):
    def __init__(self, config: dict[str, str]) -> None:
        super().__init__(config)
        self.secret = config.get("secret")
        if not self.secret:
            raise ValueError("HINDSIGHT_API_TENANT_SECRET is required")

    async def on_startup(self) -> None:
        """Open clients, warm caches. Raise to stop the server booting misconfigured."""

    async def authenticate(self, context: RequestContext) -> TenantContext:
        if context.api_key != self.secret:
            raise AuthenticationError("Invalid API key")
        return TenantContext(schema_name="my_tenant")

    async def list_tenants(self) -> list[Tenant]:
        """Schemas the background worker should process."""
        return [Tenant(schema="my_tenant")]

    async def on_shutdown(self) -> None:
        """Close what on_startup opened."""
```

Things worth knowing before you write one:

- **Validate config in `__init__`.** A misconfigured extension should fail the server's
  startup, not the first request that hits it.
- **`self.context`** is an `ExtensionContext`, the supported API into the server. For
  tenant extensions the important call is `await self.context.run_migration(schema)`,
  which provisions a new tenant schema. Cache the schemas you have already migrated —
  `authenticate` runs on every request.
- **`list_tenants()` drives the background worker.** A schema you never return gets no
  consolidation or maintenance, so returning only schemas seen since the last restart
  means tenants go stale until they are used again.
- **Extensions run in-process, inside the auth boundary.** A `TenantExtension` decides
  which tenant's data a request can reach; treat schema names derived from user input
  as untrusted and validate their shape before they reach a schema name.

The interfaces live in `hindsight-api-slim/hindsight_api/extensions/`; each base class
documents the full method set.

---

## Packaging an extension

Layout — one directory per extension, mirroring `supabase-tenant/`:

```
hindsight-extensions/<name>/
├── pyproject.toml            # test harness only — not a published distribution
├── README.md                 # config reference + how to build an image with it
├── Dockerfile                # the image that ships it
├── hindsight_ext_<name>/
│   ├── __init__.py           # re-export the class for a short import path
│   └── extension.py
└── tests/
```

Naming keeps the env var short and unambiguous:

| | |
| --- | --- |
| Directory | `hindsight-extensions/supabase-tenant/` |
| Import package | `hindsight_ext_supabase_tenant` |
| Env value | `hindsight_ext_supabase_tenant:SupabaseTenantExtension` |

There is no version number, no wheel and no release step. The unit of distribution is
the image you build, so the extension is exactly as current as the checkout you built
it from.

### The pyproject is for tests, not packaging

The `pyproject.toml` exists so `uv run pytest` works. It declares `package = false`,
so uv puts the dependencies in a virtualenv and leaves the sources on the path without
building anything:

```toml
[project]
name = "hindsight-ext-<name>"
version = "0"
requires-python = ">=3.11"
dependencies = [
    "PyJWT[crypto]>=2.12.0",   # whatever your extension imports
    "httpx>=0.27.0",
    "hindsight-api-slim",      # test-time only: the interfaces you write against
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
]

[tool.uv]
package = false

[tool.uv.sources]
hindsight-api-slim = { path = "../../hindsight-api-slim", editable = true }

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
asyncio_mode = "auto"
```

`hindsight-api-slim` belongs here and **only** here. At runtime the server is the host
process that imports your extension, not something your extension installs — an
extension that pulled the server in as a dependency could move the server version
underneath the deployment it was being added to.

### Develop and test

From the extension directory:

```bash
uv sync                  # deps plus the server from ../../hindsight-api-slim
uv run pytest tests -v
```

Extension tests are plain unit tests — construct the class with a config dict, mock
whatever it talks to, and assert on the `TenantContext` / `ValidationResult` it
returns. They do not need a database.

To run a real server against your extension without building an image:

```bash
cd ../../hindsight-api-slim
PYTHONPATH=../hindsight-extensions/myauth \
HINDSIGHT_API_TENANT_EXTENSION=hindsight_ext_myauth:MyTenantExtension \
HINDSIGHT_API_TENANT_SECRET=dev-secret \
uv run hindsight-api
```

---

## Docker packaging

The Hindsight image does not carry extensions. Ship yours by building an image on top
of it that installs the extension's dependencies and copies the extension in.

```dockerfile
FROM ghcr.io/vectorize-io/hindsight:latest

# Install into the server's virtualenv explicitly. It was created by `uv sync`
# and ships no `pip` of its own, so a bare `pip install` would land in user
# site-packages and be invisible to the running server.
RUN uv pip install --python /app/api/.venv/bin/python --no-cache \
      'PyJWT[crypto]>=2.12.0' \
      'httpx>=0.27.0'

# /app/extensions is ours — the image does not use it — so nothing the server
# ships can be shadowed by what lands here.
COPY hindsight-extensions/supabase-tenant/hindsight_ext_supabase_tenant \
     /app/extensions/hindsight_ext_supabase_tenant
ENV PYTHONPATH=/app/extensions

# Fail the build, rather than the first authenticated request.
RUN /app/api/.venv/bin/python -c "import hindsight_ext_supabase_tenant"
```

That last `import` line is the one worth keeping: without it a packaging mistake ships
happily and only surfaces when a request reaches the extension.

Build from the repository root, so the extension sources are in the build context:

```bash
docker build -f hindsight-extensions/supabase-tenant/Dockerfile -t hindsight-with-supabase .

docker run -p 8888:8888 \
  -e HINDSIGHT_API_TENANT_EXTENSION=hindsight_ext_supabase_tenant:SupabaseTenantExtension \
  -e HINDSIGHT_API_TENANT_SUPABASE_URL=https://xxx.supabase.co \
  -e HINDSIGHT_API_DATABASE_URL=postgresql://... \
  hindsight-with-supabase
```

Use `ghcr.io/vectorize-io/hindsight:latest-slim` as the base if you do not need the
bundled local embedding/reranking models.

Same shape in `docker-compose.yml` — build the image and pass the extension's variables
as environment:

```yaml
services:
  hindsight-api:
    build:
      context: .                                                  # repository root
      dockerfile: hindsight-extensions/supabase-tenant/Dockerfile
    environment:
      HINDSIGHT_API_TENANT_EXTENSION: hindsight_ext_supabase_tenant:SupabaseTenantExtension
      HINDSIGHT_API_TENANT_SUPABASE_URL: https://xxx.supabase.co
```

A worker deployment loads the same tenant extension as the API, so give both containers
the identical extension variables — otherwise the worker cannot enumerate tenant
schemas and background consolidation stops for every tenant.

> Do not install extensions at container start (an entrypoint that runs `pip install`).
> That resolves unpinned code over the network into a running server on every restart.

---

## Contributing an extension

Open a PR adding `hindsight-extensions/<name>/` with the layout above:

- a `README.md` documenting every environment variable it reads,
- tests that exercise the extension through its base-class interface,
- a `Dockerfile` that builds an image with it,
- a row in the registry table above.

Extensions here are owned by their contributors. If you would rather host yours
yourself, add a registry row pointing at your repository and package — no code needs
to live in this tree.

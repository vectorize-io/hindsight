# Hindsight Extensions

Extensions customise the Hindsight API server without forking it: multi-tenancy and
authentication, extra HTTP endpoints, extra MCP tools, and hooks around
retain/recall/reflect. They are ordinary Python packages that the server imports by
path at startup.

This directory is the **registry**. Each subdirectory is an extension distributed
separately from the server, so installing Hindsight does not drag in a third-party
vendor's client library, and shipping an extension does not require a Hindsight
release.

## Registry

| Extension | Slot | Package | What it does |
| --- | --- | --- | --- |
| [`supabase-tenant`](./supabase-tenant) | `TENANT` | `hindsight-ext-supabase-tenant` | Validates [Supabase](https://supabase.com) Auth JWTs and gives each user their own Postgres schema |

Extensions maintained outside this repository can be listed here too — open a PR
adding a row that links to your repository and package.

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
├── pyproject.toml            # distribution: hindsight-ext-<name>
├── README.md                 # config reference + install
├── Dockerfile                # example derived image
├── hindsight_ext_<name>/
│   ├── __init__.py           # re-export the class for a short import path
│   └── extension.py
└── tests/
```

Naming keeps the env var short and unambiguous:

| | |
| --- | --- |
| Directory | `hindsight-extensions/supabase-tenant/` |
| Distribution | `hindsight-ext-supabase-tenant` |
| Import package | `hindsight_ext_supabase_tenant` |
| Env value | `hindsight_ext_supabase_tenant:SupabaseTenantExtension` |

### Depend on the server only for development

**Do not put `hindsight-api-slim` in `dependencies`.** The server is the host process
that imports your extension, not something your extension installs. Declaring it means
a `pip install` of your extension into a running deployment can silently upgrade or
downgrade the server it is being added to.

Declare only what your own code imports, and take the server as a dev extra:

```toml
dependencies = [
    "PyJWT[crypto]>=2.12.0",
    "httpx>=0.27.0",
]

[project.optional-dependencies]
dev = ["hindsight-api-slim>=0.9.2", "pytest>=7.0.0", "pytest-asyncio>=0.21.0"]

[tool.uv.sources]
# Development only — uv sources are not written into the published wheel.
hindsight-api-slim = { path = "../../hindsight-api-slim", editable = true }
```

State the server version you support in your README instead; the extension interfaces
are stable within a minor release.

### Develop and test

From the extension directory:

```bash
uv sync --extra dev      # installs the extension plus the server from ../../hindsight-api-slim
uv run pytest tests -v
```

Extension tests are plain unit tests — construct the class with a config dict, mock
whatever it talks to, and assert on the `TenantContext` / `ValidationResult` it
returns. They do not need a database.

To run a real server against it:

```bash
cd ../../hindsight-api-slim
HINDSIGHT_API_TENANT_EXTENSION=hindsight_ext_myauth:MyTenantExtension \
HINDSIGHT_API_TENANT_SECRET=dev-secret \
uv run --with-editable ../hindsight-extensions/myauth hindsight-api
```

### Publish

```bash
uv build            # -> dist/hindsight_ext_<name>-<version>-py3-none-any.whl
uv publish
```

---

## Docker packaging

The Hindsight image does not carry extensions. Build a derived image that adds yours
to the same Python environment the server runs in — `/app/api/.venv`.

Two rules make this work, both inherited from the image's layout:

1. **Install with `uv pip install --python /app/api/.venv/bin/python`.** The image's
   virtualenv was created by `uv sync` and ships no `pip` of its own, so a bare
   `pip install` lands in user site-packages where the server will never see it.
2. **Never let the install resolve `hindsight-api-slim`.** Pin the extension's own
   dependencies explicitly and add the extension itself with `--no-deps`, so a
   packaging slip cannot replace the server inside its own image.

`Dockerfile` — the published extension, installed by name:

```dockerfile
FROM ghcr.io/vectorize-io/hindsight:latest

RUN uv pip install --python /app/api/.venv/bin/python --no-cache \
      'PyJWT[crypto]>=2.12.0' 'httpx>=0.27.0' \
 && uv pip install --python /app/api/.venv/bin/python --no-cache --no-deps \
      hindsight-ext-supabase-tenant \
 && /app/api/.venv/bin/python -c "import hindsight_ext_supabase_tenant"
```

The final `import` line matters: it turns a packaging mistake into a failed build
instead of a server that crashes on its first authenticated request.

To ship an extension you have not published, `uv build` it and copy the wheel in:

```dockerfile
COPY dist/hindsight_ext_supabase_tenant-*.whl /tmp/
RUN uv pip install --python /app/api/.venv/bin/python --no-cache \
      'PyJWT[crypto]>=2.12.0' 'httpx>=0.27.0' \
 && uv pip install --python /app/api/.venv/bin/python --no-cache --no-deps /tmp/*.whl \
 && rm /tmp/*.whl \
 && /app/api/.venv/bin/python -c "import hindsight_ext_supabase_tenant"
```

Use `ghcr.io/vectorize-io/hindsight:latest-slim` as the base instead if you do not need
the bundled local embedding/reranking models.

Build and run it:

```bash
docker build -t hindsight-with-supabase .

docker run -p 8000:8000 \
  -e HINDSIGHT_API_TENANT_EXTENSION=hindsight_ext_supabase_tenant:SupabaseTenantExtension \
  -e HINDSIGHT_API_TENANT_SUPABASE_URL=https://xxx.supabase.co \
  -e HINDSIGHT_API_DATABASE_URL=postgresql://... \
  hindsight-with-supabase
```

Same shape in `docker-compose.yml` — point the service at the derived image and pass
the extension's variables as environment:

```yaml
services:
  hindsight-api:
    build: ./my-extension          # the Dockerfile above
    environment:
      HINDSIGHT_API_TENANT_EXTENSION: hindsight_ext_supabase_tenant:SupabaseTenantExtension
      HINDSIGHT_API_TENANT_SUPABASE_URL: https://xxx.supabase.co
```

A worker deployment loads the same tenant extension as the API, so give both
containers the identical extension variables — otherwise the worker cannot enumerate
tenant schemas and background consolidation stops for every tenant.

> Do not install extensions at container start (an entrypoint that runs `pip install`).
> That resolves unpinned code over the network into a running server on every restart.

---

## Contributing an extension

Open a PR adding `hindsight-extensions/<name>/` with the layout above:

- a `README.md` documenting every environment variable it reads,
- tests that exercise the extension through its base-class interface,
- a `Dockerfile` showing how to add it to the image,
- a row in the registry table above.

Extensions here are owned by their contributors. If you would rather host yours
yourself, add a registry row pointing at your repository and package — no code needs
to live in this tree.

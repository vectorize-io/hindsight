# Bank LLM routing extension

A Hindsight `LLMRequestExtension` that decides, per bank and per operation, which LLM
each call goes to and with which key, by matching the bank id against patterns in
configuration.

Two uses, mixable in one config:

- **A key per customer.** With an LLM gateway (LiteLLM proxy, Bifrost, OpenRouter,
  Portkey, ...) in front of your spend, give each customer's banks their virtual key.
  Hindsight keeps one connection to the gateway; only the key on each request changes.
- **A model per bank or per operation.** Send a bank to a different provider, model or
  account, and pick a cheaper model for retain than for reflect.

What it does:

- **Every operation:** retain, reflect, consolidation and mental-model refresh, in the
  API process and in the worker.
- **First matching route wins:** routes are bank-id globs (`acme-*`, `user-?`, `*`),
  optionally narrowed to an operation glob (`reflect`, `consolidation*`), tried in order.
- **Fails closed:** a bank no route matches is refused by default, never sent on the
  server's own LLM. Opt into the server's LLM with `UNMATCHED=server`.
- **Tenant-aware:** a route can be pinned to a tenant schema, so another tenant naming a
  bank `acme-1` is not billed to Acme's key.
- **Keys stay out of logs:** startup logs the patterns and models only, and a config
  error names the field, not the value.

## Install

Extensions are not published to PyPI. Build an image with this one in it, from the
repository root:

```bash
docker build -f hindsight-extensions/bank-llm-routing/Dockerfile -t hindsight-with-bank-llm-routing .
```

See the [Dockerfile](./Dockerfile) for what it does, and the
[packaging guide](../README.md#packaging-an-extension) for the general pattern.

To run the server outside Docker, put this directory on the `PYTHONPATH` of the
environment Hindsight runs in. There are no extra dependencies to install.

## Configure

```bash
HINDSIGHT_API_LLM_REQUEST_EXTENSION=hindsight_ext_bank_llm_routing:BankLLMRoutingExtension
HINDSIGHT_API_LLM_REQUEST_ROUTES='[
  {"bank": "acme-*", "operation": "reflect",
   "llm": {"provider": "anthropic", "model": "claude-sonnet-5-5", "api_key": "sk-ant-acme"}},
  {"bank": "acme-*",   "api_key": "sk-acme-virtual-key"},
  {"bank": "globex-*", "api_key": "sk-globex-virtual-key", "headers": {"x-team": "globex"}}
]'
HINDSIGHT_API_LLM_REQUEST_UNMATCHED=deny

# The server's own LLM — here a gateway that the key-only routes reuse.
HINDSIGHT_API_LLM_PROVIDER=openai
HINDSIGHT_API_LLM_BASE_URL=https://gateway.internal/v1
HINDSIGHT_API_LLM_MODEL=gpt-4o-mini
HINDSIGHT_API_LLM_API_KEY=sk-server-key
```

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `HINDSIGHT_API_LLM_REQUEST_ROUTES` | yes | — | JSON list of routes, tried in order (fields below) |
| `HINDSIGHT_API_LLM_REQUEST_UNMATCHED` | no | `deny` | What to do with a call no route matches: `deny` refuses it, `server` sends it on the server's own LLM |

Route fields — a route needs at least one of `llm`, `api_key`, `headers`:

| Field | Description |
| --- | --- |
| `bank` | Glob over the bank id (`fnmatch`, case-sensitive). Required |
| `schema` | Only match banks in this tenant schema |
| `operation` | Glob over the operation: `retain`, `reflect`, `consolidation*`, `mental_model_*`, ... Unset matches all |
| `provider` | Only match calls the server would send to this provider |
| `model` | Only match calls the server would send to models matching this glob |
| `llm` | Send the call to this LLM instead of the server's: `provider` and `model` (required), `base_url`, `api_key`. Unset settings inherit the server's LLM defaults, except its credentials |
| `api_key` | The key for the LLM the call goes to — the server's, or `llm`'s |
| `headers` | Extra request headers |

Give the **API and every worker** the same variables. A worker without the extension
runs consolidation and mental-model refreshes on the server's LLM.

### Multi-LLM chains

With a multi-LLM chain, or a different provider per operation, every LLM in the chain
is asked separately. A key-only route matches them all, so a gateway key would also go
to the fallback provider. Pin such routes with `provider` (and `model`):

```json
[{"bank": "acme-*", "provider": "openai", "api_key": "sk-acme-gateway"}]
```

### Multi-tenant deployments

Bank ids are unique per tenant schema only. If tenants can name their own banks, pin
every route to its schema — otherwise any tenant can create a bank called `acme-1` and
have its calls billed to Acme's key:

```json
[{"bank": "*", "schema": "tenant_acme", "api_key": "sk-acme"}]
```

### Limits

These come from the extension slot, not from this extension — see
[LLMRequestExtension](https://hindsight.vectorize.io/developer/extensions#llmrequestextension):

- A key or headers on the server's own LLM work with OpenAI-compatible endpoints,
  `anthropic` and `litellm`; for other providers, route to a whole `llm` instead.
- Up to 64 distinct `llm` configs stay built per process.
- `HINDSIGHT_API_RETAIN_BATCH_ENABLED=true` is rejected at startup.
- Embeddings and reranking keep their own configuration.
- The connection check at startup uses the server's own LLM. If it has no usable key,
  set `HINDSIGHT_API_SKIP_LLM_VERIFICATION=true`.

## Develop

```bash
uv sync
uv run pytest tests -v
```

The system-test story `hindsight-system-tests/tests/test_98_bank_llm_routing.py` runs
this extension in a real API server and a standalone worker, and checks the key and
model every LLM call arrives with.

# `xai-oauth` provider — implementation report

This branch adds **SuperGrok subscription** usage to Hindsight: a flat-rate
consumer subscription authenticated by a device-code OAuth grant, with no API
key. It is **not** xAI API support — Hindsight already reaches `api.x.ai` today
via `provider: openai` + a base URL + an API key, per-token billed. Both routes
hit the same published `https://api.x.ai/v1`, so the credential *is* the
feature; this provider is the same category as `openai-codex` (ChatGPT
subscription) and `claude-code` (Claude subscription).

Branch: `upstream/xai-oauth-provider`, cut from `5b2f5d8` (v0.9.0 main).
It supersedes `upstream/xai-grok-cli-provider` (@ `9890292`), which is kept only
as reference and is not filed. This branch never contains the `xai_grok_cli_*`
files.

---

## 1. Ported verbatim from `9890292`

Carried across with the logic unchanged (prose rewritten, see section 3):

| What | Where it lives now |
|---|---|
| `_conversation_affinity_id()` — trace-id-primary, first-message fallback, sha256 truncated to 32 hex, fail-open | `xai_oauth_llm.py` |
| The `x-grok-conv-id` application on the request | `xai_oauth_llm.py::_post` |
| `reasoning_effort` written from member config into the request body | `xai_oauth_llm.py::_build_body` |
| Retry classification skeleton (`_UpstreamStatusError` with a `retryable` flag; 408/429/5xx retryable, other 4xx not) | `xai_oauth_llm.py` |
| Non-streaming `call()` + `call_with_tools()` with `tools` passthrough and named-tool-choice validation | `xai_oauth_llm.py` |
| Usage parsing: the Pydantic response models, `_token_counts`, reasoning-token subtraction, `prompt_tokens_details.cached_tokens` | `xai_oauth_llm.py` |
| Timeout wiring: factory `timeout` → `self.timeout` → `httpx.AsyncClient`, falling back to `ENV_LLM_TIMEOUT`/`DEFAULT_LLM_TIMEOUT` | `xai_oauth_llm.py::__init__` |
| `supports_attempt_scoped_concurrency() -> True` | `xai_oauth_llm.py` |
| `_record_success` / `_record_span`, `_strip_code_fence`, `_content_of` | `xai_oauth_llm.py` |
| Logging discipline: byte counts, status codes, model name and durations only — never bodies, credentials or header values | both modules |
| The admission bar `max(skew, request_timeout)` (concept; see section 3.12) | `xai_oauth_llm.py::_admission_ttl` |

## 2. Dropped, and why

| Dropped | Why |
|---|---|
| `xai_grok_cli_auth.py` in its entirety (CLI spawn, `CliSpawnSpec`, `WarmOutcome`, windowless creation flags, `resolve_cli_binary`) | The credential is now an OAuth grant this engine obtains and refreshes over plain HTTP. No process is spawned, so none of the spawn machinery has a purpose. |
| Reading `~/.grok/auth.json` | Replaced by an owned store. A test patches `open`/`Path.open` and asserts no `.grok` path is ever opened. |
| `cli-chat-proxy.grok.com` base URL | Replaced by the published `https://api.x.ai/v1`. |
| `x-grok-client-version`, `x-grok-client-identifier`, `X-XAI-Token-Auth`, `x-grok-model-override` headers | The published API needs none of them. The impersonation class is structurally gone; a parametrised test pins that each of those headers (plus `User-Agent`) is never sent. |
| The HTTP 426 client-version-floor branch, `MIN_CLIENT_VERSION`, `parse_client_version`, `resolve_client_version` | There is no client-version floor on the published API. |
| Header-based model routing | The model travels in the request body, asserted by a test. |
| The Terms-of-Service warning block and the deployment matrix in `models.mdx` | Both existed because the old provider used an unpublished endpoint with a scraped credential. What remains is the subscription-entitlement caveat (section 6) and a much shorter deployment note. |
| The `HINDSIGHT_API_XAI_GROK_CLI_AUTH_FILE` "read-side only" caveat | The new store override relocates both the read and the write, so the caveat does not apply. |

## 3. Where the house pattern, the spec, and Hermes conflicted

Every item below is a place the three sources did not agree. The resolution and
who won is recorded so a reviewer can overrule any of them.

**3.1 Login entrypoint — spec's intent won; its literal instruction was
unsatisfiable.** The spec says to make the device-code flow "runnable the same
way codex_auth's login is runnable (mirror its entrypoint mechanism)". Reading
the house: `codex_auth.py` and `nous_auth.py` have **no login entrypoint at
all** — both delegate login to a vendor CLI (`codex auth login`, `hermes
portal`) and only read + refresh the resulting file. There is therefore nothing
to mirror, and the one vendor CLI we could delegate to is the one the spec
forbids touching. Resolution: a module `__main__` —
`python -m hindsight_api.engine.providers.xai_oauth_auth login`.

**3.2 Token store location — house *convention* won, house *directory* could
not.** The spec says to use the "same directory + file-permission convention as
`codex_auth.py`". That directory is `~/.codex`, i.e. the vendor CLI's own home;
the equivalent for us would be `~/.grok`, which is forbidden. Resolution:
`~/.hindsight/xai_oauth.json`. `~/.hindsight` is already this repo's own
convention (`daemon.py`'s `DAEMON_LOG_PATH`, `llamacpp_llm.py`'s `MODELS_DIR`),
so the conventions that matter — a dot-directory under `$HOME`, `0600`, a
temp-file + `os.replace` rewrite, an `fcntl.flock` lock beside the store — are
mirrored onto a path we own. Overridable with
`HINDSIGHT_API_XAI_OAUTH_TOKEN_PATH`.

**3.3 Refresh skew — spec won over Hermes.** The spec says to take
`codex_auth.py`'s skew if it has one. It does: 60 seconds. Hermes uses 3600 for
cron/gateway-shaped workloads that may touch the provider only every half hour.
Shipped at **60**, with `HINDSIGHT_API_XAI_OAUTH_REFRESH_SKEW_SECONDS` for the
sparse-traffic shape, and the docs say when to widen it.

**3.4 Unreadable expiry — spec won over the house pattern, deliberately
reversing it.** Both `codex_auth.py` and `nous_auth.py` return `False` from
`_token_is_stale` when expiry cannot be determined, with a stated rationale
("we'd rather use a possibly-expired token and recover via the reactive 401
path"). The spec requires the opposite: "unreadable expiry means refresh-now,
never assume-valid." Implemented per spec — `StoredCredential.seconds_left()`
returns `0.0` when `expires_at` is absent — and tested both at the unit level
and through the manager.

**3.5 Expiry source — spec won.** The house decodes the access token's JWT
`exp` claim. The spec's store field list names `expiry` and `obtained_at`
explicitly, so expiry is read from the store (`expires_at`, computed from
`expires_in` at grant time) and no JWT is parsed. This removes the
base64/JWT-decode helpers the house modules carry.

**3.6 Terminal-rejection handling — spec won over Hermes.** Hermes quarantines
on the *first* terminal response with no retry. The spec says "single retry then
quarantine". Implemented literally: a 400/401/403 from the token endpoint is
retried exactly once, then the store is quarantined (tokens removed,
`last_auth_error` recorded with a code and reason and **no** credential
material) and the login remediation is raised. A test asserts exactly two POSTs
followed by the quarantine, and a sibling test proves a rejection that clears on
the retry succeeds.

**3.7 `slow_down` back-off — RFC won over Hermes.** Hermes widens the poll
interval by 1 second (capped at 30). RFC 8628 section 3.5 specifies 5 seconds.
The spec frames the poll loop as RFC 8628, so **+5** is shipped and tested.
Flagging it because it is a visible divergence from the working witness.

**3.8 Device-authorization endpoint — discovery preferred, Hermes' URL as
fallback.** The spec says to POST the discovered `device_authorization_endpoint`;
Hermes hardcodes `{issuer}/oauth2/device/code`. Implemented: read it from the
discovery document, fall back to the hardcoded value when the document omits it
(RFC 8628 registers the metadata key but does not require issuers to publish
it). Both paths are origin-validated. **Assumption flagged:** which of the two
xAI actually serves has not been observed — the live probe in section 6 settles
it.

**3.9 The three-way 403 — my reading, flagged for review.** The spec's section 5
defines two 403 branches (spending-limit, and "tier/entitlement denial (no
spending-limit code)") but its section 7 test list asks for "403 spending-limit
vs 403 entitlement vs generic 403: three distinct classifications". Taken
literally the second bullet is a catch-all and there is no third. Resolution:
spending-limit code → `XaiOAuthQuotaExhaustedError`; 403 carrying **no** error
code → `XaiOAuthEntitlementError` with the tier remediation (this is the shape
Hermes #26847 describes); 403 carrying **some other** explicit code →
non-retryable upstream error naming that code. Three types, none retryable, none
refreshing. If the intent was a two-way split, collapsing the third into the
second is a one-line change.

**3.10 Endpoint origin pinning — added, not requested.** Ported from Hermes'
`_xai_validate_oauth_endpoint`: the discovered `token_endpoint` is cached in the
store and every future refresh posts the refresh token to it, so one substituted
discovery response would be a standing credential leak. Scheme must be HTTPS and
host must be `x.ai` or a `*.x.ai` subdomain, enforced on both the discovery
result and the cached value. Not in the spec; included because it guards the
credential path, and tested.

**3.11 Construction does not read the store — divergence from the house.**
`CodexLLM.__init__` loads the credential eagerly and raises a configuration
error at startup. This provider defers to the first call, so a deployment whose
xAI credential needs a login does not take down an unrelated lane at process
start. The cost is that a missing credential surfaces on first use rather than at
boot. Flagged as a deliberate difference.

**3.12 `_admission_ttl` — ported although section 5 did not list it.** Section 5
enumerates what to port and names "timeout wiring" but not the admission bar.
The old branch's `max(skew, request_timeout)` rule is the other half of that
wiring (it stops a long request starting on a token that expires mid-flight), so
it is carried and tested. Its docstring is rewritten free of the banned
vocabulary.

**3.13 `env.example` — spec named a file the diff it points at never
touched.** Section 6 says to document the envs "in the same place the old branch
documented its envs (configuration.md + env.example, mirroring its diff)". The
old branch's diff touches `config.py`, `llm_wrapper.py`, `configuration.md`,
`models.mdx` and `llmProviders.json` — **not** `env.example` or `.env.example`.
Checking the repo, no subscription provider (`openai-codex`, `claude-code`,
`nous`) appears in `.env.example` either. Mirroring the actual diff won over the
literal filename; recorded as not-updated in section 5.

**3.14 Affinity byte-parity is tested against a restated derivation.** The spec
asks for byte-parity with `engine/cache_affinity.py:131-149` on the
`hindsight-cache-affinity` branch. That module does not exist in this tree, so
the test cannot import it. The test restates the derivation independently
(same shape guard, same `json.dumps(..., sort_keys=True, ensure_ascii=False,
default=str)`, same sha256 truncation) and asserts equality across
list / multi-message / unicode / empty-dict / string / dict / empty / None /
non-dict-element inputs, plus the 32-lowercase-hex shape and the trace-id path.
If the two branches ever land together, replace the local restatement with a
direct import.

## 4. Test evidence

`hindsight-api-slim/tests/test_xai_oauth_llm.py` — **82 passed, 1 skipped**
(the `0600` file-mode assertion is skipped on Windows; it runs on CI's Linux).
No test touches the network, a real home directory, or a real clock: HTTP is a
hand-rolled fake at the transport seam (house style — this repo uses no
respx/vcr), and the device-flow timing rules are driven through injected
`sleeper`/`monotonic` callables.

**Full suite, same interpreter, this environment (no database — hence the
constant 1237 errors, identical before and after):**

| | failed | passed | skipped | xfailed | errors | rerun |
|---|---|---|---|---|---|---|
| `5b2f5d8` (baseline) | 58 | 3677 | 230 | 4 | 1237 | 59 |
| this branch | 57 | 3760 | 231 | 4 | 1237 | 59 |

The failure count is not stable in this environment. Two further runs (baseline
in an isolated `git worktree`, and this branch) produced 57 and 55 failures on
unchanged trees, so the counts above are compared as **sets**, not totals:

- Failures present on this branch but not at baseline: **none**.
- Failures at baseline that now pass: `test_retain.py::test_chunks_extraction_mode`
  and `test_retain.py::test_strategy_overrides_extraction_mode_for_chunks` —
  both database/LLM-dependent, and untouched by this change.

**Guards RED-proven.** Each guard below was deliberately broken, the suite run,
and the break reverted; every one produced a failure, so none of these tests can
pass vacuously:

| Break applied | Test(s) that caught it |
|---|---|
| Recheck-under-lock removed | `test_proactive_refresh_happens_exactly_once_under_concurrent_managers` |
| Minimum-gap guard removed | `test_the_minimum_gap_suppresses_a_second_refresh`, `..._rides_the_store_not_an_in_memory_clock` |
| Minimum gap read from an in-memory clock instead of the store | same two |
| Affinity shape guard removed | 3 × `test_affinity_id_fails_open_on_every_non_conforming_shape` |
| 403 routed through the 401 refresh path | all 4 × 403-classification tests |
| Unreadable expiry treated as valid | `test_an_unreadable_expiry_refreshes_...`, `test_a_credential_without_a_recorded_expiry_...` |
| Second 401 no longer terminal | `test_a_second_401_is_terminal_with_no_third_attempt` |
| Terminal rejection no longer quarantines | `test_a_terminal_rejection_is_retried_once_then_quarantines` |

**Lint and types.** `ruff check .` and `ruff format --check .` are clean across
`hindsight-api-slim` (632 files already formatted). `ty check hindsight_api`
reports 5 diagnostics against `xai_oauth_auth.py` — all `Module fcntl has no
member ...`, produced on Windows only, and **exactly the same 5 that
`codex_auth.py` and `nous_auth.py` each produce**. CI runs on Linux, where
`fcntl` resolves.

## 5. Surfaces updated / deliberately not updated

The repo is much larger than the engine. Every surface below was checked; a
blank is never silence.

| Surface | Status | Detail |
|---|---|---|
| `hindsight-api-slim/hindsight_api/engine/providers/xai_oauth_auth.py` | **updated** (new) | Credential manager: device-code login, refresh, store, locking. |
| `hindsight-api-slim/hindsight_api/engine/providers/xai_oauth_llm.py` | **updated** (new) | The provider. |
| `hindsight-api-slim/tests/test_xai_oauth_llm.py` | **updated** (new) | 83 tests, no network. |
| `hindsight-api-slim/hindsight_api/engine/llm_wrapper.py` | **updated** | Factory branch, `_PROVIDERS_WITHOUT_API_KEY`, `valid_providers`. |
| `hindsight-api-slim/hindsight_api/config.py` | **updated** | `PROVIDER_DEFAULT_MODELS["xai-oauth"] = "grok-4.5"`. |
| `hindsight-docs/docs/developer/configuration.md` | **updated** | Provider list + a worked env block. |
| `hindsight-docs/docs/developer/models.mdx` | **updated** | Provider list + the "SuperGrok Subscription Setup" section, entitlement note, env-override table, deployment note. |
| `hindsight-docs/src/data/llmProviders.json` | **updated** | `{"id": "xai-oauth", "label": "SuperGrok (OAuth)", "iconKey": "sparkles", "defaultModel": "grok-4.5"}`. `sparkles` is an existing key in `ICON_REGISTRY`, so no new asset is needed — an unknown `iconKey` throws at build time. |
| `skills/hindsight-docs/references/developer/{configuration,models}.md` | **updated** | Generated by `scripts/generate-docs-skill.sh`, which CI re-runs and then fails on any drift, so these must be committed. See the note below — the generator is not currently runnable on Windows without corrupting unrelated files. |
| `hindsight-docs/static/openapi.json`, `skills/hindsight-docs/references/openapi.json` | **not updated** | Neither file enumerates LLM provider ids (`grep` for `openai-codex`/`claude-code`/`litellmrouter` returns 0 hits in both). The provider list is not part of the API schema, so `generate-openapi` produces no change. |
| `hindsight-control-plane/src/lib/harness-logo.ts` | **not updated — and it should not be** | This is **not** a provider→logo map. Its own header documents it as the *coding-agent harness* map: which agent wrote a document, keyed on `document_metadata.harness`, with ids emitted by `hindsight-coding-agents`. `claude-code` appears there as the Claude Code CLI harness, not as the `claude-code` LLM provider — and `openai-codex` does not appear at all (the entry is `codex`). The file states the rule explicitly: *"Do not add entries for agents that cannot appear yet: an id nothing writes is a logo nothing renders."* Nothing stamps `harness: xai-oauth`, and an entry would also need an icon asset under `public/img/harness/` plus the test that checks every entry ships its asset. Adding it would be wrong, not merely optional. |
| `hindsight-control-plane` (rest of `src/`) | **not updated** | No provider enumeration exists there — `grep` for `HINDSIGHT_API_LLM_PROVIDER`, `llmProvider`, and the existing provider ids returns nothing outside `harness-logo.ts`. Nothing to update. |
| `hindsight-cli` | **not updated** | No provider ids anywhere in it (`grep` for `openai-codex`/`claude-code` returns no hits). The CLI does not select LLM providers. |
| `.env.example` (repo root) | **not updated** | It carries API-key providers only; no subscription provider (`openai-codex`, `claude-code`, `nous`) appears in it. Adding `xai-oauth` would break that convention. See section 3.13. |
| `hindsight-embed/hindsight_embed/env.example` | **not updated** | Same shape and same reason; this is the embeddings service's env sample and this change adds no embeddings provider. |
| `hindsight-integrations/openclaw/openclaw.plugin.json` | **not updated — needs a judgment call** | It has an `llmProvider` JSON-Schema `enum`, but a curated subset: `["openai","anthropic","gemini","groq","ollama","openai-codex","claude-code"]` — 7 of 25+ providers, omitting `nous`, `fireworks`, `bedrock`, `deepseek`, `zai`, `litellm` and the rest. Which providers that integration surfaces is its maintainer's policy, not a mechanical consequence of adding a provider. **Follow-up if wanted:** add `"xai-oauth"` to both the `enum` and the `description` in that file. |
| `hindsight-docs/versioned_docs/**`, `hindsight-docs/blog/**`, `src/pages/changelog` | **not updated** | Frozen historical versions and dated posts. A new provider does not belong in 0.6/0.7 docs or in past release notes. A changelog entry for the release that ships this is the maintainers' call. |
| `helm/`, `docker/`, `monitoring/`, `hindsight-clients/` | **not updated** | No provider enumeration; the provider is selected purely by env var, which every deployment surface already passes through. |

**Generator caveat the reviewer needs.** `scripts/generate-docs-skill.sh` was run
to produce the two `skills/` files, and it is **not safe to run on Windows**: it
crashes partway on cp1252 unless `PYTHONUTF8=1` is set, and even then it emits
Windows path separators inside markdown links (`](api\memory-banks.md)` instead
of `](api/memory-banks.md)`), which rewrote 85 files this change never touched.
Those 85 were reverted and the two real files were taken from the generator
output with link separators normalised back to `/`. The committed diff under
`skills/` is 2 files, +110/-1, and every line of it is this change's own
content. **A maintainer on Linux should re-run
`./scripts/generate-docs-skill.sh` and confirm it produces no further diff.**
The Windows path-separator behaviour is a pre-existing defect in that generator,
unrelated to this branch.

## 6. Live probe — operator-run, not run here

The device-code flow requires a browser approval, so **no live call was made
from this branch**; the evidence above is entirely from the mocked suite. Two
things are unverified until someone runs this:

1. Whether xAI's discovery document publishes `device_authorization_endpoint`
   (section 3.8).
2. **The open question this whole provider hangs on:** whether this SuperGrok
   subscription tier is entitled to `api.x.ai` through an OAuth grant. Hermes
   issue #26847 reports some tiers answering HTTP 403.

Run from a checkout of this branch, on a machine with a browser:

```bash
# 1. Install the engine package (editable is fine) and log in.
#    Prints a verification URL and a user code; approve it in the browser.
python -m hindsight_api.engine.providers.xai_oauth_auth login

# 2. Confirm the store landed, owner-only, with a refresh token.
ls -l ~/.hindsight/xai_oauth.json          # expect -rw------- on POSIX
python - <<'PY'
import json, pathlib, time
d = json.loads((pathlib.Path.home() / ".hindsight" / "xai_oauth.json").read_text())
t = d.get("tokens", {})
print("access_token bytes :", len(t.get("access_token", "")))
print("refresh_token bytes:", len(t.get("refresh_token", "")))
print("scope              :", d.get("scope"))
print("token_endpoint     :", d.get("token_endpoint"))
print("expires in (s)     :", round((d.get("expires_at") or 0) - time.time()))
PY
```

Nothing above prints a token value.

```bash
# 3. One real call through the provider. This is the tier-entitlement test.
python - <<'PY'
import asyncio
from hindsight_api.engine.providers.xai_oauth_llm import (
    XaiOAuthLLM, XaiOAuthEntitlementError, XaiOAuthQuotaExhaustedError,
)

async def main():
    llm = XaiOAuthLLM(provider="xai-oauth", api_key="", base_url="",
                      model="grok-4.5", reasoning_effort="low", timeout=60.0)
    try:
        print("reply:", await llm.call(
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_completion_tokens=16, max_retries=0, scope="verification"))
        print("RESULT: entitled — this subscription reaches api.x.ai over OAuth.")
    except XaiOAuthEntitlementError as e:
        print("RESULT: NOT entitled (HTTP 403 tier gate). This is the #26847 case.")
        print(e)
    except XaiOAuthQuotaExhaustedError as e:
        print("RESULT: entitled, but the account spending limit stopped the call.")
        print(e)
    finally:
        await llm.cleanup()

asyncio.run(main())
PY
```

Expected outcomes, and what each means:

| Outcome | Meaning |
|---|---|
| `RESULT: entitled` | The provider works end to end. Report the model's reply and, if convenient, whether a second call reports non-zero `prompt_tokens_details.cached_tokens` (that would confirm `x-grok-conv-id` affinity is doing its job). |
| `RESULT: NOT entitled` | The grant is valid but this tier cannot use `api.x.ai`. That does not invalidate the code, but it *does* mean the PR should say so plainly — the entitlement note in `models.mdx` exists for exactly this. |
| `RESULT: ... spending limit` | Entitled; the account's spend cap is the blocker. |
| A login-remediation error | The store is missing or the grant was rejected. Re-run step 1. |

To re-check refresh specifically, wait until the token is inside the skew window
(or set `HINDSIGHT_API_XAI_OAUTH_REFRESH_SKEW_SECONDS` to something larger than
the remaining life) and run step 3 again: `obtained_at` in the store should move
forward and the call should still succeed.

## 7. Not done here, by design

Per the spec's non-goals: no engine cutover, no compose changes, no SSE
streaming, no PR body, and no change to the cache-affinity branch.

## 8. Post-incident fixes (2026-08-08)

A production incident hit this provider: one bad backend event turned into
20/20 sticky 502s because every retry attempt reused the single pooled
connection on the provider's one process-lifetime `httpx.AsyncClient`, while
fresh-dialing probes routed around the same event. Three bounded fixes
landed in response, none touching streaming, the engine cutover, or the
cache-affinity branch:

1. **Connection recycling on a retryable >=500.** The retry loop now drops
   the shared client and swaps in a fresh one (closing the stale client's
   pooled connections) before the next attempt, but only for a genuine >=500
   status — 4xx and 408/429 leave the connection alone, since neither is
   evidence the connection itself is bad. Covers both `call()` and
   `call_with_tools()`.
2. **Debug-gated response-header logging on non-2xx.** Behind
   `HINDSIGHT_API_XAI_OAUTH_DEBUG_HEADERS` (default off), a non-2xx reply now
   logs its status plus an allowlist of routing/diagnostic headers (`via`,
   `x-request-id`, `cf-ray`, `server`, `date`). Never fires on a 2xx reply,
   never logs the request's own headers (the bearer token lives there), and
   never logs a body.
3. **Usage-accounting fix: `output_tokens` reading 0 on real completions.**
   xAI does not reliably fold `reasoning_tokens` into `completion_tokens` the
   way the OpenAI o1/o3 contract does; when `completion_tokens` is already
   visible-only, subtracting `reasoning_tokens` from it a second time could
   clamp a real completion down to 0 visible output. `output_tokens` is now
   derived from `total_tokens` (which reads as prompt + visible + reasoning
   under both shapes) instead of from the ambiguous `completion_tokens`
   field.

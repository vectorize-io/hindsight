# Hindsight for Zapier

A [Zapier](https://zapier.com) app that brings [Hindsight](https://hindsight.vectorize.io) long-term memory into your Zaps — store content, search memories, get grounded answers, and start Zaps from memory events.

Built with the [Zapier Platform CLI](https://platform.zapier.com/). The source lives in the Hindsight monorepo for versioning and CI; the app itself is published to Zapier's platform with `zapier-platform push` / `zapier-platform promote` (see [Publishing](#publishing)).

## What's included

### Actions

- **Retain Memory** (create) — store content in a memory bank (`POST /memories`).
- **Recall Memories** (search) — search a bank with a natural-language query (`POST /memories/recall`).
- **Reflect** (search) — get an LLM-synthesized, memory-grounded answer (`POST /reflect`).

### Triggers (instant, via REST Hooks)

Each subscribes to Hindsight's webhook API (`POST /webhooks`) and is removed on teardown (`DELETE /webhooks/{id}`):

- **Retain Completed** — `retain.completed`
- **Consolidation Completed** — `consolidation.completed`
- **Memory Defense Triggered** — `memory_defense.triggered`

The **Bank** field on every action/trigger is a dynamic dropdown populated from `GET /v1/default/banks` (you can also type a new bank id — banks are created on first use).

## Authentication

API-key auth. Provide:

- **API Key** — your Hindsight key (starts with `hsk_`), sent as `Authorization: Bearer <key>`. Required for Hindsight Cloud; **optional** — leave it blank — for a self-hosted instance running without authentication.
- **API URL** — defaults to Hindsight Cloud (`https://api.hindsight.vectorize.io`); set it to your own instance for self-hosted (e.g. `http://localhost:8888`).

### Self-hosted

Point **API URL** at your instance. If it runs without auth, leave **API Key** blank — no `Authorization` header is sent. Triggers also work self-hosted: they rely on your instance making an _outbound_ POST to Zapier's webhook URL, which works for any box with outbound internet (only fully air-gapped instances can't).

> Each trigger registers its webhook with a freshly generated HMAC secret and verifies the `X-Hindsight-Signature: sha256=<hmac>` header on every delivery, rejecting any payload whose signature doesn't match.

## Development

```bash
npm install
npm run validate   # zapier validate — structural check, no login needed
npm test           # mocha + nock unit tests, no network
```

Live checks against a real instance. `zapier invoke` reads credentials from a local
`.env` (gitignored) — write it directly to skip the interactive `auth start`:

```bash
# `zapier invoke` reads auth fields from .env with an `authData_` prefix.
# Hindsight Cloud:
printf "authData_apiKey=hsk_your_key\nauthData_apiUrl=https://api.hindsight.vectorize.io\n" > .env
# …or self-hosted without auth:
printf "authData_apiKey=\nauthData_apiUrl=http://localhost:8888\n" > .env

npx zapier-platform invoke auth test
npx zapier-platform invoke trigger bankList
npx zapier-platform invoke create retain --inputData '{"bank_id":"zapier-test","content":"hello"}'
npx zapier-platform invoke search recall --inputData '{"bank_id":"zapier-test","query":"hello"}'
```

> The CLI binary is `zapier-platform` (v19 renamed it from `zapier`). `npx zapier-platform …`
> uses the local devDependency, so no global install or PATH setup is needed.

## Publishing

Requires a Zapier developer account (`npx zapier-platform login`); cannot run in CI without a `ZAPIER_DEPLOY_KEY`.

```bash
npx zapier-platform register "Hindsight"   # first time only — writes .zapierapprc (gitignored)
npx zapier-platform push                   # upload the current version (private/invite testing)
npx zapier-platform promote 1.0.0          # make a version the default for new users
```

Public-directory listing is a separate, manual step through Zapier's app-review (branding, descriptions, and a dedicated Hindsight Cloud test bank + API key for reviewers).

> **Release note:** this integration is **not** released via the monorepo's `release-integration.sh` / `release-integration.yml` (those handle PyPI/npm). It is published to Zapier manually. `package.json` is marked `"private": true` so it can never be accidentally `npm publish`ed. Do not add `zapier` to `VALID_INTEGRATIONS` in `release-integration.sh`.

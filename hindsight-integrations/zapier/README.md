# Hindsight for Zapier

A [Zapier](https://zapier.com) app that brings [Hindsight](https://hindsight.vectorize.io) long-term memory into your Zaps — store content, search memories, get grounded answers, and start Zaps from memory events.

Built with the [Zapier Platform CLI](https://platform.zapier.com/). The source lives in the Hindsight monorepo for versioning and CI; the app itself is published to Zapier's platform with `zapier push` / `zapier promote` (see [Publishing](#publishing)).

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
- **API Key** — your Hindsight key (starts with `hsk_`), sent as `Authorization: Bearer <key>`.
- **API URL** — defaults to Hindsight Cloud (`https://api.hindsight.vectorize.io`); override for self-hosted.

> Triggers rely on Hindsight making an outbound POST to Zapier's webhook URL, so they work for Cloud and any self-hosted instance with outbound internet — but not fully air-gapped instances. Webhook deliveries are unsigned in this version; security relies on Zapier's unguessable target URL.

## Development

```bash
npm install
npm run validate   # zapier validate — structural check, no login needed
npm test           # mocha + nock unit tests, no network
```

Live checks against a real key:

```bash
export HINDSIGHT_KEY=hsk_...
zapier invoke auth test
zapier invoke trigger bankList
zapier invoke create retain
zapier invoke search recall
```

## Publishing

Requires a Zapier developer account (`zapier login`); cannot run in CI without a `ZAPIER_DEPLOY_KEY`.

```bash
zapier register "Hindsight"   # first time only — writes .zapierapprc (gitignored)
zapier push                   # upload the current version (private/invite testing)
zapier promote 1.0.0          # make a version the default for new users
```

Public-directory listing is a separate, manual step through Zapier's app-review (branding, descriptions, and a dedicated Hindsight Cloud test bank + API key for reviewers).

> **Release note:** this integration is **not** released via the monorepo's `release-integration.sh` / `release-integration.yml` (those handle PyPI/npm). It is published to Zapier manually. `package.json` is marked `"private": true` so it can never be accidentally `npm publish`ed. Do not add `zapier` to `VALID_INTEGRATIONS` in `release-integration.sh`.

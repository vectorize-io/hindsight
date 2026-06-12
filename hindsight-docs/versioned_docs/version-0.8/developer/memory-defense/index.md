---
sidebar_position: 95
---

# Memory Defense

Hindsight scrubs secrets and PII from retain content using a 44-pattern regex set. Each match is replaced with a `[REDACTED:type]` marker before content reaches memory units or the document body. The feature is configured per bank and disabled by default.

## How it works

Memory Defense is opt-in per bank. The extension is always present, but it sits dormant until you give a bank a policy that turns it on. When a policy is set, every memory the agent writes to that bank is scanned before it lands in storage. When the scanner recognizes a credential, an API key, a database connection string, or a known PII format, the matched substring is replaced with a redaction marker like `[REDACTED:github_token]`.

The scrubbed version is what actually gets stored. Memory units and document bodies persist the redacted text, so future recall responses, exports, and reflect operations never see the original secret.

A policy only affects future retain calls on the bank where it is set. Existing memories are not retroactively scanned when you add or change a policy.

## Configuring Memory Defense

Memory Defense is configured per bank via the bank's `memory_defense` config field. You can set the policy at bank creation time or update it later via `PATCH /v1/{tenant}/banks/{bank_id}/config`.

The open-source version implements the `sensitive_data` rule with two possible actions:

- **`redact`** — replace each matched secret with a `[REDACTED:type]` marker and store the scrubbed memory.
- **`block`** — drop any item that contains a match. If every item in a retain request is blocked, the call returns `422`.

A minimal policy:

```json
{
  "memory_defense": {
    "enabled": true,
    "rules": [
      { "on": "sensitive_data", "action": "redact" }
    ]
  }
}
```

Once that policy is on a bank, every retain to that bank is screened with the 44 redaction patterns documented below.

:::note Existing memories are not retroactively scanned
Enabling Memory Defense on a bank only affects future retain calls. Memories already in the bank are not re-scanned or modified when you add or change a policy. If you need to scrub a bank that already contains unredacted content, you have to re-ingest the affected memories or remove them manually.
:::

### Disabled by default

Memory Defense is off on every bank until you set a policy. A bank with no `memory_defense` field, with `enabled: false`, or with no `sensitive_data` rule is treated identically: the extension returns ALLOW and content passes through unchanged. To stop redacting on a bank that has it on, set `enabled: false` or remove the policy.

## Notifications

When an item is redacted or blocked, Hindsight fires a [`memory_defense.triggered` webhook](../api/webhooks.mdx#memory_defensetriggered) if a webhook on the bank is subscribed to that event type. The payload reports the action taken, the document ID, and which redaction patterns matched — useful for routing security alerts to a SIEM or Slack. Clean items fire no event.

The same redact/block decisions are also recorded as `memory_defense` entries in the [audit log](../configuration.md#audit-logging) (when audit logging is enabled), with the action and matched pattern labels in the entry metadata.

## Patterns covered

The 44 bundled patterns cover the categories below.

### AI and LLM providers

| Label | Catches |
|---|---|
| `anthropic_key` | `sk-ant-...` |
| `openai_key`, `openai_project_key`, `openai_admin_key` | `sk-...`, `sk-proj-...`, `sk-admin-...` |
| `google_api_key` | `AIza...` (39 chars) |
| `google_oauth_token` | `ya29.<token>` |
| `xai_key` | `xai-...` |
| `groq_key` | `gsk_...` |
| `huggingface_token` | `hf_...` |
| `replicate_token` | `r8_...` |
| `perplexity_key` | `pplx-...` |
| `databricks_token` | `dapi<hex32>` |

### Cloud providers

| Label | Catches |
|---|---|
| `aws_access_key` | `AKIA<16>` |
| `aws_session_token` | `ASIA<16>` |
| `digitalocean_token` | `dop_v1_<hex64>` |

### Source control and CI

| Label | Catches |
|---|---|
| `github_fg_pat` | `github_pat_...` |
| `github_token` | `ghp_<36>` |
| `github_app_token` | `ghs_<36>` |
| `github_user_token` | `ghu_<36>` |
| `github_refresh` | `ghr_<36>` |
| `github_oauth` | `gho_<36>` |
| `gitlab_pat` | `glpat-...` |
| `npm_token` | `npm_...` |
| `pypi_token` | `pypi-AgEIcHlwaS5vcmc...` |

### Payment processors

| Label | Catches |
|---|---|
| `stripe_secret` | `sk_live_...`, `sk_test_...` |
| `stripe_restricted` | `rk_live_...`, `rk_test_...` |
| `square_token` | `sq0...` |
| `braintree_token` | `access_token$production$...` |

### Communications and email

| Label | Catches |
|---|---|
| `slack_token` | `xoxb-`, `xoxp-`, `xoxa-`, `xoxr-` |
| `slack_webhook` | `https://hooks.slack.com/services/...` |
| `twilio_api_key` | `SK<hex32>` |
| `twilio_account_sid` | `AC<hex32>` |
| `sendgrid_key` | `SG.<22>.<43>` |
| `mailgun_key` | `key-<32>` |
| `discord_bot` | `<MNO><23>.<6>.<27>` |
| `telegram_bot` | `<8-10 digits>:<35>` |

### Commerce

| Label | Catches |
|---|---|
| `shopify_token` | `shpat_<hex32>` |

### Database connection strings

| Label | Catches |
|---|---|
| `db_url_postgres` | `postgres://user:pass@host` or `postgresql://...` |
| `db_url_mysql` | `mysql://user:pass@host` |
| `db_url_mongodb` | `mongodb://user:pass@host` or `mongodb+srv://...` |

### Private keys, JWTs, and generic credentials

| Label | Catches |
|---|---|
| `private_key_pem` | `-----BEGIN ... PRIVATE KEY-----` PEM blocks |
| `jwt` | `eyJ<header>.eyJ<payload>.<signature>` |

### PII (US defaults)

| Label | Catches |
|---|---|
| `credit_card` | 13 to 19 digits with regular separators |
| `ssn_us` | `123-45-6789` shape |

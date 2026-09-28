---
sidebar_position: 39
title: "Hermes Desktop App: Configure Hindsight Memory | Integration"
description: "Set up Hindsight as the memory provider for the Hermes desktop app — entirely in Settings. Pick a mode, paste an API key, and your agent has persistent long-term memory. No config files, no environment variables."
---

# Hermes Desktop

Configure [Hindsight](https://vectorize.io/hindsight) as the memory provider for the **[Hermes](https://github.com/NousResearch/hermes-agent) desktop app** — entirely from Settings. No `config.json`, no `.env`, no terminal. Pick a mode, paste an API key, and Hermes remembers across every session.

:::tip
Prefer the command line, or running Hermes as a CLI/gateway? See the [Hermes Agent integration](/sdks/integrations/hermes) for the `hermes memory setup` wizard, plugin architecture, and the full configuration reference.

:::note Hindsight now installs from the Hermes plugin catalog
Nous Research moved every memory provider out of the Hermes core tree. The desktop app handles this for you: it installs the catalog plugin the first time an agent starts, and your settings, bank and API key are unchanged. Nothing in this guide changes. See [How to Migrate Your Hindsight Memory in Hermes](/blog/2026/09/28/migrate-hindsight-hermes-plugin) if you also use the CLI.
:::
:::

## Setup

**1. Open Settings → Memory & Context.** In the **Memory Provider** dropdown, choose **Hindsight**.

![Selecting Hindsight as the memory provider in the Hermes desktop app's Settings → Memory & Context](/img/integrations/hermes-desktop-dropdown.png)

**2. Fill in the Hindsight settings panel.** Selecting Hindsight reveals its configuration fields:

![The Hindsight memory provider configuration panel in the Hermes desktop app](/img/integrations/hermes-desktop-config-panel.png)

| Field             | What it does                                                                                    | Default                              |
| ----------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------ |
| **Mode**          | `Cloud` (just needs an API key) or `Local External` (connect to an existing Hindsight instance) | `Cloud`                              |
| **API key**       | Authenticates with the Hindsight API — stored as a write-only secret                            | —                                    |
| **API URL**       | The Hindsight endpoint                                                                          | `https://api.hindsight.vectorize.io` |
| **Bank ID**       | Which memory bank this Hermes profile reads and writes                                          | `hermes`                             |
| **Recall budget** | How hard recall works each turn: `low` / `mid` / `high`                                         | `mid`                                |

**3. Click Save.** That's it — Hermes now has persistent long-term memory.

## Connection Modes

### Cloud (recommended)

The fast path. Choose **Cloud**, then paste an API key from [ui.hindsight.vectorize.io/connect](https://ui.hindsight.vectorize.io/connect). Nothing to host — Hindsight Cloud handles storage, extraction, and retrieval.

### Local External

Already running your own Hindsight instance (Docker or self-hosted)? Choose **Local External** and set the **API URL** to your instance (for example `http://localhost:8888`). Your memory never leaves your infrastructure.

## How Memory Is Stored

Your settings are saved to the right place automatically:

- **API key** goes to the secret store — it's never read back into the form (you'll see an _API key set_ badge once it's saved).
- **Mode, API URL, Bank ID, and Recall budget** are written to your Hermes profile config.

Each profile points at one **Bank ID**, so memory is isolated per profile and follows you across every machine that profile runs on.

## Next Steps

- **Hindsight Cloud (free):** [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io)
- **Hermes CLI / plugin setup:** [Hermes Agent integration](/sdks/integrations/hermes)
- **Hermes Agent:** [github.com/NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)

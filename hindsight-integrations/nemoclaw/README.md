# hindsight-nemoclaw

One-command setup for [Hindsight](https://hindsight.vectorize.io) persistent memory on [NemoClaw](https://nemoclaw.ai) sandboxes.

NemoClaw runs [OpenClaw](https://openclaw.ai) inside an OpenShell sandbox with strict network egress policies. This package automates the full setup: installing the `hindsight-openclaw` plugin, configuring external API mode, merging the Hindsight egress rule into your sandbox policy, and restarting the gateway.

## Quick Start

> ✨ **Recommended:** Use [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) — sign up free and get an API key instantly. No infrastructure to run.

```bash
npx @vectorize-io/hindsight-nemoclaw setup \
  --sandbox my-assistant \
  --api-token <your-api-key> \
  --bank-prefix my-sandbox
```

## Documentation

Full setup guide, pitfalls, and troubleshooting:

**[NemoClaw Integration Documentation](https://vectorize.io/hindsight/sdks/integrations/nemoclaw)**

Or see [NEMOCLAW.md](./NEMOCLAW.md) in this directory for a step-by-step walkthrough.

## CLI Reference

```
hindsight-nemoclaw setup [options]

Options:
  --sandbox <name>       NemoClaw sandbox name (required)
  --api-token <token>    Hindsight API token (required)
  --api-url <url>        Hindsight API URL (default: https://api.hindsight.vectorize.io)
  --bank-prefix <prefix> Memory bank prefix (default: "nemoclaw")
  --skip-policy          Skip sandbox network policy update
  --skip-plugin-install  Skip openclaw plugin installation
  --dry-run              Preview changes without applying
  --help                 Show help
```

## What It Does

1. **Preflight** — verifies `openshell` and `openclaw` are installed
2. **Install plugin** — runs `openclaw plugins install @vectorize-io/hindsight-openclaw`
3. **Configure plugin** — writes external API mode config to `~/.openclaw/openclaw.json`
4. **Apply policy** — reads current sandbox policy, merges Hindsight egress rule, re-applies via `openshell policy set`
5. **Restart gateway** — runs `openclaw gateway restart`

## Links

- [Hindsight Documentation](https://vectorize.io/hindsight)
- [NemoClaw](https://nemoclaw.ai)
- [OpenClaw](https://openclaw.ai)
- [GitHub Repository](https://github.com/vectorize-io/hindsight)

## License

MIT

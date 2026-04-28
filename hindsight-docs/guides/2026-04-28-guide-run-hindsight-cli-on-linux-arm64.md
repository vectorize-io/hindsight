---
title: "Run Hindsight CLI on Linux ARM64 Without Workarounds"
authors: [benfrank241]
date: 2026-04-28T14:00:00Z
tags: [cli, linux, arm64, guide]
description: "Run Hindsight CLI on Linux ARM64 using the new release asset, then configure profiles, API access, and daily memory commands on Pi or ARM hosts."
image: /img/guides/guide-run-hindsight-cli-on-linux-arm64.png
hide_table_of_contents: true
---

![Run Hindsight CLI on Linux ARM64 Without Workarounds](/img/guides/guide-run-hindsight-cli-on-linux-arm64.png)

If you want to **run Hindsight CLI on Linux ARM64**, the setup is finally straightforward. The latest release flow now ships a first class `hindsight-linux-arm64` asset, which means Raspberry Pi boxes, Graviton instances, and small ARM servers no longer need a local rebuild or an unofficial copy step just to get the CLI running. If you want the surrounding docs while you work, keep [the CLI reference](https://hindsight.vectorize.io/sdks/cli), [the installation guide](https://hindsight.vectorize.io/sdks/developer/installation), [the quickstart guide](https://hindsight.vectorize.io/sdks/developer/quickstart), and [the docs home](https://hindsight.vectorize.io) nearby.

<!-- truncate -->

## The quick answer

- Linux ARM64 is now included in the published release assets, alongside the existing AMD64 and macOS binaries.
- The fastest path is to download `hindsight-linux-arm64`, mark it executable, and move it onto your PATH.
- Once the binary is installed, the normal `configure`, `bank`, `retain`, and `recall` commands work the same way they do on other platforms.

## Why this update matters

Linux ARM64 support matters because a lot of self hosted Hindsight deployments land on exactly that hardware class. A Raspberry Pi in a closet, a cheap ARM VPS, or an AWS Graviton instance is often enough for a lightweight memory service, especially if you are following the newer [installation guidance](https://hindsight.vectorize.io/sdks/developer/installation) and sizing around the slim image or external providers.

Before this release asset was wired into the release job, the CLI itself was easy to miss even when the rest of the platform ran fine. The new asset closes that gap. It is a small release workflow change, but it makes ARM64 a real supported path instead of a near miss.

## Install the Linux ARM64 binary

Use the published release asset directly:

```bash
curl -L   -o hindsight   https://github.com/vectorize-io/hindsight/releases/latest/download/hindsight-linux-arm64

chmod +x hindsight
sudo mv hindsight /usr/local/bin/hindsight
hindsight --help
```

If you prefer to keep local tools in your home directory, move the binary into `~/.local/bin` instead and make sure that directory is on your PATH. The key check is simple: `hindsight --help` should print the command tree instead of an architecture or permission error.

## Configure the CLI for cloud or local API access

After install, point the CLI at the API you actually want to use.

```bash
# Managed cloud
hindsight configure   --api-url https://api.hindsight.vectorize.io   --api-key YOUR_API_KEY

# Or a local deployment
hindsight configure --api-url http://localhost:8888
```

If you switch between environments, use named profiles from [the CLI reference](https://hindsight.vectorize.io/sdks/cli) instead of rewriting one shared config file over and over:

```bash
hindsight profile create prod   --api-url https://api.hindsight.vectorize.io   --api-key YOUR_API_KEY

hindsight -p prod bank list
```

Environment variables still win over profile files, so CI jobs can export `HINDSIGHT_API_URL` and `HINDSIGHT_API_KEY` without fighting local defaults.

## Verify the core workflow end to end

Once the CLI is configured, test the whole path with a tiny bank and a simple memory round trip:

```bash
hindsight bank list
hindsight memory retain test-bank "Alice prefers async updates"
hindsight memory recall test-bank "How should I update Alice?"
```

If that works, the ARM64 story is done. You are using the same memory commands documented in [the retain API guide](https://hindsight.vectorize.io/sdks/api/retain), [the recall API guide](https://hindsight.vectorize.io/sdks/api/recall), and [the quickstart guide](https://hindsight.vectorize.io/sdks/developer/quickstart), just from a Linux ARM64 host.

## Troubleshooting common Linux ARM64 misses

A few failures are worth checking first:

- **`Exec format error`** usually means you downloaded the wrong asset. Double check that the filename is `hindsight-linux-arm64`, not the AMD64 build.
- **`Permission denied`** means the binary is missing execute bits. Re-run `chmod +x hindsight`.
- **Connection refused** usually means your local API is not up yet, or you pointed the CLI at the wrong host and port.
- **401 or 403 responses** usually mean the API key is missing, invalid, or aimed at the wrong Hindsight environment.

If the CLI itself works but recall is slow or the host feels tight on RAM, compare your box against the new [hardware guidance in the installation docs]({INSTALL}). That is usually a deployment sizing issue, not a CLI issue.

## FAQ

### Does this replace the install script?

No. The release asset simply makes Linux ARM64 a clean download target. If your install flow already wraps published release assets, this change is what makes ARM64 fit into that path cleanly.

### Can I use the CLI against Hindsight Cloud and a local server?

Yes. Use profiles or environment variables. That is the cleanest way to switch between cloud, staging, and local deployments.

### Is ARM64 only for development?

No. It is a sensible production target for small and medium workloads, especially if you size the API, worker, and database according to the current installation guidance.

## Next Steps

- [Hindsight Cloud](https://hindsight.vectorize.io)
- [the CLI reference](https://hindsight.vectorize.io/sdks/cli)
- [the installation guide](https://hindsight.vectorize.io/sdks/developer/installation)
- [the quickstart guide](https://hindsight.vectorize.io/sdks/developer/quickstart)
- [the docs home](https://hindsight.vectorize.io)

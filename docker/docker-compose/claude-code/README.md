# Hindsight with Claude Code (Claude Pro/Max subscription)

Run Hindsight inside Docker using the `claude-code` LLM provider, backed by
your host machine's Claude Pro or Max subscription credentials.

The standalone Hindsight Docker image ships `claude-agent-sdk` but does **not**
bundle the host `claude` CLI binary or any Claude credentials. This Compose
file bind-mounts the host's CLI install and credentials into the container so
the `claude-code` provider works without an API key.

## When to use this

- You have an active Claude Pro or Max subscription and want to use it for
  Hindsight without paying separate Anthropic API costs.
- You want a one-command `docker compose up` instead of a long `docker run`
  invocation with many flags.
- You are running on **Linux/amd64** — macOS Docker Desktop and Windows host
  paths differ and are not yet covered (please open an issue if you'd like to
  contribute a verified recipe for either).

> **Personal-use only.**  Anthropic's
> [Agent SDK documentation](https://docs.claude.com/en/api/agent-sdk/overview)
> states that third-party developers should not offer claude.ai login or rate
> limits for their products. Hindsight does **not** perform any login on your
> behalf — it uses credentials you've already authenticated via
> `claude auth login`. In January 2026, Anthropic
> [enforced restrictions](https://paddo.dev/blog/anthropic-walled-garden-crackdown/)
> against tools that spoofed the Claude Code client identity; Hindsight uses
> the official Claude Agent SDK instead.
>
> Do not deploy this configuration to shared environments or production. For
> that, use the `anthropic` provider with an API key from the
> [Anthropic Console](https://console.anthropic.com/). Usage counts against
> your Claude Pro/Max subscription limits.

## Prerequisites

- Host has `claude` CLI installed (e.g., `npm install -g @anthropics/claude-code`)
  and `claude auth login` has been run successfully.
- `~/.claude.json` and `~/.claude/.credentials.json` exist on the host.
- Host `claude` CLI version is **2.1.128 or newer** — the version bundled with
  `claude-agent-sdk` 0.5.x has a protocol incompatibility in containers, so
  the recipe overrides it with the host binary.

## Quick start

```bash
# Set your host UID/GID (defaults to 1000:1000 if unset)
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

docker compose -f docker/docker-compose/claude-code/docker-compose.yaml up -d
```

- API: http://localhost:8888
- Control Plane: http://localhost:9999

## Post-setup (one-time)

After the container starts for the first time, run these commands to fix
permissions and symlink the host `claude` binary into `$PATH`:

```bash
# Make ~/.claude writable by your UID (the CLI writes session/project state)
docker exec --user 0:0 hindsight-claude-code chown $(id -u):$(id -g) /home/hindsight/.claude
docker exec --user 0:0 hindsight-claude-code chmod 755 /home/hindsight/.claude

# Symlink the host claude binary into PATH
docker exec --user 0:0 hindsight-claude-code \
  ln -sf /home/hindsight/.local/share/claude/versions/2.1.128 /usr/local/bin/claude
```

If you set `CLAUDE_CLI_VERSION` to a version other than `2.1.128`, update the
symlink path accordingly.

## Notes on the bind-mount surface (every flag is load-bearing)

- **Host `claude` binary required** — the image ships only `claude-agent-sdk`,
  not the CLI itself.
- **SDK bundled-binary override** — the override of
  `claude_agent_sdk/_bundled/claude` works around a protocol issue in the
  bundled v2.1.121 binary inside containers. Once `claude-agent-sdk` ships
  with v2.1.128+ this override can be dropped. Set `CLAUDE_CLI_VERSION` to
  match your installed version.
- **Single-file credential mounts** — credentials are mounted as individual
  `:ro` files rather than a whole-directory `:ro` mount of `~/.claude`,
  because the CLI writes session/project state at runtime and a read-only
  directory mount silently breaks it.
- **`--user` / `user:`** — the `user: ${HOST_UID}:${HOST_GID}` pattern
  requires `chmod 755 /home/hindsight`, which is built into the image since
  v0.6.0 (see [#1481](https://github.com/vectorize-io/hindsight/issues/1481)).
- **`~/.hindsight-docker` data directory** — the pg0 data bind mount must be
  writable by your host UID (see
  [#1483](https://github.com/vectorize-io/hindsight/issues/1483)).
- **Verified** on `linux/amd64` against `ghcr.io/vectorize-io/hindsight:latest`
  v0.5.6+.

## Using a different Claude CLI version

If your host has a `claude` version other than 2.1.128, set
`CLAUDE_CLI_VERSION` before starting:

```bash
export CLAUDE_CLI_VERSION=2.2.0
docker compose -f docker/docker-compose/claude-code/docker-compose.yaml up -d
```

Then update the post-setup symlink to match:

```bash
docker exec --user 0:0 hindsight-claude-code \
  ln -sf /home/hindsight/.local/share/claude/versions/2.2.0 /usr/local/bin/claude
```

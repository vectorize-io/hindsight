# hindsight-fs

Mount a [Hindsight](https://github.com/vectorize-io/hindsight) memory bank's **knowledge base** as a live, auto-refreshing folder of markdown files on your local disk.

The knowledge base is a tree of **folders** and **pages**. hindsight-fs mirrors it one-to-one: each folder becomes a directory, each page becomes a real `.md` file (the page's [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf) document — YAML frontmatter + markdown body). A background loop re-syncs from the API on an interval, so ordinary shell tools — `ls`, `cat`, `grep`, `rg`, `find`, `fzf`, your editor, anything — just work against current memory. Think of it as a read-only filesystem view over an agent's living knowledge, in the spirit of [Supermemory's SMFS](https://supermemory.ai/docs/smfs/overview).

```
./kb/
├── profile/                 # ← folder "Profile"
│   ├── user-preferences.md  # ← page "User Preferences"
│   └── communication.md
├── policies/
│   └── billing-policy.md
├── project-status.md        # ← a root-level page
└── .hindsight-fs/           # control data (config, state, daemon log, index.md)
```

Each page file looks like:

```markdown
---
id: 0f3c…
type: knowledge-page
title: User Preferences
tags: [ui, comms]
timestamp: "2026-06-25T10:00:00Z"
---

The user prefers dark mode and async, written communication. They dislike
status meetings and want concise PR descriptions.
```

## Install

```bash
npm install -g @vectorize-io/hindsight-fs
# or run without installing:
npx @vectorize-io/hindsight-fs --help
```

Requires Node.js ≥ 18.

## Quick start

```bash
# Mirror a bank into ./memory and keep it fresh in the background
hindsight-fs start ./memory --bank my-agent --api-url http://localhost:8000 --interval 15

# Now use plain shell tools — these are real files
ls ./memory
cat ./kb/profile/user-preferences.md
grep -ril "dark mode" ./memory

# See what's going on
hindsight-fs status ./memory
hindsight-fs logs ./memory

# Stop the background refresher
hindsight-fs stop ./memory
```

## Commands

| Command         | Description                                                                                                                                      |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `mount [dir]`   | Mirror the bank into `dir` and keep it refreshed in the **foreground** (Ctrl-C to stop). `--detach` backgrounds it; `--once` does a single pass. |
| `start [dir]`   | Mount in the **background** (alias for `mount --detach`).                                                                                        |
| `stop [dir]`    | Stop the background daemon for `dir`.                                                                                                            |
| `restart [dir]` | Restart the background daemon.                                                                                                                   |
| `sync [dir]`    | Run a single refresh pass and exit.                                                                                                              |
| `status [dir]`  | Show daemon + last-sync health. `--json` for a machine-readable report; **exits non-zero when unhealthy**.                                       |
| `list`          | List the bank's knowledge-base folders + pages without writing files.                                                                            |
| `logs [dir]`    | Print the tail of the background daemon log.                                                                                                     |
| `unmount [dir]` | Stop the daemon and remove mirrored files + control data.                                                                                        |

## Options

| Flag                   | Env var                 | Default                 | Description                                               |
| ---------------------- | ----------------------- | ----------------------- | --------------------------------------------------------- |
| `-b, --bank <id>`      | `HINDSIGHT_BANK_ID`     | —                       | Bank to mirror (required).                                |
| `-u, --api-url <url>`  | `HINDSIGHT_API_URL`     | `http://localhost:8000` | API base URL.                                             |
| `-t, --token <token>`  | `HINDSIGHT_API_TOKEN`   | —                       | Bearer token, if the API requires auth.                   |
| `-i, --interval <sec>` | `HINDSIGHT_FS_INTERVAL` | `30`                    | Refresh interval in seconds.                              |
| `-d, --dir <path>`     | `HINDSIGHT_FS_DIR`      | `./hindsight-fs`        | Mount directory (overrides the positional arg).           |
| `--writable`           |                         |                         | Make mirrored files editable (default: read-only `0444`). |

Settings are remembered per mount in `<dir>/.hindsight-fs/config.json`, so after the first `start`/`mount` you can just run `hindsight-fs status ./memory` (or `sync`, `stop`) without re-passing `--bank`/`--api-url`.

## How syncing works

It's **pull-based polling**, not a push/webhook or a kernel filesystem. Each tick the engine:

1. Fetches the knowledge-base tree (`GET …/knowledge-base/tree`) and the page bundle (`GET …/knowledge-base/export`) — two requests, regardless of bank size.
2. Mirrors the tree to disk: creates folder directories, writes each page's markdown at its nested path, reconciles against disk (writes new/changed/tampered files, leaves identical files untouched), and prunes files + emptied folders for pages that no longer exist.
3. Records a per-file content hash and the last-sync time in `.hindsight-fs/state.json`.

The staleness window is therefore up to one `--interval`. There is no diffing on the wire — the whole list is fetched each tick — but only files whose **bytes actually differ** are rewritten, so disk churn is minimal.

## One-way mirror — agents can't edit it

Pages are owned by the API, so the mirror is strictly read-only at the filesystem level, enforced two ways:

- **Read-only files (default).** Mirrored files are written with mode `0444`, so an agent's in-place edit, `>>`, or editor-save fails immediately with `EACCES`. Pass `--writable` to opt out (e.g. if you want to scratch-edit locally).
- **Tamper-revert backstop.** Change detection compares the **on-disk bytes** against the freshly rendered content — not just the last API hash — so if a file drifts anyway (a force-`chmod`, an external tool), it's overwritten on the next tick and reset to `0444`. `status` and the sync log report a `reverted` count when this happens.

> A hard, instantaneous block (writes rejected at the VFS layer) would require a FUSE read-only mount; that's a heavier, kernel-extension dependency and is intentionally **not** how this works today. `0444` + revert blocks ordinary agent/editor writes without any system dependency. If you need true `EROFS` semantics, open an issue.

## Other guarantees

- **Safe on errors.** A transient API/network failure never wipes the mirror; the previous files are left in place and the error is recorded in `status`.
- **Pruning.** When a page is removed from the bank, its file (and any emptied folder) is removed on the next successful sync.
- **Atomic writes.** Files are written to a temp file and renamed, so readers never see a half-written document.
- **Quiet files.** Frontmatter carries no per-poll timestamp, so an unchanged model keeps identical bytes and mtime across refreshes — editors and file watchers stay calm.

## Monitoring & healthchecks

`status` doubles as a healthcheck. It combines two signals — is the daemon process alive, and did a sync succeed recently — into one verdict, and **exits non-zero when the mount is unhealthy**:

| `status` | Meaning                                                    | Exit |
| -------- | ---------------------------------------------------------- | ---- |
| `ok`     | daemon alive and a sync succeeded within the stale window  | `0`  |
| `failed` | daemon alive but the last sync errored (bad URL/auth/bank) | `1`  |
| `stale`  | daemon alive but no fresh sync (wedged, or never synced)   | `1`  |
| `dead`   | no daemon running                                          | `1`  |

```bash
# Human-readable (adds a "Health:" line)
hindsight-fs status ./memory

# Machine-readable + scriptable exit code
hindsight-fs status ./memory --json

# Use it as a guard / in a watchdog / container healthcheck
if ! hindsight-fs status ./memory --json >/dev/null; then
  hindsight-fs restart ./memory
fi
```

The `--json` report looks like:

```json
{
  "healthy": false,
  "status": "stale",
  "mount": "/…/memory",
  "bank": "my-agent",
  "mode": "read-only",
  "daemon": { "running": true, "pid": 78765, "startedAt": "…", "intervalSeconds": 15 },
  "lastSync": { "at": "…", "ok": true, "ageSeconds": 92, "error": null },
  "staleAfterSeconds": 45,
  "mirroredFiles": 12
}
```

A sync is "stale" once it's older than `max(interval × 3, 15s)`; override with `--stale-after <seconds>`. The same verdict is available programmatically via `computeHealth(config)`.

## Programmatic use

The package also exports its engine for embedding in other tools:

```ts
import { runSync, runLoop, resolveConfig } from "@vectorize-io/hindsight-fs";

const config = await resolveConfig({ dir: "./memory", bankId: "my-agent" }, { requireBank: true });
await runSync(config); // one pass
```

## Development

```bash
npm run build        # tsc → dist/
npm test             # vitest
```

# Canva MCP Integration — oma-slide

> Optional export/import channel connecting oma-slide decks to Canva via the Canva Remote MCP server.
> This channel is **never required** — all existing local export paths (HTML, PDF, PNG, PPTX) remain fully functional without it.

## Prerequisites

### Canva Remote MCP Server

- **Endpoint:** `https://mcp.canva.com/mcp`
- **Transport:** Streamable HTTP (remote MCP)
- **Authentication:** OAuth 2.0 via Canva Connect — user must authorize the MCP connection through their Canva account.

### MCP Client Configuration

The Canva Remote MCP server must be registered in the project's MCP client configuration.
Canva uses a **remote URL** transport — no local npm package or binary is required.

The config entry shape (inside `mcpServers`):

```json
"canva": {
  "url": "https://mcp.canva.com/mcp"
}
```

> **Note (Antigravity CLI):** Some agy versions use `serverUrl` instead of `url` for
> remote MCP servers. The auto-provisioning step detects the existing convention in
> the target config file and matches it.

#### Vendor-specific config file locations

| Vendor | Scope | Config file | Key path |
|--------|-------|-------------|----------|
| Claude / Cursor | project | `.mcp.json` | `mcpServers.canva` |
| Gemini VS Code Extension | project | `.gemini/settings.json` | `mcpServers.canva` |
| Antigravity CLI (agy) | project | `.agents/mcp_config.json` | `mcpServers.canva` |
| Antigravity CLI (agy) | user global | `~/.gemini/antigravity-cli/mcp_config.json` | `mcpServers.canva` |
| Antigravity shared | user global (IDE+CLI) | `~/.gemini/config/mcp_config.json` | `mcpServers.canva` |
| OMA shared | project | `.agents/mcp.json` | `mcpServers.canva` |

All config files use the same `{ "url": "https://mcp.canva.com/mcp" }` shape
(or `{ "serverUrl": ... }` for Antigravity if that convention is detected).

### Minimum Canva Plan

- **Free / Pro:** `create_design`, `upload_asset`, `export_design`, `list_designs`, `import_design` are available.
- **Enterprise:** Adds `autofill_design` (brand template autofill). Not required for oma-slide integration.

---

## Canva MCP Tool Mapping

| oma-slide Operation | Canva MCP Tool | Direction | Notes |
|---|---|---|---|
| Verify auth / connectivity | `list_designs` | probe | Returns designs if authed; errors if not |
| Upload slide images | `upload_asset` | push | Accepts PNG/JPG; returns `asset_id` |
| Create Canva presentation | `create_design` | push | Type: `Presentation`; attach uploaded assets as pages |
| Export from Canva to file | `export_design` | pull | Formats: PDF, PNG, JPG, PPTX, MP4, GIF |
| Import a Canva design | `list_designs` → `export_design` | pull | Export as PPTX → `oma slide import-pptx` |
| Browse Canva library | `list_designs` | read | Filter by query or folder |
| Get design metadata | `get_design` | read | Title, pages, dimensions, timestamps |

---

## Export Pipeline: oma-slide → Canva

**Trigger:** User requests Canva export during Phase 6 (Deliver), or says "export to Canva" / "캔바로 내보내기".

### Steps

```
1. PROBE        → list_designs (verify Canva MCP is connected + authenticated)
2. RENDER       → oma slide png --dir <slug> --out-dir <slug>/out/png/ --resolution 2x
3. UPLOAD       → upload_asset for each slide PNG → collect asset_ids[]
4. CREATE       → create_design (type: "Presentation", assets: asset_ids[])
5. REPORT       → include Canva design URL in delivery summary
```

### Step Details

**Step 1 — Probe:**
Call `list_designs` with a minimal query. If it errors (401/403/timeout), notify the user:
> "Canva MCP is not connected or not authenticated. Skipping Canva export. Local exports are available."

Do NOT retry or prompt for credentials — the OAuth flow is handled externally.

**Step 2 — Render PNGs:**
Use `oma slide png --resolution 2x` to produce high-resolution per-slide images (3840×2160).
These become the raster backing for each Canva presentation page.

**Step 3 — Upload Assets:**
For each `slide-NN.png` in the output directory:
- Call `upload_asset` with the file path.
- Record the returned `asset_id`.
- On individual upload failure: log, skip that slide, continue with remaining.

**Step 4 — Create Presentation:**
Call `create_design` with:
- `type`: `"Presentation"` (or equivalent Canva preset type)
- Uploaded assets mapped as slide pages in `meta.json` order.

**Step 5 — Report:**
Include in the Phase 6c delivery summary:
- Canva design URL
- Number of slides successfully pushed
- Any skipped slides (upload failures)

### Limitations

> [!IMPORTANT]
> Canva export via this pipeline produces **raster-backed slides** (PNG images per page).
> Text is NOT editable in Canva. For editable Canva presentations, export PPTX first
> (`oma slide pptx`) and use Canva's native PPTX import UI manually.

---

## Import Pipeline: Canva → oma-slide

**Trigger:** User provides a Canva design URL/ID, or says "import from Canva" / "캔바에서 가져오기".
Detected in Phase 0 as `import-canva` mode.

### Steps

```
1. PROBE        → list_designs (verify connectivity)
2. IDENTIFY     → parse design ID from user input (URL or raw ID)
3. EXPORT       → export_design (format: PPTX) → download to workdir
4. IMPORT       → oma slide import-pptx <downloaded.pptx> --dir <slug>
5. CONTINUE     → proceed to Phase 3 (generate/enhance with style overlay)
```

### Step Details

**Step 1–2 — Probe + Identify:**
Extract the Canva design ID from the user's input. Accept formats:
- Full URL: `https://www.canva.com/design/DAF.../edit`
- Raw ID: `DAF...`

**Step 3 — Export from Canva:**
Call `export_design` with format `PPTX`. This is an asynchronous operation —
the Canva MCP may return a job ID; poll or await completion per MCP protocol.

Download the exported file to `<workdir>/imports/`.

**Step 4 — Import via CLI:**
Run `oma slide import-pptx <file> --dir <slug>` to extract slide fragments
into the working directory.

**Step 5 — Continue:**
The imported fragments become the generation base. The user picks a style
(Phase 2), and the skill overlays the chosen design on top.

---

## Browse Pipeline: Canva Library

**Trigger:** User says "show my Canva designs" / "캔바 디자인 목록" before providing a specific design.

### Steps

```
1. PROBE        → list_designs (verify connectivity)
2. LIST         → list_designs (optional: filter by query)
3. PRESENT      → show design titles + thumbnails to user
4. SELECT       → user picks a design → proceed to Import Pipeline
```

---

## Error Handling

| Error | Response |
|---|---|
| Canva MCP server not configured | Offer to auto-provision (see §Auto-Provisioning); skip if user declines |
| OAuth not authorized (401/403) | Notify: "Canva is not authenticated. Please connect your Canva account to the MCP server." Skip Canva ops. |
| `upload_asset` fails for one slide | Log warning; skip that slide; continue uploading remaining slides |
| `create_design` fails | Notify user; fall back to local exports (HTML/PDF/PNG/PPTX) |
| `export_design` timeout | Retry once after 10s; on second failure, notify and abort Canva import |
| Design ID not found | Notify: "Design not found in your Canva account." Offer to `list_designs` instead. |

### Graceful Degradation Priority

```
Canva MCP available + authed       →  full Canva export/import
Canva MCP available + unauthed     →  notify user; local exports only
Canva MCP not configured + user ok →  auto-provision config → retry probe
Canva MCP not configured + decline →  silent skip; local exports only
```

---

## Auto-Provisioning: Canva MCP Setup

When the skill detects that Canva MCP is not configured and the user has requested a Canva
operation, the skill **offers** to add the Canva MCP entry to the project config files.

### Detection

The skill checks for the `canva` key in `mcpServers` across known config files:

```
Project-level:
1. .agents/mcp.json                               (OMA shared SSOT)
2. .agents/mcp_config.json                         (Antigravity CLI project-scoped config)
3. .mcp.json                                       (Claude / Cursor project config)
4. .gemini/settings.json                            (Gemini VS Code Extension project settings)

User-global (optional, only if project-level is absent):
5. ~/.gemini/antigravity-cli/mcp_config.json        (agy CLI global config)
6. ~/.gemini/config/mcp_config.json                 (Antigravity shared IDE+CLI config)
```

If `canva` is absent from **all** project-level files, the skill surfaces a setup prompt.
User-global files are checked as a fallback — if Canva is configured globally but not
in the project, the skill notifies instead of re-provisioning.

### Setup Prompt

Ask the user:
> "Canva MCP is not configured in this project. Would you like me to add it to your
>  MCP config files? This adds `{ "url": "https://mcp.canva.com/mcp" }` to
>  mcpServers — no local packages are installed."

Options:
- **Yes, add to all config files** — write to all detected config files
- **Yes, add to current vendor only** — write to the active vendor's config only
- **No, skip Canva** — proceed without Canva; use local exports

### Provisioning Steps

For each target config file:

1. **Read** the existing JSON file.
2. **Parse** and verify it has a `mcpServers` object.
3. **Add** the `canva` entry:
   ```json
   "canva": {
     "url": "https://mcp.canva.com/mcp"
   }
   ```
4. **Write** the updated JSON back with the same formatting (2-space indent).
5. **Verify** the file is valid JSON after write.

### Post-Provisioning

After writing config files:

1. **Notify the user** that a session restart may be needed for the MCP client to pick up
   the new server. Some runtimes (e.g., Gemini CLI) require a restart; others hot-reload.
2. **Attempt a probe** (`list_designs`) — if it succeeds, continue with the Canva operation.
   If it fails (expected on first run before OAuth), notify:
   > "Canva MCP config added. You'll need to authenticate with Canva on first use.
   >  The OAuth flow will be triggered automatically by your MCP client."
3. **Record the setup** in a serena memory (`canva-mcp-provisioned`) so future sessions
   know the config has been written and don't re-prompt.

### Config File Safety

- **Never overwrite** an existing `canva` entry — if it exists with different settings, skip.
- **Never modify** non-`mcpServers` fields in any config file.
- **Backup** is not created (JSON merge is additive and reversible by removing the key).
- **`.agents/` SSOT rule**: the skill writes to `.agents/mcp.json` and `.agents/mcp_config.json`
  only when the user explicitly approves. This is a config-level change, not a skill definition change.

---

## Security Considerations

1. **OAuth tokens are managed by the MCP client** — the skill never handles or stores Canva credentials.
2. **Design data stays between Canva and the MCP server** — the skill only sends/receives files and metadata.
3. **Uploaded assets are stored in the user's Canva account** — the skill does not control retention or sharing.
4. **No Canva API calls outside MCP** — all Canva interactions go through the registered MCP server tools.
5. **Auto-provisioning is user-approved** — the skill never writes MCP config without explicit user consent.

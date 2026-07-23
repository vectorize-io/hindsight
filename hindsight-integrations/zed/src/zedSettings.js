/**
 * Wire Hindsight into Zed's MCP `context_servers` block.
 *
 * Zed has no native HTTP-MCP transport yet, so we connect to Hindsight's HTTP MCP
 * endpoint through the `mcp-remote` stdio bridge (run via `npx`). The server is
 * registered under `context_servers.hindsight` in Zed's `settings.json`.
 *
 * Zed's `settings.json` is JSONC (it allows comments and trailing commas), which
 * a strict JSON parser can't round-trip without dropping the user's comments. So
 * we only edit the file in place when it parses cleanly as strict JSON; otherwise
 * we return the exact snippet for the user to paste, never risking their config.
 */

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { join, dirname } from "node:path";
import { isDeepStrictEqual } from "node:util";

import { isFile } from "./fsutil.js";

export const SERVER_NAME = "hindsight";

/** Zed's user `settings.json` (`~/.config/zed` on macOS and Linux). */
export function defaultSettingsPath() {
  return join(homedir(), ".config", "zed", "settings.json");
}

/** The Hindsight MCP endpoint for a bank (bank is the last path segment). */
export function mcpEndpointUrl(apiUrl, bankId) {
  return `${apiUrl.replace(/\/+$/, "")}/mcp/${bankId}/`;
}

/**
 * Build the `context_servers.hindsight` entry for Zed's settings.
 *
 * Returns the Zed settings JSON object for the server: an `mcp-remote` bridge to
 * the Hindsight MCP endpoint, with a Bearer auth header when a token is set
 * (omitted for an open self-hosted server).
 */
export function buildContextServer(apiUrl, apiToken, bankId) {
  const args = ["-y", "mcp-remote", mcpEndpointUrl(apiUrl, bankId)];
  if (apiToken) {
    args.push("--header", `Authorization: Bearer ${apiToken}`);
  }
  return { source: "custom", command: "npx", args };
}

/** Render the settings snippet the user can paste into `settings.json`. */
export function renderSnippet(server) {
  return JSON.stringify({ context_servers: { [SERVER_NAME]: server } }, null, 2);
}

/** Parse `path` as strict JSON, or return `null` if absent/not strict. */
function loadStrict(path) {
  if (!isFile(path)) return null;
  let data;
  try {
    data = JSON.parse(readFileSync(path, "utf-8"));
  } catch {
    return null;
  }
  return data && typeof data === "object" && !Array.isArray(data) ? data : null;
}

function isPlainObject(value) {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

/**
 * Add/update `context_servers.hindsight` in Zed's settings at `path`.
 *
 * Returns `{ action, path, snippet? }` where `action` is one of `created`,
 * `merged`, `unchanged`, or `manual` (JSONC we won't rewrite — `snippet` holds
 * what to paste).
 */
export function applyToSettings(path, server) {
  if (!isFile(path)) {
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(
      path,
      JSON.stringify({ context_servers: { [SERVER_NAME]: server } }, null, 2) + "\n",
      "utf-8"
    );
    return { action: "created", path };
  }

  const data = loadStrict(path);
  if (data === null) {
    // JSONC (comments/trailing commas) or unreadable — don't risk a rewrite.
    return { action: "manual", path, snippet: renderSnippet(server) };
  }

  let servers = data.context_servers;
  if (!isPlainObject(servers)) servers = {};
  if (isDeepStrictEqual(servers[SERVER_NAME], server)) {
    return { action: "unchanged", path };
  }
  servers[SERVER_NAME] = server;
  data.context_servers = servers;
  writeFileSync(path, JSON.stringify(data, null, 2) + "\n", "utf-8");
  return { action: "merged", path };
}

/** Remove `context_servers.hindsight` from Zed's settings at `path`. */
export function removeFromSettings(path) {
  const data = loadStrict(path);
  if (data === null) {
    if (isFile(path)) return { action: "manual", path };
    return { action: "unchanged", path };
  }

  const servers = data.context_servers;
  if (!isPlainObject(servers) || !(SERVER_NAME in servers)) {
    return { action: "unchanged", path };
  }
  delete servers[SERVER_NAME];
  if (Object.keys(servers).length > 0) {
    data.context_servers = servers;
  } else {
    delete data.context_servers;
  }
  writeFileSync(path, JSON.stringify(data, null, 2) + "\n", "utf-8");
  return { action: "removed", path };
}

/** Whether our context server is present in Zed's settings at `path`. */
export function isInstalled(path) {
  const data = loadStrict(path);
  if (data === null) return false;
  const servers = data.context_servers;
  return isPlainObject(servers) && SERVER_NAME in servers;
}

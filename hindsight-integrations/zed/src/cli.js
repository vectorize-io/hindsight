/**
 * CLI for the Hindsight Zed integration.
 *
 * `hindsight-zed init` wires Zed's MCP `context_servers` to the Hindsight MCP
 * endpoint and writes a recall/retain rule into Zed's global instructions file.
 * After that, Zed's Agent Panel has `recall`/`retain`/`reflect` tools and is told
 * (via the rule) to use them automatically. There is no background process.
 */

import { writeFileSync, mkdirSync, existsSync, statSync } from "node:fs";
import { dirname, join, delimiter } from "node:path";
import { parseArgs } from "node:util";

import { VERSION } from "./version.js";
import { isFile } from "./fsutil.js";
import { USER_CONFIG_FILE, loadConfig } from "./config.js";
import {
  RULE_TEXT,
  clearRule,
  defaultRulesPath,
  writeRule,
  isInstalled as ruleInstalled,
} from "./rulesFile.js";
import {
  applyToSettings,
  buildContextServer,
  defaultSettingsPath,
  removeFromSettings,
  renderSnippet,
  isInstalled as serverInstalled,
} from "./zedSettings.js";

/** Apply the MCP server entry and the recall/retain rule (the testable core). */
export function buildInstall(config, settingsPath, rulesPath) {
  const server = buildContextServer(
    config.hindsightApiUrl,
    config.hindsightApiToken,
    config.bankId
  );
  const settings = applyToSettings(settingsPath, server);
  writeRule(rulesPath);
  return { settings, rulesPath };
}

function configPath(values) {
  return values["config-path"] || USER_CONFIG_FILE;
}

/** Config from file/env, overridden by any explicitly-passed CLI flags. */
function resolveConfig(values) {
  const cfg = loadConfig({ configFile: configPath(values) });
  if (values["api-url"]) cfg.hindsightApiUrl = values["api-url"];
  if (values["api-token"]) cfg.hindsightApiToken = values["api-token"];
  if (values["bank-id"]) cfg.bankId = values["bank-id"];
  return cfg;
}

/** Persist the resolved connection settings so re-runs remember them. */
function scaffoldConfig(cfg, path) {
  if (isFile(path)) return;
  const data = { hindsightApiUrl: cfg.hindsightApiUrl, bankId: cfg.bankId };
  if (cfg.hindsightApiToken) data.hindsightApiToken = cfg.hindsightApiToken;
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(data, null, 2) + "\n", "utf-8");
}

/** Whether `cmd` resolves on PATH (mirrors Python's `shutil.which`). */
function commandExists(cmd) {
  const dirs = (process.env.PATH || "").split(delimiter).filter(Boolean);
  const exts = process.platform === "win32" ? [".exe", ".cmd", ".bat", ""] : [""];
  for (const dir of dirs) {
    for (const ext of exts) {
      const candidate = join(dir, cmd + ext);
      try {
        if (existsSync(candidate) && statSync(candidate).isFile()) return true;
      } catch {
        // ignore unreadable PATH entries
      }
    }
  }
  return false;
}

function cmdInit(values) {
  const cfg = resolveConfig(values);
  const settingsPath = values["settings-path"] || defaultSettingsPath();
  const rulesPath = values["rules-path"] || defaultRulesPath();
  const server = buildContextServer(cfg.hindsightApiUrl, cfg.hindsightApiToken, cfg.bankId);

  if (values["print-only"]) {
    console.log("Add this to your Zed settings.json:\n");
    console.log(renderSnippet(server));
    console.log("\nAnd add this rule to ~/.config/zed/AGENTS.md:\n");
    console.log(RULE_TEXT);
    return 0;
  }

  console.log("Setting up Hindsight for Zed ...");
  scaffoldConfig(cfg, configPath(values));
  const outcome = buildInstall(cfg, settingsPath, rulesPath);

  if (outcome.settings.action === "manual") {
    console.log(`  Your ${outcome.settings.path} has comments, so I won't rewrite it.`);
    console.log("  Add this `context_servers` entry yourself:\n");
    console.log(renderSnippet(server));
  } else {
    const verb = { created: "Created", merged: "Updated", unchanged: "Already configured in" }[
      outcome.settings.action
    ];
    console.log(
      `  ${verb} ${outcome.settings.path} (MCP server: hindsight → bank '${cfg.bankId}')`
    );
  }
  console.log(`  Wrote recall/retain rule to ${outcome.rulesPath}`);

  if (!commandExists("npx")) {
    console.log("\n  warning: `npx` (Node.js) was not found on PATH. Zed runs the MCP");
    console.log("  bridge via `npx mcp-remote`, so install Node.js for the server to start.");
  }

  console.log("\nDone. Restart Zed, open the Agent Panel, and the `hindsight` MCP server");
  console.log("should show a green dot. Memory recall/retain then happen automatically.");
  return 0;
}

function cmdStatus(values) {
  const settingsPath = values["settings-path"] || defaultSettingsPath();
  const rulesPath = values["rules-path"] || defaultRulesPath();
  console.log(
    `MCP server in ${settingsPath}: ${serverInstalled(settingsPath) ? "installed" : "not installed"}`
  );
  console.log(
    `Recall/retain rule in ${rulesPath}: ${ruleInstalled(rulesPath) ? "installed" : "not installed"}`
  );
  return 0;
}

function cmdUninstall(values) {
  const settingsPath = values["settings-path"] || defaultSettingsPath();
  const rulesPath = values["rules-path"] || defaultRulesPath();
  const result = removeFromSettings(settingsPath);
  if (result.action === "manual") {
    console.log(
      `  ${settingsPath} has comments — remove the \`hindsight\` context_servers entry yourself.`
    );
  } else if (result.action === "removed") {
    console.log(`  Removed the hindsight MCP server from ${settingsPath}`);
  } else {
    console.log(`  No hindsight MCP server found in ${settingsPath}`);
  }
  clearRule(rulesPath);
  console.log(`  Removed the recall/retain rule from ${rulesPath}`);
  return 0;
}

function printHelp() {
  console.log(`hindsight-zed — Hindsight memory for Zed (via MCP)

Usage: hindsight-zed <command> [options]

Commands:
  init         Configure Zed's MCP server + recall/retain rule
  status       Show whether the MCP server + rule are configured
  uninstall    Remove the MCP server + rule

init options:
  --api-url <url>      Hindsight API URL (default: cloud)
  --api-token <token>  Hindsight API token (for Cloud)
  --bank-id <id>       Memory bank for the MCP server (default: zed)
  --print-only         Print the config to add manually; write nothing

  --version            Print version`);
}

const OPTIONS = {
  version: { type: "boolean" },
  help: { type: "boolean" },
  "api-url": { type: "string" },
  "api-token": { type: "string" },
  "bank-id": { type: "string" },
  "print-only": { type: "boolean" },
  // Hidden overrides used by tests and advanced setups.
  "settings-path": { type: "string" },
  "rules-path": { type: "string" },
  "config-path": { type: "string" },
};

export function main(argv = process.argv.slice(2)) {
  let parsed;
  try {
    parsed = parseArgs({ args: argv, allowPositionals: true, options: OPTIONS });
  } catch (err) {
    process.stderr.write(`${err.message}\n`);
    return 2;
  }

  const { values, positionals } = parsed;
  if (values.version) {
    console.log(`hindsight-zed ${VERSION}`);
    return 0;
  }

  const command = positionals[0];
  if (!command || values.help) {
    printHelp();
    return command ? 0 : 1;
  }

  switch (command) {
    case "init":
      return cmdInit(values);
    case "status":
      return cmdStatus(values);
    case "uninstall":
      return cmdUninstall(values);
    default:
      process.stderr.write(`Unknown command: ${command}\n`);
      printHelp();
      return 2;
  }
}

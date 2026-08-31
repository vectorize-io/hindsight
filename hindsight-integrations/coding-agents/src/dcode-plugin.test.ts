import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { HOOK_HARNESSES } from "./harness/hook-lifecycle";

const packageRoot = join(dirname(fileURLToPath(import.meta.url)), "..");

describe("native Dcode plugin", () => {
  it("ships a root Agent Plugin manifest with the shared skill and harness-scoped MCP", () => {
    const manifest = JSON.parse(readFileSync(join(packageRoot, "plugin.json"), "utf8"));
    const packageManifest = JSON.parse(readFileSync(join(packageRoot, "package.json"), "utf8"));
    expect(manifest.version).toBe(packageManifest.version);
    expect(manifest.skills).toBe("./skill");
    expect(manifest.hooks).toBe("./hooks/hooks.json");
    expect(manifest.mcpServers.hindsight).toMatchObject({
      command: "node",
      args: ["${PLUGIN_ROOT}/dist/mcp-server.js"],
      env: { HINDSIGHT_MCP_HARNESS: "dcode" },
    });
  });

  it("registers exactly the lifecycle hooks Dcode owns", () => {
    const hooks = JSON.parse(readFileSync(join(packageRoot, "hooks/hooks.json"), "utf8"));
    expect(Object.keys(hooks.hooks).sort()).toEqual(["SessionStart", "Stop", "UserPromptSubmit"]);
    expect(hooks.hooks.SessionStart[0].hooks[0].command).toBe(
      'node "${PLUGIN_ROOT}/dist/dcode-sessionstart-hook.js"'
    );
    expect(hooks.hooks.UserPromptSubmit[0].hooks[0].command).toBe(
      'node "${PLUGIN_ROOT}/dist/dcode-hook.js"'
    );
    expect(hooks.hooks.Stop[0].hooks[0].command).toBe(
      'node "${PLUGIN_ROOT}/dist/dcode-stop-hook.js"'
    );
  });

  /**
   * Every other hook harness installs its lifecycle FROM `HOOK_HARNESSES[h].install`
   * (installer.ts's mergeHarnessHooks), which is what keeps the events a host fires identical to
   * the events we register. Dcode's plugin manager owns its own registration, so `hooks/hooks.json`
   * is the file that actually runs and the install spec is never read at install time — the two
   * would drift silently. Assert them equal so the declaration stays the single source of truth.
   */
  it("keeps hooks.json byte-identical in meaning to the shared install declaration", () => {
    const hooks = JSON.parse(readFileSync(join(packageRoot, "hooks/hooks.json"), "utf8"));
    const declared = Object.values(HOOK_HARNESSES.dcode.install);
    expect(Object.keys(hooks.hooks).sort()).toEqual(declared.map((h) => h.event).sort());
    for (const hook of declared) {
      const entry = hooks.hooks[hook.event][0].hooks[0];
      expect(entry.command).toBe(`node "\${PLUGIN_ROOT}/dist/${hook.entry}"`);
      expect(entry.timeout).toBe(hook.timeout);
      expect(entry.type).toBe("command");
    }
  });

  it("registers a Stop timeout the retain hook actually budgets against", () => {
    // retain-hook.ts sizes its rate-limit retry window from hostTimeoutSec; a hooks.json that
    // said less would have the host kill the write mid-flight.
    expect(HOOK_HARNESSES.dcode.install.stop.timeout).toBe(
      HOOK_HARNESSES.dcode.retain.hostTimeoutSec
    );
  });
});

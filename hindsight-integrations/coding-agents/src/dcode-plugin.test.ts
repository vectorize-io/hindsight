import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

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
});

import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    // Kilo CLI (an opencode fork) loads this as its persistent plugin — see src/kilo.ts.
    kilo: "src/kilo.ts",
    deepen: "src/deepen.ts",
    installer: "src/installer.ts",
    status: "src/status.ts",
    "claude-hook": "src/claude-hook.ts",
    "claude-stop-hook": "src/claude-stop-hook.ts",
    "claude-sessionstart-hook": "src/claude-sessionstart-hook.ts",
    "cursor-hook": "src/cursor-hook.ts",
    "cursor-sessionstart-hook": "src/cursor-sessionstart-hook.ts",
    "cursor-stop-hook": "src/cursor-stop-hook.ts",
    "copilot-hook": "src/copilot-hook.ts",
    "copilot-sessionstart-hook": "src/copilot-sessionstart-hook.ts",
    "copilot-stop-hook": "src/copilot-stop-hook.ts",
    // Cline CLI loads this module through `cline plugin install`; file hooks cannot mutate
    // model requests, so the native beforeModel hook is the reliable injection path.
    cline: "src/cline.ts",
    // DeepSeek Harness loads this as a native Cordis plugin — see src/dsh.ts.
    dsh: "src/dsh.ts",
    // Prime Agent loads this module as an extension (default export) by absolute path from its
    // settings.json `extensions` array, so it must be self-contained like the hook bins.
    "prime-agent": "src/prime-agent.ts",
    "grok-hook": "src/grok-hook.ts",
    "grok-sessionstart-hook": "src/grok-sessionstart-hook.ts",
    "grok-stop-hook": "src/grok-stop-hook.ts",
    "codex-hook": "src/codex-hook.ts",
    "codex-sessionstart-hook": "src/codex-sessionstart-hook.ts",
    "codex-stop-hook": "src/codex-stop-hook.ts",
    "antigravity-hook": "src/antigravity-hook.ts",
    "antigravity-stop-hook": "src/antigravity-stop-hook.ts",
    "antigravity-statusline": "src/antigravity-statusline.ts",
    "devin-hook": "src/devin-hook.ts",
    "devin-sessionstart-hook": "src/devin-sessionstart-hook.ts",
    "devin-stop-hook": "src/devin-stop-hook.ts",
    // Spawned DETACHED to start the local daemon — a cold start outlives every hook timeout.
    "daemon-start": "src/daemon-start.ts",
    "mcp-server": "src/mcp-server.ts",
    "hindsight-seed": "src/hindsight-seed.ts",
  },
  format: ["esm"],
  target: "node18",
  clean: true,
  dts: { entry: "src/index.ts" },
  shims: false,
  // Each bin entry (claude-hook.js, cursor-hook.js, codex-hook.js, deepen.js, status.js, mcp-server.js,
  // hindsight-seed.js) must be a single self-contained file: hosts wire a hook by absolute path to
  // ONE dist file and never load the rest of the package, so shared code can't live in a separate
  // chunk-*.js.
  splitting: false,
  // mcp-server.js additionally needs its npm deps (the MCP SDK + zod) inlined, since the wrapper
  // only copies the single bundle file, not node_modules. The regexes catch subpath imports too
  // (e.g. "@modelcontextprotocol/sdk/server/mcp.js", "zod/v4/...").
  // hindsight-all (the local-daemon lifecycle manager) is inlined for the same reason: hooks are
  // wired by absolute path to ONE dist file and never load the package's node_modules. It has zero
  // dependencies of its own, so inlining costs almost nothing.
  noExternal: [/^@modelcontextprotocol\/sdk/, /^zod/, /^@vectorize-io\/hindsight-all/],
});

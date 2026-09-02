import esbuild from "esbuild";
import process from "node:process";
// Node's own builtins list — replaces the external "builtin-modules" package.
import { builtinModules as builtins } from "node:module";

const production = process.argv[2] === "production";

// The Obsidian plugin bundle (loaded by the Obsidian/Electron host at runtime).
const pluginContext = await esbuild.context({
  entryPoints: ["src/main.ts"],
  bundle: true,
  // Obsidian and Electron are provided by the host at runtime; never bundle them.
  // Externalize node builtins in both bare ("crypto") and prefixed ("node:crypto") forms.
  external: ["obsidian", "electron", "node:*", ...builtins],
  format: "cjs",
  target: "es2022",
  logLevel: "info",
  sourcemap: production ? false : "inline",
  treeShaking: true,
  outfile: "main.js",
  minify: production,
});

// The headless CLI bin (`hindsight-obsidian-sync`). Runs in plain Node — keep
// node builtins and chokidar external so npm resolves them at install time.
const cliContext = await esbuild.context({
  entryPoints: ["src/node/cli-bin.ts"],
  bundle: true,
  platform: "node",
  external: ["obsidian", "electron", "chokidar", "node:*", ...builtins],
  format: "cjs",
  target: "node20",
  logLevel: "info",
  sourcemap: production ? false : "inline",
  treeShaking: true,
  outfile: "dist/cli.js",
  banner: { js: "#!/usr/bin/env node" },
  minify: production,
});

if (production) {
  await Promise.all([pluginContext.rebuild(), cliContext.rebuild()]);
  await Promise.all([pluginContext.dispose(), cliContext.dispose()]);
} else {
  await Promise.all([pluginContext.watch(), cliContext.watch()]);
}

import esbuild from "esbuild";
import process from "node:process";
// Node's own builtins list — replaces the external "builtin-modules" package.
import { builtinModules as builtins } from "node:module";

const production = process.argv[2] === "production";

const context = await esbuild.context({
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

if (production) {
  await context.rebuild();
  await context.dispose();
} else {
  await context.watch();
}

import esbuild from "esbuild";

const watch = process.argv.includes("--watch");

const sharedConfig = {
  bundle: true,
  platform: "node",
  target: "node20",
  format: "esm",
  external: ["@paperclipai/plugin-sdk"],
};

const builds = [
  { entryPoints: ["src/manifest.ts"], outfile: "dist/manifest.js" },
  { entryPoints: ["src/worker.ts"], outfile: "dist/worker.js" },
];

if (watch) {
  const contexts = await Promise.all(builds.map((b) => esbuild.context({ ...sharedConfig, ...b })));
  await Promise.all(contexts.map((ctx) => ctx.watch()));
  console.log("Watching for changes…");
} else {
  await Promise.all(builds.map((b) => esbuild.build({ ...sharedConfig, ...b })));
  console.log("Build complete.");
}

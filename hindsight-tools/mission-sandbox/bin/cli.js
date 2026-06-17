#!/usr/bin/env node
// Thin launcher: delegate to the compiled CLI. Run `npm run build:lib` (or `npm run build`)
// to produce dist/. For development without a build, use `npm run cli -- <args>` (tsx).
import("../dist/cli/index.js").catch((err) => {
  if (err && err.code === "ERR_MODULE_NOT_FOUND") {
    console.error(
      "mission-sandbox: build output missing. Run `npm run build` in the package first."
    );
  } else {
    console.error(err);
  }
  process.exit(1);
});

/**
 * Executable entrypoint for the `hindsight-obsidian-sync` bin. Kept separate from
 * `cli.ts` (which is side-effect-free and unit-tested) so importing the CLI's
 * functions never runs the process or calls `process.exit`.
 */

import { runCli } from "./cli";

runCli(process.argv.slice(2)).then(
  (code) => process.exit(code),
  (err) => {
    console.error(err instanceof Error ? err.message : String(err));
    process.exit(1);
  }
);

import { runSetup } from "./setup.js";
import type { CliArgs } from "./types.js";

function usage(): void {
  process.stdout.write(`
hindsight-nemoclaw — Setup CLI for Hindsight memory on NemoClaw sandboxes

Usage:
  hindsight-nemoclaw setup [options]

Required options:
  --sandbox <name>       NemoClaw sandbox name (e.g. my-assistant)
  --api-url <url>        Hindsight Cloud API URL (https://api.hindsight.vectorize.io)
  --api-token <token>    Hindsight API key from https://ui.hindsight.vectorize.io
  --bank-prefix <prefix> Bank ID prefix (memories go to <prefix>-openclaw)

Optional options:
  --skip-policy          Skip the openshell policy update
  --skip-plugin-install  Skip openclaw plugins install
  --dry-run              Print what would be changed without executing
  --help                 Show this help

Example:
  hindsight-nemoclaw setup \\
    --sandbox my-assistant \\
    --api-url https://api.hindsight.vectorize.io \\
    --api-token hsk_abc123 \\
    --bank-prefix my-sandbox
`);
}

function parseArgs(argv: string[]): CliArgs | null {
  const args = argv.slice(2);

  if (args.length === 0 || args.includes("--help") || args.includes("-h")) {
    usage();
    return null;
  }

  if (args[0] !== "setup") {
    process.stderr.write(`Unknown command: ${args[0]}\nRun with --help for usage.\n`);
    process.exit(1);
  }

  const get = (flag: string): string | undefined => {
    const idx = args.indexOf(flag);
    if (idx === -1 || idx + 1 >= args.length) return undefined;
    return args[idx + 1];
  };

  const sandbox = get("--sandbox");
  const apiUrl = get("--api-url");
  const apiToken = get("--api-token");
  const bankPrefix = get("--bank-prefix");

  const missing: string[] = [];
  if (!sandbox) missing.push("--sandbox");
  if (!apiUrl) missing.push("--api-url");
  if (!apiToken) missing.push("--api-token");
  if (!bankPrefix) missing.push("--bank-prefix");

  if (missing.length > 0) {
    process.stderr.write(
      `Missing required options: ${missing.join(", ")}\nRun with --help for usage.\n`
    );
    process.exit(1);
  }

  return {
    sandbox: sandbox!,
    apiUrl: apiUrl!,
    apiToken: apiToken!,
    bankPrefix: bankPrefix!,
    skipPolicy: args.includes("--skip-policy"),
    skipPluginInstall: args.includes("--skip-plugin-install"),
    dryRun: args.includes("--dry-run"),
  };
}

const args = parseArgs(process.argv);
if (args) {
  runSetup(args).catch((err) => {
    process.stderr.write(`\nError: ${err instanceof Error ? err.message : String(err)}\n`);
    process.exit(1);
  });
}

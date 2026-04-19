import { execFile } from "child_process";
import { promisify } from "util";
import yaml from "js-yaml";
import type { SandboxPolicy } from "./types.js";

const execFileAsync = promisify(execFile);

/** Strip ANSI escape codes from a string */
export function stripAnsi(str: string): string {
  return str.replace(/\x1B\[[0-9;]*m/g, "");
}

/**
 * Extract and dedent the policy section from `openshell sandbox get` output.
 * The output looks like:
 *
 *   Sandbox:
 *     Id: ...
 *     Name: ...
 *
 *   Policy:
 *     version: 1
 *     filesystem_policy:
 *       ...
 *
 * We need to extract everything after `Policy:` and dedent by 2 spaces.
 */
export function extractPolicyYaml(raw: string): string {
  const stripped = stripAnsi(raw);
  const lines = stripped.split("\n");

  const policyHeaderIdx = lines.findIndex((l) => l.trimEnd() === "Policy:");
  if (policyHeaderIdx === -1) {
    throw new Error('Could not find "Policy:" section in `openshell sandbox get` output');
  }

  const policyLines = lines.slice(policyHeaderIdx + 1);

  // Dedent by 2 spaces (the policy block is indented under `Policy:`)
  const dedented = policyLines.map((l) => {
    if (l.startsWith("  ")) return l.slice(2);
    return l;
  });

  // Drop trailing empty lines
  while (dedented.length > 0 && dedented[dedented.length - 1].trim() === "") {
    dedented.pop();
  }

  return dedented.join("\n");
}

/**
 * Parse `openshell sandbox get <name>` output into a SandboxPolicy object.
 * Throws a descriptive error if parsing fails.
 */
export function parseSandboxPolicy(rawOutput: string): SandboxPolicy {
  const policyYaml = extractPolicyYaml(rawOutput);

  try {
    const parsed = yaml.load(policyYaml);
    if (typeof parsed !== "object" || parsed === null) {
      throw new Error("Parsed policy is not an object");
    }
    return parsed as SandboxPolicy;
  } catch (err) {
    throw new Error(
      `Failed to parse sandbox policy YAML.\n` +
        `This may mean the openshell output format has changed.\n` +
        `Apply the Hindsight policy manually using the instructions in NEMOCLAW.md.\n` +
        `Parse error: ${err}`
    );
  }
}

/** Run `openshell sandbox get <sandbox>` and return parsed policy */
export async function readSandboxPolicy(sandboxName: string): Promise<SandboxPolicy> {
  let stdout: string;
  try {
    const result = await execFileAsync("openshell", ["sandbox", "get", sandboxName]);
    stdout = result.stdout;
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(`Failed to run \`openshell sandbox get ${sandboxName}\`: ${msg}`);
  }

  return parseSandboxPolicy(stdout);
}

import { homedir } from "node:os";
import { join } from "node:path";
import type { HarnessDockerSetup } from "./harness";

export const codexDockerSetup: HarnessDockerSetup = {
  name: "codex",
  hindsightHarness: "codex",
  credentialPath: () => process.env.CODEX_E2E_AUTH_PATH || join(homedir(), ".codex", "auth.json"),
  credentialTarget: "/root/.codex/auth.json",
  installCommand: "hindsight-coding-agents install codex",
  command: (prompt) => [
    "codex",
    "exec",
    "--sandbox",
    "workspace-write",
    "--dangerously-bypass-hook-trust",
    "--cd",
    "/workspace",
    "--output-last-message",
    "/results/last-message.txt",
    prompt,
  ],
};

/**
 * Companion-skill self-update: `npm update -g` refreshes the PACKAGE's skill/, but the copies the
 * installer placed in each host's skills directory would go stale. Every session start compares
 * the installed copy against the packaged one and re-copies on drift — so upgrading the plugin
 * upgrades the skill, no re-install needed.
 *
 * Presence-gated: a host where the skill was never installed (or was uninstalled) is left alone —
 * installation remains the installer's decision; this only keeps existing copies current.
 */
import { cpSync, existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Where each host keeps the installed skill, so a session start can refresh it in place.
 *
 * MUST list every harness whose installer calls `installSkill` — the structural guard in
 * skill-sync.test.ts asserts that, with an explicit exemption list for the hosts that predate it
 * (see there). A harness missing here installs the skill once and then keeps that copy forever.
 */
export const SKILL_DIRS: Record<string, string[]> = {
  "claude-code": [".claude", "skills"],
  codex: [".agents", "skills"], // agentskills-standard shared dir
  "antigravity-cli": [".gemini", "config", "skills"],
  "cursor-cli": [".cursor", "skills"],
  "qwen-code": [".qwen", "skills"], // Qwen's user-level skills root (Storage.getUserSkillsDirs)
};

/** The packaged skill dir (pkgRoot/skill, resolved relative to the built dist). */
function packagedSkillDir(): string {
  return join(dirname(fileURLToPath(import.meta.url)), "..", "skill");
}

export function syncCompanionSkill(
  harness: string,
  opts: { home?: string; srcDir?: string } = {}
): void {
  try {
    const parts = SKILL_DIRS[harness];
    if (!parts) return; // host without a skills mechanism (opencode)
    const src = opts.srcDir ?? packagedSkillDir();
    const srcMd = join(src, "SKILL.md");
    if (!existsSync(srcMd)) return;
    const dst = join(opts.home ?? homedir(), ...parts, "hindsight-coding-agent");
    const dstMd = join(dst, "SKILL.md");
    if (!existsSync(dstMd)) return; // never installed here — not ours to decide
    if (readFileSync(srcMd, "utf8") !== readFileSync(dstMd, "utf8")) {
      cpSync(src, dst, { recursive: true });
    }
  } catch {
    /* best-effort — skill freshness must never break a session */
  }
}

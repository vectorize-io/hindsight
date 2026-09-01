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
import { SKILL_DIRS } from "./skill-dirs";

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

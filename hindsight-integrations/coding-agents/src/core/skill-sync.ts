/**
 * Companion-skill self-update: `npm update -g` refreshes the PACKAGE's skill/, but the copies the
 * installer placed in each host's skills directory would go stale. Every session start compares
 * the installed copy against the packaged one and re-copies on drift — so upgrading the plugin
 * upgrades the skill, no re-install needed.
 *
 * Presence-gated BY DEFAULT: a host where the skill was never installed (or was uninstalled) is
 * left alone, because for a hook harness the hook registration and the skill are installed by the
 * same command — our installer — so an absent skill is a decision.
 *
 * `install: true` flips that, and is what the persistent-plugin hosts pass (core/runtime.ts): those
 * can be wired by the HOST's own plugin manager — `dsh plugin add @vectorize-io/…`,
 * `cline plugin install`, a hand-written config entry — a route our installer never sees, so the
 * plugin loads with its tools registered and no skill anywhere (#4406). There the plugin BEING
 * LOADED is the installation signal: uninstalling removes the entry that loads us, so a running
 * plugin can only mean the host still wants us.
 */
import { cpSync, existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { SKILL_DIRS } from "./skill-dirs";

/** The packaged skill dir (pkgRoot/skill): one level up from the flat `dist/` bundle, two from this
 *  file in the source tree — so the source tree resolves it too, and a test asserts on the real
 *  packaged SKILL.md rather than on a fixture that cannot notice a missing skill build. */
function packagedSkillDir(): string {
  const here = dirname(fileURLToPath(import.meta.url));
  const dist = join(here, "..", "skill");
  return existsSync(join(dist, "SKILL.md")) ? dist : join(here, "..", "..", "skill");
}

export function syncCompanionSkill(
  harness: string,
  opts: { home?: string; srcDir?: string; install?: boolean } = {}
): void {
  try {
    const parts = SKILL_DIRS[harness];
    if (!parts) return; // host without a skills DIRECTORY (opencode, opencode2 — see skill-dirs.ts)
    const src = opts.srcDir ?? packagedSkillDir();
    const srcMd = join(src, "SKILL.md");
    if (!existsSync(srcMd)) return;
    const dst = join(opts.home ?? homedir(), ...parts, "hindsight-coding-agent");
    const dstMd = join(dst, "SKILL.md");
    const installed = existsSync(dstMd);
    if (!installed && !opts.install) return; // never installed here — not ours to decide
    if (!installed || readFileSync(srcMd, "utf8") !== readFileSync(dstMd, "utf8")) {
      cpSync(src, dst, { recursive: true }); // recursive cpSync creates the skills root too
    }
  } catch {
    /* best-effort — skill freshness must never break a session */
  }
}

/**
 * The packaged skill in the shape a host that registers skills IN MEMORY wants (opencode2's
 * `ctx.skill.transform`): the frontmatter `description` the host matches on, the body, and where
 * it came from on disk. opencode v2 (2.0.16+) requires `path`, the absolute SKILL.md file, and
 * rejects a draft without it (#4732); `location`, the directory, was the field sent before that
 * and stays for earlier v2 hosts, so both go out. Read at setup, so `npm update -g` upgrades this
 * copy too.
 */
export function readPackagedSkill(
  srcDir = packagedSkillDir()
): { location: string; path: string; description: string; content: string } | undefined {
  try {
    const path = join(srcDir, "SKILL.md");
    const raw = readFileSync(path, "utf8");
    const frontmatter = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(raw);
    return {
      location: srcDir,
      path,
      description: frontmatter ? (/^description:\s*(.+)$/m.exec(frontmatter[1])?.[1] ?? "") : "",
      content: frontmatter ? raw.slice(frontmatter[0].length) : raw,
    };
  } catch {
    return undefined; // no packaged skill (a dev tree before `npm run skill:build`)
  }
}

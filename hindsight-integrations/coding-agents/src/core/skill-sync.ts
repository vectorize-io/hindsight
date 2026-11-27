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
import { cpSync, existsSync, readdirSync, readFileSync, renameSync, rmSync } from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { SKILL_NAMES, resolveSkillDirs } from "./skill-dirs";

/** The packaged skills root (pkgRoot/skill, one directory per SKILL_NAMES entry): one level up
 *  from the flat `dist/` bundle, two from this file in the source tree — so the source tree
 *  resolves it too, and a test asserts on the real packaged SKILL.md rather than on a fixture that
 *  cannot notice a missing skill build. */
function packagedSkillDir(): string {
  const here = dirname(fileURLToPath(import.meta.url));
  const dist = join(here, "..", "skill");
  return existsSync(join(dist, SKILL_NAMES[0], "SKILL.md"))
    ? dist
    : join(here, "..", "..", "skill");
}

/** Clear what a replace killed midway (a hook timeout) leaves next to `dst`: `<name>.tmp-<pid>`
 *  copies, which a host loading every `<dir>/SKILL.md` would load as a duplicate skill, and
 *  `<name>.old-<pid>` asides — put back when `dst` itself is missing, since that is the installed
 *  skill caught between the two renames. */
function clearLeftovers(dst: string): void {
  for (const { path, kind } of leftoversOf(dst)) {
    if (kind === "old" && !existsSync(dst)) renameSync(path, dst);
    else rmSync(path, { recursive: true, force: true });
  }
}

/** The `<name>.tmp-<pid>` / `<name>.old-<pid>` siblings replaceSkillDir creates next to `dst` —
 *  only those exact names, so a user's own `hindsight-coding-agent.old-backup` is never touched. */
export function leftoversOf(dst: string): { path: string; kind: "tmp" | "old" }[] {
  const dir = dirname(dst);
  let entries: string[];
  try {
    entries = readdirSync(dir);
  } catch {
    return []; // no skills root yet
  }
  const escaped = basename(dst).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const ours = new RegExp(`^${escaped}\\.(tmp|old)-\\d+$`);
  return entries.flatMap((entry) => {
    const m = ours.exec(entry);
    return m ? [{ path: join(dir, entry), kind: m[1] as "tmp" | "old" }] : [];
  });
}

/** Replace `dst` with a copy of `src`, whole, so a file the package dropped goes too. The copy goes
 *  to a temp sibling first and swaps in by two renames, the old copy set aside rather than deleted:
 *  a copy that fails (disk full, killed by a hook timeout) must not leave the host without the
 *  everyday skill, which the next sync would read as "uninstalled" and never restore. A kill
 *  between the renames leaves the aside, which clearLeftovers puts back. */
export function replaceSkillDir(src: string, dst: string): void {
  clearLeftovers(dst);
  const tmp = `${dst}.tmp-${process.pid}`;
  const old = `${dst}.old-${process.pid}`;
  try {
    cpSync(src, tmp, { recursive: true }); // recursive cpSync creates the skills root too
    if (existsSync(dst)) renameSync(dst, old);
    renameSync(tmp, dst);
  } catch (e) {
    rmSync(tmp, { recursive: true, force: true });
    if (!existsSync(dst) && existsSync(old)) renameSync(old, dst);
    throw e;
  }
  rmSync(old, { recursive: true, force: true });
}

export function syncCompanionSkill(
  harness: string,
  opts: { home?: string; srcDir?: string; install?: boolean } = {}
): void {
  try {
    const parts = resolveSkillDirs(harness, opts.home ?? homedir());
    if (!parts) return; // host without a skills DIRECTORY (opencode, opencode2 — see skill-dirs.ts)
    const src = opts.srcDir ?? packagedSkillDir();
    const base = join(opts.home ?? homedir(), ...parts);
    // Before the presence check: an everyday skill caught mid-swap is still installed.
    for (const name of SKILL_NAMES) clearLeftovers(join(base, name));
    // The everyday skill decides for both: an install from before the setup skill existed gets it
    // added on the next session, and an uninstalled host stays uninstalled.
    if (!existsSync(join(base, SKILL_NAMES[0], "SKILL.md")) && !opts.install) return;
    for (const name of SKILL_NAMES) {
      const srcMd = join(src, name, "SKILL.md");
      if (!existsSync(srcMd)) continue;
      const dstMd = join(base, name, "SKILL.md");
      if (!existsSync(dstMd) || readFileSync(srcMd, "utf8") !== readFileSync(dstMd, "utf8")) {
        replaceSkillDir(join(src, name), join(base, name));
      }
    }
  } catch {
    /* best-effort — skill freshness must never break a session */
  }
}

/**
 * The packaged skills in the shape a host that registers skills IN MEMORY wants (opencode2's
 * `ctx.skill.transform`): the frontmatter `description` the host matches on, the body, and where
 * it came from on disk. opencode v2 (2.0.16+) requires `path`, the absolute SKILL.md file, and
 * rejects a draft without it (#4732); `location`, the directory, was the field sent before that
 * and stays for earlier v2 hosts, so both go out. Read at setup, so `npm update -g` upgrades this
 * copy too.
 */
export interface PackagedSkill {
  name: string;
  location: string;
  path: string;
  description: string;
  content: string;
}

export function readPackagedSkills(srcDir = packagedSkillDir()): PackagedSkill[] {
  const out: PackagedSkill[] = [];
  for (const name of SKILL_NAMES) {
    const location = join(srcDir, name);
    const path = join(location, "SKILL.md");
    let raw: string;
    try {
      raw = readFileSync(path, "utf8");
    } catch {
      continue; // not packaged (a dev tree before `npm run skill:build`)
    }
    const frontmatter = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(raw);
    out.push({
      name,
      location,
      path,
      description: frontmatter ? (/^description:\s*(.+)$/m.exec(frontmatter[1])?.[1] ?? "") : "",
      content: frontmatter ? raw.slice(frontmatter[0].length) : raw,
    });
  }
  return out;
}

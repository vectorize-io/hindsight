/**
 * Where each host keeps the companion skill, as home-relative path parts.
 *
 * ONE map, because two independent code paths write and refresh those directories: the installer
 * copies the packaged skill in and removes it again (src/installer.ts), and every session start
 * re-copies it on drift so `npm update -g` upgrades the skill too (core/skill-sync.ts).
 *
 * They used to hold the paths separately, and the self-update copy listed only four of the ten
 * hosts the installer writes — so Copilot, Grok Build, Cline, dsh, pi and Prime Agent stayed pinned
 * to whichever SKILL.md they happened to be installed with, forever. Same shape as #3524: the
 * sibling nobody wrote a test for is the sibling that gets forgotten, so the list lives once and
 * `installer.test.ts` asserts it over the whole family.
 *
 * A host absent here writes no skills DIRECTORY: opencode and its Kilo fork have no skills
 * mechanism at all, and opencode2 has one but no directory of its own to install into — its plugin
 * registers the packaged skill in memory instead (harness/opencode2.ts, #4352).
 */
import { existsSync } from "node:fs";
import { join } from "node:path";

export const SKILL_DIRS: Record<string, string[]> = {
  "claude-code": [".claude", "skills"],
  // Codex and dsh share the agentskills-standard root; uninstalling either removes the one copy.
  codex: [".agents", "skills"],
  dsh: [".agents", "skills"],
  "antigravity-cli": [".gemini", "config", "skills"],
  "cursor-cli": [".cursor", "skills"],
  "copilot-cli": [".copilot", "skills"],
  "grok-build": [".grok", "skills"],
  "cline-cli": [".cline", "data", "settings", "skills"],
  "qwen-code": [".qwen", "skills"], // Qwen's user-level skills root (Storage.getUserSkillsDirs)
  "factory-droid": [".factory", "skills"], // Droid's user-level skills root
  // ZCode scans TWO user roots by default (its resolveDefaultSkillRoots): `~/.zcode/skills` and the
  // shared agentskills `~/.agents/skills`. Write its OWN, for the same reason the pi family does —
  // skill removal is by fixed directory name, so installing into the shared root would make
  // `uninstall zcode` take Codex's and dsh's copy with it.
  zcode: [".zcode", "skills"],
  // TraeCode's user-level skills root, resolved per edition by traecodeDotDir below — like ZCode's
  // its own, never the shared agentskills root other hosts install into.
  traecode: [".trae-cn", "skills"],
  // The pi family reads the shared ~/.agents/skills too, but writes its OWN root: skill removal is
  // by fixed directory name, so installing to the shared one would make `uninstall pi` take Codex's
  // and dsh's copy with it.
  pi: [".pi", "agent", "skills"],
  "prime-agent": [".prime", "agent", "skills"],
};

/**
 * TRAE ships two editions whose user-level dot-dir differs: the CN build (TraeCode, this harness's
 * target) uses `~/.trae-cn` — verified against a live install — while the international build uses
 * `~/.trae` (docs.trae.ai: rules, commands and skills all live there). No other spelling exists.
 * The app creates the dir itself, so whichever exists wins; with neither present, default to the
 * CN name. The same candidates drive the Electron userData probe for mcp.json in installer.ts.
 */
const TRAE_DOT_DIRS = [".trae-cn", ".trae"];

/** The edition's dot-dir NAME under home (kept relative — SKILL_DIRS parts are home-relative). */
export const traecodeDotDirName = (
  home: string,
  exists: (p: string) => boolean = existsSync
): string => {
  for (const dir of TRAE_DOT_DIRS) {
    if (exists(join(home, dir))) return dir;
  }
  return ".trae-cn";
};

/** Resolve a host's skills root, applying the per-edition probe for hosts whose dot-dir depends on
 * the installed build (traecode). Callers that only need the static map keep using SKILL_DIRS. */
export const resolveSkillDirs = (harness: string, home: string): string[] | undefined => {
  const parts = SKILL_DIRS[harness];
  if (harness === "traecode" && parts) {
    return [traecodeDotDirName(home), ...parts.slice(1)];
  }
  return parts;
};

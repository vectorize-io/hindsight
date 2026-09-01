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
 * A host absent here has no skills mechanism at all (opencode and its Kilo fork).
 */
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
  // The pi family reads the shared ~/.agents/skills too, but writes its OWN root: skill removal is
  // by fixed directory name, so installing to the shared one would make `uninstall pi` take Codex's
  // and dsh's copy with it.
  pi: [".pi", "agent", "skills"],
  "prime-agent": [".prime", "agent", "skills"],
};

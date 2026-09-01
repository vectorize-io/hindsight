import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { SKILL_DIRS, syncCompanionSkill } from "./skill-sync";

describe("syncCompanionSkill", () => {
  const dirs: string[] = [];
  const tmp = (p: string) => {
    const d = mkdtempSync(join(tmpdir(), p));
    dirs.push(d);
    return d;
  };
  afterEach(() => {
    for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true });
  });

  const setup = (installedContent?: string) => {
    const home = tmp("skill-home-");
    const src = tmp("skill-src-");
    writeFileSync(join(src, "SKILL.md"), "NEW CONTENT v2");
    if (installedContent !== undefined) {
      const dst = join(home, ".claude", "skills", "hindsight-coding-agent");
      mkdirSync(dst, { recursive: true });
      writeFileSync(join(dst, "SKILL.md"), installedContent);
    }
    return { home, src };
  };

  it("updates a stale installed copy to the packaged content", () => {
    const { home, src } = setup("OLD CONTENT v1");
    syncCompanionSkill("claude-code", { home, srcDir: src });
    expect(
      readFileSync(join(home, ".claude", "skills", "hindsight-coding-agent", "SKILL.md"), "utf8")
    ).toBe("NEW CONTENT v2");
  });

  it("does NOT install where the skill was never installed (uninstall stays respected)", () => {
    const { home, src } = setup(undefined);
    syncCompanionSkill("claude-code", { home, srcDir: src });
    expect(existsSync(join(home, ".claude", "skills", "hindsight-coding-agent"))).toBe(false);
  });

  it("no-ops on identical content and on hosts without a skills mechanism", () => {
    const { home, src } = setup("NEW CONTENT v2");
    syncCompanionSkill("claude-code", { home, srcDir: src }); // same content — no throw, unchanged
    syncCompanionSkill("opencode", { home, srcDir: src }); // no mechanism — no-op
    expect(
      readFileSync(join(home, ".claude", "skills", "hindsight-coding-agent", "SKILL.md"), "utf8")
    ).toBe("NEW CONTENT v2");
  });
});

/**
 * Family-wide guard (#3524 shape): a harness whose installer copies the skill but that SKILL_DIRS
 * does not map installs it once and then never refreshes it — `npm update -g` upgrades the package
 * while that host keeps the old SKILL.md until someone re-installs. A per-harness test cannot catch
 * this, because the harness that forgets is by definition the one whose test nobody wrote. So
 * enumerate the installer's own call sites instead and assert each is mapped.
 *
 * KNOWN_UNMAPPED are pre-existing gaps found while adding qwen-code; they are real defects, listed
 * so they stay visible and so a NEW harness cannot silently join them.
 */
describe("SKILL_DIRS covers every harness the installer copies the skill into", () => {
  const KNOWN_UNMAPPED = new Set(["copilot-cli", "grok-build", "cline-cli", "dsh", "prime-agent"]);

  it("maps every installSkill target", () => {
    const src = readFileSync(new URL("../installer.ts", import.meta.url), "utf8");
    const targets = [...src.matchAll(/installSkill\(c, "([^"]+)"/g)].map((m) => m[1]);
    expect(targets.length).toBeGreaterThan(5); // the regex still matches the real call sites
    expect(targets.filter((h) => !KNOWN_UNMAPPED.has(h) && !(h in SKILL_DIRS))).toEqual([]);
  });
});

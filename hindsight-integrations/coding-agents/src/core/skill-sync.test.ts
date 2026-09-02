import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { INSTALLERS } from "../installer";
import { SKILL_DIRS } from "./skill-dirs";
import { syncCompanionSkill } from "./skill-sync";

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
 * this, because the harness that forgets is by definition the one whose test nobody wrote.
 *
 * This once grepped `installSkill(c, "<harness>"` out of installer.ts, with an exemption list for
 * five hosts it found unmapped. Both halves are gone. The installer now derives every skills
 * directory from this map, so an unmapped host cannot install at all (skillsBaseFor throws) — and
 * the pi family, which reaches installSkill through a shared factory with a `harness` variable,
 * matched no literal and so was invisible to that regex anyway. installer.test.ts owns the forward
 * direction by behaviour instead: it installs EVERY entry of INSTALLERS, finds where the skill
 * actually landed, and requires the self-update to refresh that copy.
 *
 * What is left here is the reverse direction, which nothing else states: a mapped host must be a
 * host that exists. A stale entry is a path this module would keep writing to for an agent the
 * installer no longer supports.
 */
describe("SKILL_DIRS maps only real harnesses", () => {
  it("names no harness the installer does not offer", () => {
    const installable = new Set(INSTALLERS.map((i) => i.name));
    expect(Object.keys(SKILL_DIRS).filter((h) => !installable.has(h))).toEqual([]);
  });
});

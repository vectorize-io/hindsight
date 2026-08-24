/**
 * The README is the single source of truth for configuration (scripts/build-skill.mjs copies its
 * marked regions into the companion skill; hindsight-docs/scripts/sync-coding-agents-doc.mjs copies
 * the whole thing into the docs site). These tests keep it honest in both directions:
 *
 *   - the generated skill matches what the README currently says, and
 *   - no config field is readable but undocumented — the drift issue #3735 was filed about, where
 *     `bankIdTemplate`, `optInOnly`, `banks.<id>.bank` and nine others were reachable in code while
 *     the docs a user actually reads never named them.
 */
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
// @ts-expect-error — plain .mjs build script, no type declarations
import { regions } from "../scripts/build-skill.mjs";

const pkgRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const readme = () => readFileSync(join(pkgRoot, "README.md"), "utf8");

describe("companion skill", () => {
  it("is up to date with the README", () => {
    // Throws with the regeneration command in its message when the skill is stale.
    execFileSync("node", [join(pkgRoot, "scripts", "build-skill.mjs"), "--check"], {
      cwd: pkgRoot,
      stdio: "pipe",
    });
  });

  it("carries the configuration reference, not a subset of it", () => {
    const skill = readFileSync(join(pkgRoot, "skill", "SKILL.md"), "utf8");
    // The keys that fix the failure mode #3735 reported: a bank shared across unrelated projects.
    for (const key of ["bankIdTemplate", "dynamicBankId", "mapPathToBank", "disabled", "bank"]) {
      expect(skill).toContain(`\`${key}\``);
    }
  });
});

describe("configuration reference", () => {
  /** Field names declared in `interface RawConfig` — the whole surface a config file may set. */
  const rawConfigFields = (): string[] => {
    const src = readFileSync(join(pkgRoot, "src", "core", "config.ts"), "utf8");
    const block = /export interface RawConfig \{([\s\S]*?)\n\}/.exec(src);
    if (!block) throw new Error("could not find interface RawConfig in core/config.ts");
    return [...block[1].matchAll(/^ {2}(\w+)\??:/gm)].map((m) => m[1]);
  };

  it("documents every field of RawConfig", () => {
    const doc = readme();
    const undocumented = rawConfigFields().filter((f) => !doc.includes(`\`${f}\``));
    expect(undocumented).toEqual([]);
  });

  it("documents the env-var rule truthfully — every key really is HINDSIGHT_<FIELD_IN_CAPS>", () => {
    // The README documents the env layer as a RULE rather than a 35-row table, so the rule has to
    // hold for every key: one deviating name would be a setting nobody could derive from the docs.
    const src = readFileSync(join(pkgRoot, "src", "core", "config.ts"), "utf8");
    const block = /const ENV_KEYS = \{([\s\S]*?)\n\} as const/.exec(src);
    if (!block) throw new Error("could not find ENV_KEYS in core/config.ts");
    const pairs = [...block[1].matchAll(/^\s*(\w+): "(HINDSIGHT_\w+)"/gm)];
    expect(pairs.length).toBeGreaterThan(30);
    const expected = (field: string) =>
      "HINDSIGHT_" + field.replace(/([a-z0-9])([A-Z])/g, "$1_$2").toUpperCase();
    expect(pairs.filter(([, field, env]) => expected(field) !== env).map(([, f]) => f)).toEqual([]);
  });

  it("keeps the map-valued settings out of the env layer, as documented", () => {
    // The README tells users these four are file-only because per-key branching cannot survive
    // flattening into one variable. If one ever gained an env var, that sentence would be a lie.
    const src = readFileSync(join(pkgRoot, "src", "core", "config.ts"), "utf8");
    const block = /const ENV_KEYS = \{([\s\S]*?)\n\} as const/.exec(src);
    const fields = [...block![1].matchAll(/^\s*(\w+): "HINDSIGHT_/gm)].map((m) => m[1]);
    for (const fileOnly of ["mapPathToBank", "harnesses", "banks", "retainMetadata"]) {
      expect(fields).not.toContain(fileOnly);
    }
    expect(readme()).toContain("are file-only");
  });

  it("documents the per-bank section's own `bank` rename field", () => {
    expect(readme()).toMatch(/`banks\.<bankId>`|`banks`/);
    expect(readme()).toContain('"bank": "team::shared"');
  });
});

describe("skill region extraction", () => {
  const extract = (src: string): string[] => regions(src) as string[];

  it("normalises a region's headings so its top level becomes the skill's ##", () => {
    const [out] = extract(
      [
        "<!-- skill:begin -->",
        "### Reference",
        "",
        "#### Recipe",
        "",
        "text",
        "<!-- skill:end -->",
      ].join("\n")
    );
    expect(out).toContain("## Reference");
    expect(out).toContain("### Recipe");
  });

  it("keeps a region that already starts at ## unchanged", () => {
    const [out] = extract(
      ["<!-- skill:begin -->", "## Configuration", "", "### Opt-in", "<!-- skill:end -->"].join(
        "\n"
      )
    );
    expect(out).toContain("## Configuration");
    expect(out).toContain("### Opt-in");
  });

  it("synthesises a heading for a region that starts mid-section", () => {
    const [out] = extract(
      ['<!-- skill:begin title="Install / update" -->', "run it", "<!-- skill:end -->"].join("\n")
    );
    expect(out).toBe("## Install / update\n\nrun it");
  });

  it("refuses a headingless region with no title rather than silently merging it upward", () => {
    expect(() =>
      extract(["<!-- skill:begin -->", "orphan text", "<!-- skill:end -->"].join("\n"))
    ).toThrow(/must declare title=/);
  });

  it("refuses an unterminated region rather than swallowing the rest of the README", () => {
    expect(() => extract(["<!-- skill:begin -->", "## A", "text"].join("\n"))).toThrow(
      /unterminated/
    );
  });

  it("refuses a README with no marked regions at all", () => {
    expect(() => extract("## Configuration\n\nno markers here")).toThrow(/no <!-- skill:begin -->/);
  });
});

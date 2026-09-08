/**
 * The docs site draws a row of harness logos in two places — the integrations gallery card and the
 * coding-agent sidebar preview — from ONE list: hindsight-docs/src/lib/coding-agent-harnesses.ts.
 *
 * That list used to be two hand-maintained copies, and both drifted: `dcode`, `opencode2`, `pi` and
 * `prime-agent` were each missing from one of them, so agents this package fully supported were
 * invisible on the site that advertises them. Merging them fixes the drift BETWEEN the two surfaces;
 * this test fixes the drift between the site and reality, by binding the roster to the registry that
 * actually decides what `install <harness>` accepts.
 *
 * Adding a harness therefore means adding its logo in the same change — which is the point.
 */
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { HARNESS_NAMES } from "./harness/registry";

const pkgRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const docsRoot = join(pkgRoot, "..", "..", "hindsight-docs");
const rosterFile = join(docsRoot, "src", "lib", "coding-agent-harnesses.ts");

/** The roster entries, parsed out of the source rather than imported: docs is a separate package. */
function docsRoster(): { id: string; label: string; file: string }[] {
  const src = readFileSync(rosterFile, "utf8");
  const block =
    /export const CODING_AGENT_HARNESSES: CodingAgentHarness\[\] = \[([\s\S]*?)\n\];/.exec(src);
  if (!block) throw new Error(`could not find CODING_AGENT_HARNESSES in ${rosterFile}`);
  return [...block[1].matchAll(/\{id: '([^']+)', label: '([^']*)', file: '([^']+)'\}/g)].map(
    (m) => ({ id: m[1], label: m[2], file: m[3] })
  );
}

describe("docs coding-agent roster", () => {
  it("lists exactly the harnesses this package can install", () => {
    // Sorted: presentation order on the site is editorial and deliberately not the registry's.
    expect(
      docsRoster()
        .map((h) => h.id)
        .sort()
    ).toEqual([...HARNESS_NAMES].sort());
  });

  it("ships the icon every entry points at", () => {
    // A missing file is a broken image on the site's busiest card, and nothing else would catch it:
    // the roster is plain data, so a typo'd file name still type-checks and still builds.
    const missing = docsRoster()
      .map((h) => join(docsRoot, "static", "img", "harness", h.file))
      .filter((path) => !existsSync(path));
    expect(missing).toEqual([]);
  });
});

describe("README agent roster", () => {
  const readme = () => readFileSync(join(pkgRoot, "README.md"), "utf8");

  it("names every harness in the opening paragraph", () => {
    // The third copy of this list, and the one a reader meets first — it had gone stale by two
    // agents (Qwen Code, Factory Droid) while both per-agent sections were present and correct.
    const intro = readme().split("\n").slice(0, 8).join("\n");
    const missing = docsRoster()
      .map((h) => h.label)
      .filter((label) => !intro.includes(`**${label}**`));
    expect(missing).toEqual([]);
  });

  it("gives every harness its own install section", () => {
    const sections = readme();
    const missing = docsRoster()
      .map((h) => h.label)
      .filter((label) => !sections.includes(`/> ${label}\n`));
    expect(missing).toEqual([]);
  });
});

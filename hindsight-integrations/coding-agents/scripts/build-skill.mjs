#!/usr/bin/env node
/**
 * Generate skill/SKILL.md from the README + skill-src/preamble.md.
 *
 * The companion skill and the README documented the same configuration in two different files, and
 * they drifted apart in BOTH directions: the skill never named `bankIdTemplate`, `optInOnly`,
 * `retainTags` or the daemon settings and claimed outright that "no environment variables" exist
 * (there are 30-odd, see core/config.ts ENV_KEYS), while the README's reference table had lost
 * `autoReflect` and `surveyRefreshCommits`, which only the skill had. Issue #3735 was filed against
 * the half of that drift that faces users.
 *
 * So the README is the single source for everything BOTH audiences need — configuration, install,
 * diagnostics — and this script copies the regions it marks into the skill:
 *
 *     <!-- skill:begin -->                          … <!-- skill:end -->
 *     <!-- skill:begin title="Install / update" --> … <!-- skill:end -->
 *
 * The skill-only half — what the agent does with the memory (tools, crediting, corrections) — has
 * no place in an npm README, so it stays in skill-src/preamble.md, which carries the frontmatter
 * and is emitted first.
 *
 * Heading levels are normalised per region: a region's first heading becomes `##` and everything
 * below it shifts by the same delta, so a marked `### Reference` reads as a top-level section of
 * the skill without the README having to know that. A region that starts mid-section (no heading of
 * its own) declares `title=` and gets an `##` synthesised.
 *
 * Run: node scripts/build-skill.mjs [--check]
 * `--check` fails when skill/SKILL.md is out of date instead of writing it (for CI — see
 * src/docs-freshness.test.ts, which is what actually gates it and unit-tests the marker parsing).
 */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pkgRoot = join(here, "..");
const readme = join(pkgRoot, "README.md");
const preamble = join(pkgRoot, "skill-src", "preamble.md");
const skill = join(pkgRoot, "skill", "SKILL.md");

const BEGIN = /^<!--\s*skill:begin(?:\s+title="([^"]*)")?\s*-->$/;
const END = /^<!--\s*skill:end\s*-->$/;

/** The marked regions of the README, in file order, with heading levels normalised. */
export function regions(src) {
  const out = [];
  let current = null;
  for (const line of src.split("\n")) {
    const begin = BEGIN.exec(line);
    if (begin) {
      if (current) throw new Error("nested <!-- skill:begin -->");
      current = { title: begin[1], lines: [] };
      continue;
    }
    if (END.test(line)) {
      if (!current) throw new Error("<!-- skill:end --> without a begin");
      out.push(render(current));
      current = null;
      continue;
    }
    if (current) current.lines.push(line);
  }
  if (current) throw new Error("unterminated <!-- skill:begin -->");
  if (!out.length) throw new Error("no <!-- skill:begin --> regions in the README");
  return out;
}

function render({ title, lines }) {
  const body = lines.join("\n").trim();
  const first = /^(#{1,6})\s/m.exec(body);
  if (!first) {
    if (!title) throw new Error("a region with no heading must declare title=");
    return `## ${title}\n\n${body}`;
  }
  const shift = 2 - first[1].length;
  const shifted = body.replace(/^(#{1,6})(\s)/gm, (_, hashes, space) => {
    const level = Math.min(6, Math.max(1, hashes.length + shift));
    return "#".repeat(level) + space;
  });
  return title ? `## ${title}\n\n${shifted}` : shifted;
}

export function build() {
  const head = readFileSync(preamble, "utf8").trimEnd();
  const marker =
    "<!-- GENERATED from README.md (its skill:begin regions) + skill-src/preamble.md.\n" +
    "     Edit those, then run: npm run skill:build -->";
  // The frontmatter block has to stay first for the host to parse the skill at all.
  const fm = /^---\n[\s\S]*?\n---\n/.exec(head);
  if (!fm) throw new Error("skill-src/preamble.md must start with a frontmatter block");
  const withMarker = `${fm[0]}\n${marker}\n\n${head.slice(fm[0].length).trimStart()}`;
  const body = [withMarker, ...regions(readFileSync(readme, "utf8"))].join("\n\n");
  return `${body.replace(/\n{3,}/g, "\n\n").trimEnd()}\n`;
}

// Guarded so the test can import regions()/build() without the CLI running on import.
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const generated = build();
  if (process.argv.includes("--check")) {
    if (readFileSync(skill, "utf8") !== generated) {
      console.error(
        "[coding-agents] ❌ skill/SKILL.md is out of date with the README.\n" +
          "  Run: npm run skill:build"
      );
      process.exit(1);
    }
    console.log("[coding-agents] ✅ skill/SKILL.md matches the README.");
  } else {
    writeFileSync(skill, generated);
    console.log(`[coding-agents] wrote ${skill} from the README + preamble.`);
  }
}

#!/usr/bin/env node
/**
 * Generate the two companion skills from the README + skill-src/:
 *
 *   skill/hindsight-coding-agent/SKILL.md        everyday use (skill-src/preamble.md + unmarked regions)
 *   skill/hindsight-coding-agent-setup/SKILL.md  install, configuration, diagnostics
 *                                                (skill-src/setup.md + regions marked skill="setup")
 *
 * A host loads a skill whole every time it triggers, so the everyday skill stays short and the
 * install/config/diagnostics reference only loads when that is what the task is about (#6749).
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
 *     <!-- skill:begin skill="setup" -->            … <!-- skill:end -->
 *
 * `skill="setup"` sends a region to the setup skill; a region without it goes to the everyday one.
 *
 * The skill-only half — what the agent does with the memory (tools, crediting, corrections) — has
 * no place in an npm README, so it stays in skill-src/preamble.md; skill-src/setup.md is the setup
 * skill's head. Each carries its skill's frontmatter and is emitted first.
 *
 * Heading levels are normalised per region: a region's first heading becomes `##` and everything
 * below it shifts by the same delta, so a marked `### Reference` reads as a top-level section of
 * the skill without the README having to know that. A region that starts mid-section (no heading of
 * its own) declares `title=` and gets an `##` synthesised.
 *
 * Run: node scripts/build-skill.mjs [--check]
 * `--check` fails when either SKILL.md is out of date instead of writing it (for CI — see
 * src/docs-freshness.test.ts, which is what actually gates it and unit-tests the marker parsing).
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pkgRoot = join(here, "..");
const readmePath = join(pkgRoot, "README.md");

/** Region `skill=` value → the skill it lands in. Keep the names in step with core/skill-dirs.ts. */
const SKILLS = {
  everyday: { name: "hindsight-coding-agent", head: "preamble.md" },
  setup: { name: "hindsight-coding-agent-setup", head: "setup.md" },
};

const BEGIN = /^<!--\s*skill:begin((?:\s+[a-z]+="[^"]*")*)\s*-->$/;
const END = /^<!--\s*skill:end\s*-->$/;

function attributes(text) {
  const out = {};
  for (const [, name, value] of text.matchAll(/([a-z]+)="([^"]*)"/g)) {
    if (name !== "title" && name !== "skill")
      throw new Error(`unknown <!-- skill:begin --> attribute ${name}=`);
    if (name === "skill" && !SKILLS[value]) throw new Error(`unknown skill="${value}"`);
    out[name] = value;
  }
  return out;
}

/** The marked regions of the README, in file order, with heading levels normalised. */
export function regions(src) {
  return markedRegions(src).map((r) => r.text);
}

/** Same, each with the skill it belongs to. */
function markedRegions(src) {
  const out = [];
  let current = null;
  for (const line of src.split("\n")) {
    const begin = BEGIN.exec(line);
    if (begin) {
      if (current) throw new Error("nested <!-- skill:begin -->");
      current = { ...attributes(begin[1]), lines: [] };
      continue;
    }
    if (END.test(line)) {
      if (!current) throw new Error("<!-- skill:end --> without a begin");
      out.push({ skill: current.skill ?? "everyday", text: render(current) });
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

/** Every generated skill file: { path, content }. */
export function build() {
  const readme = readFileSync(readmePath, "utf8");
  const marked = markedRegions(readme);
  return Object.entries(SKILLS).map(([key, { name, head: headFile }]) => {
    const head = readFileSync(join(pkgRoot, "skill-src", headFile), "utf8").trimEnd();
    const marker =
      `<!-- GENERATED from README.md (its skill:begin regions) + skill-src/${headFile}.\n` +
      "     Edit those, then run: npm run skill:build -->";
    // The frontmatter block has to stay first for the host to parse the skill at all.
    const fm = /^---\n[\s\S]*?\n---\n/.exec(head);
    if (!fm) throw new Error(`skill-src/${headFile} must start with a frontmatter block`);
    const withMarker = `${fm[0]}\n${marker}\n\n${head.slice(fm[0].length).trimStart()}`;
    const own = marked.filter((r) => r.skill === key).map((r) => r.text);
    const body = [withMarker, ...own].join("\n\n");
    return {
      path: join(pkgRoot, "skill", name, "SKILL.md"),
      content: `${body.replace(/\n{3,}/g, "\n\n").trimEnd()}\n`,
    };
  });
}

// Guarded so the test can import regions()/build() without the CLI running on import.
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const files = build();
  if (process.argv.includes("--check")) {
    const stale = files.filter(
      (f) => !existsSync(f.path) || readFileSync(f.path, "utf8") !== f.content
    );
    if (stale.length) {
      console.error(
        `[coding-agents] ❌ out of date with the README: ${stale.map((f) => relative(pkgRoot, f.path)).join(", ")}\n` +
          "  Run: npm run skill:build"
      );
      process.exit(1);
    }
    console.log("[coding-agents] ✅ both skills match the README.");
  } else {
    for (const f of files) {
      mkdirSync(dirname(f.path), { recursive: true });
      writeFileSync(f.path, f.content);
      console.log(`[coding-agents] wrote ${relative(pkgRoot, f.path)}`);
    }
  }
}

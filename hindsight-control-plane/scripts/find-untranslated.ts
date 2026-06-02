#!/usr/bin/env -S npx tsx
/**
 * Static finder for hardcoded English strings that look user-facing.
 *
 * Walks every .tsx file under src/components and src/app, parses with the
 * TypeScript AST, and reports:
 *   - JSXText nodes whose content looks like English prose (≥ 2 word chars,
 *     starts with a letter, not purely identifiers).
 *   - String literals passed as JSX attributes whose names typically render
 *     to the user: placeholder, title, aria-label, alt, label, description.
 *
 * Findings inside literal `t(...)` / `useTranslations()` call expressions are
 * skipped because next-intl is doing the translation.
 *
 * Exit code: 0 if no findings, 1 otherwise. Wire into CI for a hard guard,
 * or run locally as a periodic audit.
 *
 *   npx tsx scripts/find-untranslated.ts            # whole tree
 *   npx tsx scripts/find-untranslated.ts src/app/   # narrowed
 */

import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import ts from "typescript";

const ROOT = resolve(__dirname, "..");
const DEFAULT_TARGETS = ["src/components", "src/app"];

// JSX attributes whose string values render directly to the user.
const HUMAN_ATTRS = new Set([
  "placeholder",
  "title",
  "alt",
  "aria-label",
  "ariaLabel",
  "label",
  "description",
  "tooltip",
  "summary",
]);

// A finding is "interesting" if it contains a run of ≥ 2 alpha words and
// includes a lowercase letter (filters out CONSTANT_CASE and acronyms).
// Allow trailing punctuation, accents, etc.
const INTERESTING = /[A-Za-zÀ-ÿ]{2,}.+[a-z]/;

// Skip strings that look like identifiers, paths, URLs, mime types, or short
// HTTP / verb tokens we intentionally keep untranslated.
function isSkippable(s: string): boolean {
  const trimmed = s.trim();
  if (trimmed.length < 3) return true;
  if (!INTERESTING.test(trimmed)) return true;
  // HTML entity only (e.g. &middot;, &quot;, &mdash;)
  if (/^&[a-zA-Z]+;$/.test(trimmed)) return true;
  // ENV_VAR=value (uppercase identifier + optional value)
  if (/^[A-Z][A-Z0-9_]+(=\S+)?$/.test(trimmed)) return true;
  // JSON-shaped placeholder (starts with { or [, has quotes and braces)
  if (/^[{[].*["}\]]/.test(trimmed) && /["{}[\]]/.test(trimmed)) return true;
  // Example-prefixed sample (e.g., …) — short-form example data, not a UI label.
  if (/^e\.g\.[\s,]/.test(trimmed)) return true;
  // Math / formula soup (contains × ± · ÷ and few alpha chars OR plenty of underscores).
  if (/[×±·÷]/.test(trimmed)) return true;
  // Multi-line "key: value" example soup (newlines + colons, no sentence punctuation).
  if (trimmed.includes("\n") && !/[.?!]\s/.test(trimmed)) {
    const lines = trimmed.split("\n").map((l) => l.trim()).filter(Boolean);
    if (lines.every((l) => /^[a-z][\w.-]*:/.test(l))) return true;
  }
  // Short comma-separated tokens looking like sample inputs (no sentence-ending punctuation,
  // each token is a single capitalized word or short identifier).
  if (
    !/[.?!]/.test(trimmed) &&
    trimmed.includes(",") &&
    trimmed.split(/,\s*/).every((tok) => /^[A-Za-z][\w ]{0,15}$/.test(tok.trim()))
  ) {
    return true;
  }
  // Single identifier (no spaces, no punctuation other than _ . - / :)
  if (/^[\w./:-]+$/.test(trimmed) && !/\s/.test(trimmed)) return true;
  // URL / scheme
  if (/^(https?:|mailto:|tel:|data:)/.test(trimmed)) return true;
  // CSS class / tailwind-style soup
  if (/^[a-z-]+(:[a-z-]+)+$/.test(trimmed) && trimmed.includes(":")) return true;
  return false;
}

function walkDir(dir: string, files: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    if (name.startsWith(".") || name === "node_modules") continue;
    const full = join(dir, name);
    const st = statSync(full);
    if (st.isDirectory()) {
      walkDir(full, files);
    } else if (name.endsWith(".tsx")) {
      files.push(full);
    }
  }
  return files;
}

interface Finding {
  file: string;
  line: number;
  col: number;
  kind: "jsx-text" | "attr";
  attr?: string;
  text: string;
}

function scanFile(filePath: string): Finding[] {
  const src = readFileSync(filePath, "utf-8");
  const sourceFile = ts.createSourceFile(
    filePath,
    src,
    ts.ScriptTarget.Latest,
    /* setParentNodes */ true,
    ts.ScriptKind.TSX
  );
  const findings: Finding[] = [];

  // Track whether we're inside a `t(...)` / `tCommon(...)` / etc. call so we
  // don't flag strings that next-intl already handles.
  function inTranslationCall(node: ts.Node): boolean {
    let parent: ts.Node | undefined = node.parent;
    while (parent) {
      if (ts.isCallExpression(parent)) {
        const expr = parent.expression;
        // matches: t("...") / tFoo("...") / t.rich("...") / useTranslations(...)
        if (ts.isIdentifier(expr) && /^t([A-Z]\w*)?$|^useTranslations$/.test(expr.text)) {
          return true;
        }
        if (
          ts.isPropertyAccessExpression(expr) &&
          ts.isIdentifier(expr.expression) &&
          /^t([A-Z]\w*)?$/.test(expr.expression.text)
        ) {
          return true;
        }
      }
      parent = parent.parent;
    }
    return false;
  }

  function pos(node: ts.Node) {
    const { line, character } = sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile));
    return { line: line + 1, col: character + 1 };
  }

  function visit(node: ts.Node) {
    // 1. JSX text content
    if (ts.isJsxText(node)) {
      const text = node.text.replace(/\s+/g, " ").trim();
      if (text && !isSkippable(text)) {
        const { line, col } = pos(node);
        findings.push({ file: filePath, line, col, kind: "jsx-text", text });
      }
    }

    // 2. JSX attribute string literals on human-facing attrs
    if (ts.isJsxAttribute(node) && node.name && node.initializer) {
      const attrName = (node.name as ts.Identifier).text;
      if (HUMAN_ATTRS.has(attrName)) {
        let literal: string | undefined;
        if (ts.isStringLiteral(node.initializer)) {
          literal = node.initializer.text;
        } else if (
          ts.isJsxExpression(node.initializer) &&
          node.initializer.expression &&
          ts.isStringLiteral(node.initializer.expression)
        ) {
          literal = node.initializer.expression.text;
        } else if (
          ts.isJsxExpression(node.initializer) &&
          node.initializer.expression &&
          ts.isNoSubstitutionTemplateLiteral(node.initializer.expression)
        ) {
          literal = node.initializer.expression.text;
        }
        if (literal && !isSkippable(literal) && !inTranslationCall(node.initializer)) {
          const { line, col } = pos(node);
          findings.push({ file: filePath, line, col, kind: "attr", attr: attrName, text: literal });
        }
      }
    }

    ts.forEachChild(node, visit);
  }

  visit(sourceFile);
  return findings;
}

function main(argv: string[]): number {
  const targets = argv.length > 0 ? argv : DEFAULT_TARGETS;
  const files: string[] = [];
  for (const target of targets) {
    const abs = resolve(ROOT, target);
    const st = statSync(abs);
    if (st.isDirectory()) walkDir(abs, files);
    else if (abs.endsWith(".tsx")) files.push(abs);
  }
  files.sort();

  const all: Finding[] = [];
  for (const f of files) all.push(...scanFile(f));

  if (all.length === 0) {
    console.log("✓ no hardcoded user-facing strings found");
    return 0;
  }

  const byFile = new Map<string, Finding[]>();
  for (const f of all) {
    const rel = relative(ROOT, f.file);
    if (!byFile.has(rel)) byFile.set(rel, []);
    byFile.get(rel)!.push(f);
  }

  for (const [file, items] of byFile) {
    console.log(`\n${file}`);
    for (const it of items) {
      const tag = it.kind === "attr" ? `${it.attr}=` : "text:";
      console.log(`  ${file}:${it.line}:${it.col}  ${tag} ${JSON.stringify(it.text)}`);
    }
  }
  console.log(`\n${all.length} suspicious string${all.length === 1 ? "" : "s"} across ${byFile.size} file${byFile.size === 1 ? "" : "s"}.`);
  return 1;
}

const exitCode = main(process.argv.slice(2));
process.exit(exitCode);

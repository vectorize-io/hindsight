import { describe, it, expect } from "vitest";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import ts from "typescript";

import { defaultLocale } from "@/i18n/config";

/**
 * Static guard for translation keys *referenced in code*.
 *
 * The sibling parity test (`messages.test.ts`) only compares locale catalogs
 * against each other, so a key that is missing from *every* catalog — including
 * the default locale — passes parity unnoticed. And `find-untranslated.ts` does
 * the inverse, flagging hardcoded strings that are NOT wrapped in `t(...)`.
 * Neither one ever walks from a `t("key")` call site back to the catalog, so a
 * `t("dataView.filterActive")` with no entry anywhere only surfaces as a
 * runtime next-intl error in the browser.
 *
 * This test closes that gap. It walks every `.ts`/`.tsx` file under `src`,
 * resolves each `useTranslations("ns")` binding to its namespace, and asserts
 * that every statically-resolvable `t("key")` / `t.rich("key")` reference maps
 * to a leaf key present in the default-locale catalog.
 *
 * Limitations (intentional, to avoid false positives that would block
 * unrelated PRs):
 *   - Dynamic keys (`t(`a_${x}`)`, `t(someVar)`) can't be resolved statically
 *     and are skipped.
 *   - `t.has("key")` is skipped — it exists precisely to probe for keys that
 *     may be absent.
 *   - If a variable name is bound to more than one namespace in a single file
 *     (rare), a key is accepted when it resolves under *any* of them.
 */

const ROOT = resolve(__dirname, "..", "..");
const SRC_DIRS = ["src/components", "src/app", "src/lib", "src/hooks"];
const MESSAGES_DIR = join(ROOT, "src", "messages");

// next-intl accessor methods that take a message key as their first argument.
// `has` is deliberately excluded: it is the sanctioned way to test for a key's
// presence, so an absent key there is expected, not a bug.
const KEY_METHODS = new Set(["rich", "markup", "raw"]);

type Catalog = Record<string, unknown>;

function collectLeafKeys(obj: unknown, prefix = "", out: Set<string> = new Set()): Set<string> {
  if (obj && typeof obj === "object" && !Array.isArray(obj)) {
    for (const [k, v] of Object.entries(obj as Catalog)) {
      const path = prefix ? `${prefix}.${k}` : k;
      if (v && typeof v === "object" && !Array.isArray(v)) {
        collectLeafKeys(v, path, out);
      } else {
        out.add(path);
      }
    }
  }
  return out;
}

function walkDir(dir: string, files: string[] = []): string[] {
  let entries: string[];
  try {
    entries = readdirSync(dir);
  } catch {
    return files; // directory may not exist (e.g. no src/hooks) — that's fine
  }
  for (const name of entries) {
    if (name.startsWith(".") || name === "node_modules") continue;
    const full = join(dir, name);
    if (statSync(full).isDirectory()) {
      walkDir(full, files);
    } else if (name.endsWith(".tsx") || name.endsWith(".ts")) {
      files.push(full);
    }
  }
  return files;
}

interface MissingKey {
  file: string;
  line: number;
  attempted: string[];
}

// Unwrap `("key" as const)` / parenthesized expressions down to the inner node.
function unwrap(node: ts.Expression): ts.Expression {
  let cur = node;
  while (ts.isAsExpression(cur) || ts.isParenthesizedExpression(cur)) {
    cur = cur.expression;
  }
  return cur;
}

function staticString(node: ts.Expression): string | undefined {
  const inner = unwrap(node);
  if (ts.isStringLiteral(inner) || ts.isNoSubstitutionTemplateLiteral(inner)) {
    return inner.text;
  }
  return undefined;
}

interface VarBinding {
  namespaces: Set<string>; // resolvable namespaces this var name was bound to
  dynamic: boolean; // bound at least once to a non-static namespace
}

function scanFile(filePath: string, leafKeys: Set<string>): MissingKey[] {
  const src = readFileSync(filePath, "utf-8");
  const sf = ts.createSourceFile(filePath, src, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);

  // Pass 1: map each variable bound to `useTranslations(...)` to its namespace.
  const bindings = new Map<string, VarBinding>();
  const record = (name: string, ns: string | undefined) => {
    const b = bindings.get(name) ?? { namespaces: new Set<string>(), dynamic: false };
    if (ns === undefined) b.dynamic = true;
    else b.namespaces.add(ns);
    bindings.set(name, b);
  };

  const collectBindings = (node: ts.Node) => {
    if (
      ts.isVariableDeclaration(node) &&
      node.initializer &&
      ts.isCallExpression(node.initializer) &&
      ts.isIdentifier(node.initializer.expression) &&
      node.initializer.expression.text === "useTranslations" &&
      ts.isIdentifier(node.name)
    ) {
      const arg = node.initializer.arguments[0];
      const ns = arg === undefined ? "" : staticString(arg);
      record(node.name.text, ns);
    }
    ts.forEachChild(node, collectBindings);
  };
  collectBindings(sf);

  if (bindings.size === 0) return [];

  // Pass 2: find every `t("key")` / `t.rich("key")` call on a tracked binding.
  const missing: MissingKey[] = [];
  const resolves = (key: string, b: VarBinding): boolean => {
    if (b.dynamic) return true; // can't prove a violation
    for (const ns of b.namespaces) {
      if (leafKeys.has(ns === "" ? key : `${ns}.${key}`)) return true;
    }
    return false;
  };
  const attemptedKeys = (key: string, b: VarBinding): string[] =>
    [...b.namespaces].map((ns) => (ns === "" ? key : `${ns}.${key}`));

  const checkCalls = (node: ts.Node) => {
    if (ts.isCallExpression(node)) {
      const callee = node.expression;
      let varName: string | undefined;
      if (ts.isIdentifier(callee)) {
        varName = callee.text; // t("key")
      } else if (
        ts.isPropertyAccessExpression(callee) &&
        ts.isIdentifier(callee.expression) &&
        KEY_METHODS.has(callee.name.text)
      ) {
        varName = callee.expression.text; // t.rich("key")
      }

      if (varName) {
        const binding = bindings.get(varName);
        const firstArg = node.arguments[0];
        if (binding && firstArg) {
          const key = staticString(firstArg);
          // Static key that doesn't resolve under any bound namespace → missing.
          if (key !== undefined && !resolves(key, binding)) {
            const { line } = sf.getLineAndCharacterOfPosition(node.getStart(sf));
            missing.push({
              file: relative(ROOT, filePath),
              line: line + 1,
              attempted: attemptedKeys(key, binding),
            });
          }
        }
      }
    }
    ts.forEachChild(node, checkCalls);
  };
  checkCalls(sf);

  return missing;
}

describe("translation key references resolve against the default catalog", () => {
  const baseline = JSON.parse(readFileSync(join(MESSAGES_DIR, `${defaultLocale}.json`), "utf-8"));
  const leafKeys = collectLeafKeys(baseline);

  const files: string[] = [];
  for (const dir of SRC_DIRS) walkDir(join(ROOT, dir), files);
  files.sort();

  it("scans a non-trivial number of source files", () => {
    // Guards against a refactor silently pointing the walker at an empty tree.
    expect(files.length).toBeGreaterThan(10);
  });

  it("every static t(...) key exists in the default locale", () => {
    const missing: MissingKey[] = [];
    for (const f of files) missing.push(...scanFile(f, leafKeys));

    const report = missing.map(
      (m) =>
        `${m.file}:${m.line}  t("…") → none of [${m.attempted.join(", ")}] exist in ${defaultLocale}.json`
    );
    expect(
      report,
      `translation keys referenced in code but missing from ${defaultLocale}.json`
    ).toEqual([]);
  });
});

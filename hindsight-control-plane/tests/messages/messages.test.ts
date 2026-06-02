import { describe, it, expect } from "vitest";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { locales, defaultLocale } from "@/i18n/config";

/**
 * Static guard for locale catalog parity:
 *   1. Every locale declared in `i18n/config.ts` has a JSON file on disk.
 *   2. There are no extra JSON files that aren't declared as a locale.
 *   3. Every locale has exactly the same set of (nested) keys as the default
 *      locale. Missing or extra keys fail the build.
 *   4. Every leaf value is a non-empty string.
 *
 * This runs in CI as part of `npm test --workspace=hindsight-control-plane`,
 * so a missing translation or typoed key in any locale fails the build
 * before a release.
 */

type Catalog = Record<string, unknown>;

const MESSAGES_DIR = join(__dirname, "..", "..", "src", "messages");

function readCatalog(locale: string): Catalog {
  const raw = readFileSync(join(MESSAGES_DIR, `${locale}.json`), "utf-8");
  return JSON.parse(raw) as Catalog;
}

function collectKeys(obj: unknown, prefix = "", out: string[] = []): string[] {
  if (obj && typeof obj === "object" && !Array.isArray(obj)) {
    for (const [k, v] of Object.entries(obj as Catalog)) {
      const path = prefix ? `${prefix}.${k}` : k;
      if (v && typeof v === "object" && !Array.isArray(v)) {
        collectKeys(v, path, out);
      } else {
        out.push(path);
      }
    }
  }
  return out;
}

function getLeaf(obj: unknown, path: string): unknown {
  return path.split(".").reduce<unknown>((acc, segment) => {
    if (acc && typeof acc === "object" && segment in (acc as Catalog)) {
      return (acc as Catalog)[segment];
    }
    return undefined;
  }, obj);
}

describe("locale catalog parity", () => {
  it("every declared locale has a JSON file and no JSON files are undeclared", () => {
    const filesOnDisk = readdirSync(MESSAGES_DIR)
      .filter((f) => f.endsWith(".json"))
      .map((f) => f.replace(/\.json$/, ""))
      .sort();
    const declared = [...locales].sort();
    expect(filesOnDisk).toEqual(declared);
  });

  const baseline = readCatalog(defaultLocale);
  const baselineKeys = collectKeys(baseline).sort();

  it.each(locales.filter((l) => l !== defaultLocale))(
    "%s has exactly the same key set as %s",
    (locale) => {
      const catalog = readCatalog(locale);
      const keys = collectKeys(catalog).sort();
      const missing = baselineKeys.filter((k) => !keys.includes(k));
      const extra = keys.filter((k) => !baselineKeys.includes(k));
      expect({ missing, extra }, `locale "${locale}" diverges from "${defaultLocale}"`).toEqual({
        missing: [],
        extra: [],
      });
    }
  );

  it.each(locales)("%s has no empty string values", (locale) => {
    const catalog = readCatalog(locale);
    const empties: string[] = [];
    for (const path of collectKeys(catalog)) {
      const value = getLeaf(catalog, path);
      if (typeof value !== "string") {
        empties.push(`${path}: not a string (got ${typeof value})`);
      } else if (value.trim() === "") {
        empties.push(`${path}: empty`);
      }
    }
    expect(empties, `empty or non-string values in ${locale}.json`).toEqual([]);
  });
});

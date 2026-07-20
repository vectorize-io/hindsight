import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

function routeFiles(dir: string): string[] {
  const files: string[] = [];
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) {
      files.push(...routeFiles(path));
    } else if (entry === "route.ts") {
      files.push(path);
    }
  }
  return files;
}

describe("dataplane API route auth forwarding", () => {
  it("does not use global dataplane clients from app API routes", () => {
    const apiDir = join(process.cwd(), "src", "app", "api");
    const offenders = routeFiles(apiDir).filter((file) => {
      const source = readFileSync(file, "utf8");
      return /\bimport\s*\{[^}]*\b(?:hindsightClient|lowLevelClient)\b[^}]*\}\s*from\s*["']@\/lib\/hindsight-client["']/.test(
        source
      );
    });

    expect(offenders).toEqual([]);
  });
});

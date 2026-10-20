import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { expect, it } from "vitest";

const read = (path: string) => readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf8");

it("offers every audited HTTP action in the audit log filter", () => {
  // The filter list is hardcoded, so a new @audited route on the API is invisible
  // to it until someone adds it there too.
  const http = read("../../../hindsight-api-slim/hindsight_api/api/http.py");
  const audited = [...http.matchAll(/^\s*@audited\(\s*"([a-z_]+)"/gm)].map((m) => m[1]);
  expect(audited.length).toBeGreaterThan(30);

  const view = read("../../src/components/audit-logs-view.tsx");
  const offered = new Set([...view.matchAll(/value: "([a-z_]+)"/g)].map((m) => m[1]));
  expect(audited.filter((action) => !offered.has(action))).toEqual([]);
});

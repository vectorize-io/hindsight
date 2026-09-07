import { describe, it, expect } from "vitest";
import { readFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const packageDir = resolve(__dirname, "..");
const manifest = JSON.parse(readFileSync(resolve(packageDir, "package.json"), "utf-8"));
const packScript = readFileSync(resolve(packageDir, "scripts", "pack-manifest.mjs"), "utf-8");

/** The field names `pack-manifest.mjs` strips, read from the script itself. */
function strippedFields(): Set<string> {
  const list = packScript.match(/const DEV_ONLY_FIELDS = \[([^\]]*)\]/);
  if (!list) throw new Error("Could not locate DEV_ONLY_FIELDS in pack-manifest.mjs");
  return new Set([...list[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]));
}

// OpenClaw installs a plugin by running a full `npm install` inside the
// extracted package directory, so anything left in the published manifest is
// resolved on the user's machine. Shipping devDependencies broke every install
// on npm 10 once vitest 5 was published ("Cannot read properties of null
// (reading 'edgesOut')"). These assertions are the cheap guard; the real
// end-to-end proof is the `smoke-openclaw-install` CI job.
describe("published manifest", () => {
  it("wires the prepack/postpack pair that strips dev-only fields", () => {
    expect(manifest.scripts.prepack).toBe("node scripts/pack-manifest.mjs strip");
    expect(manifest.scripts.postpack).toBe("node scripts/pack-manifest.mjs restore");
  });

  it("strips every dev-only field the package actually declares", () => {
    // A new dev-only top-level field added to package.json without being added
    // to DEV_ONLY_FIELDS would ship to users and be installed by OpenClaw.
    const devOnlyPresent = ["devDependencies", "overrides"].filter((f) => f in manifest);
    const stripped = strippedFields();
    const unstripped = devOnlyPresent.filter((f) => !stripped.has(f));
    expect(unstripped, "dev-only package.json fields that would ship to users").toEqual([]);
  });

  it("keeps the runtime dependencies the plugin needs at runtime", () => {
    // The counterpart risk: stripping too much. These four are imported by
    // dist/ and must survive into the published manifest.
    const stripped = strippedFields();
    expect(stripped.has("dependencies")).toBe(false);
    expect(Object.keys(manifest.dependencies).sort()).toEqual([
      "@clack/prompts",
      "@vectorize-io/hindsight-agent-sdk",
      "@vectorize-io/hindsight-all",
      "@vectorize-io/hindsight-client",
    ]);
  });
});

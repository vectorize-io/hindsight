import { describe, it, expect } from "vitest";
import { readFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const manifestPath = resolve(__dirname, "..", "openclaw.plugin.json");
const typesPath = resolve(__dirname, "types.ts");

function extractPluginConfigKeys(): Set<string> {
  const src = readFileSync(typesPath, "utf-8");
  const block = src.match(/export interface PluginConfig\s*\{([\s\S]*?)^\}/m);
  if (!block) throw new Error("Could not locate PluginConfig interface in types.ts");
  // Capture identifier before `?:` or `:` at the start of a property line.
  // Matches lines like `  fooBar?: number;` and skips JSDoc / nested objects.
  const keys = new Set<string>();
  for (const match of block[1].matchAll(/^\s*(\w+)\??:\s*/gm)) {
    keys.add(match[1]);
  }
  return keys;
}

describe("openclaw.plugin.json", () => {
  it("is valid JSON", () => {
    const raw = readFileSync(manifestPath, "utf-8");
    expect(() => JSON.parse(raw)).not.toThrow();
  });

  it("has required top-level fields", () => {
    const manifest = JSON.parse(readFileSync(manifestPath, "utf-8"));
    expect(manifest.id).toBe("hindsight-openclaw");
    expect(manifest.name).toBeTypeOf("string");
    expect(manifest.configSchema).toBeDefined();
    expect(manifest.configSchema.properties).toBeDefined();
  });

  it("declares every PluginConfig field in configSchema and uiHints", () => {
    // Manifest must stay in sync with the runtime type or OpenClaw's strict
    // schema validation (additionalProperties: false) will reject user config
    // for fields the plugin actually accepts. (#1443 was the same drift class
    // applied to our internal whitelist.)
    const manifest = JSON.parse(readFileSync(manifestPath, "utf-8"));
    const schemaKeys = new Set<string>(Object.keys(manifest.configSchema.properties));
    const uiKeys = new Set<string>(Object.keys(manifest.uiHints ?? {}));
    const typeKeys = extractPluginConfigKeys();

    const missingFromSchema = [...typeKeys].filter((k) => !schemaKeys.has(k));
    const missingFromUi = [...typeKeys].filter((k) => !uiKeys.has(k));
    const extraInSchema = [...schemaKeys].filter((k) => !typeKeys.has(k));

    expect(missingFromSchema, "PluginConfig fields missing from configSchema").toEqual([]);
    expect(missingFromUi, "PluginConfig fields missing from uiHints").toEqual([]);
    expect(extraInSchema, "configSchema declares fields not in PluginConfig").toEqual([]);
  });
});

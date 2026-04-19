import { describe, it, expect } from "vitest";
import { readFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const manifestPath = resolve(__dirname, "..", "openclaw.plugin.json");

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
});

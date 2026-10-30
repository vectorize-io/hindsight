import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import HindsightOpencode2Plugin from "./opencode2";

describe("opencode package entry (#5823, #6138)", () => {
  it("points exports['./server'] at the root index.js and keeps main and '.' on the v1 plugin", () => {
    const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
    expect(pkg.exports["./server"]).toBe("./index.js");
    expect(pkg.main).toBe("dist/index.js");
    expect(pkg.exports["."].default).toBe("./dist/index.js");
    expect(readFileSync(new URL("../index.js", import.meta.url), "utf8")).toContain(
      'from "./dist/opencode2.js"'
    );
  });

  it("carries both contracts: v2's {id, setup} and v1's plugin function as server", () => {
    expect(HindsightOpencode2Plugin.id).toBe("hindsight-opencode2");
    expect(typeof HindsightOpencode2Plugin.setup).toBe("function");
    expect(typeof HindsightOpencode2Plugin.server).toBe("function");
  });
});

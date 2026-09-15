import { createRequire } from "node:module";

import { describe, expect, it } from "vitest";

const { resolveBindHost } = createRequire(import.meta.url)("../../bin/cli.js");

describe("cli resolveBindHost (#1926)", () => {
  it("binds a loopback literal as localhost, keeping the requested family", () => {
    expect(resolveBindHost("127.0.0.1", undefined)).toEqual({
      hostname: "localhost",
      nodeOptions: "--dns-result-order=ipv4first",
    });
    for (const v6 of ["::1", "[::1]"]) {
      expect(resolveBindHost(v6, "")).toEqual({
        hostname: "localhost",
        nodeOptions: "--dns-result-order=ipv6first",
      });
    }
  });

  it("appends to an operator's NODE_OPTIONS instead of replacing it", () => {
    expect(resolveBindHost("127.0.0.1", "--max-old-space-size=2048").nodeOptions).toBe(
      "--max-old-space-size=2048 --dns-result-order=ipv4first"
    );
  });

  it("leaves every other hostname and NODE_OPTIONS untouched", () => {
    for (const host of ["0.0.0.0", "localhost", "cp.internal"]) {
      expect(resolveBindHost(host, "--foo")).toEqual({ hostname: host, nodeOptions: "--foo" });
    }
  });
});

import type { Stats } from "node:fs";
import { afterEach, describe, expect, it, vi } from "vitest";

const lstatSync = vi.fn();
vi.mock("node:fs", async () => ({
  ...(await vi.importActual<typeof import("node:fs")>("node:fs")),
  lstatSync,
}));

const { probeGitLayout } = await import("./git-layout");

function transient(code: string): Error {
  return Object.assign(new Error(code), { code });
}

describe("transient probe failures", () => {
  afterEach(() => vi.clearAllMocks());

  it("retries, then answers normally once the pressure clears", () => {
    lstatSync
      .mockImplementationOnce(() => {
        throw transient("EAGAIN"); // spawn-pressure's filesystem twin — the #3950 condition
      })
      .mockImplementation(() => ({ isDirectory: () => true }) as Stats);

    const layout = probeGitLayout("/home/me/dev/myrepo-wt1");
    expect(layout.status).toBe("resolved");
  });

  it("reports failure — never absence — when every attempt is starved", () => {
    lstatSync.mockImplementation(() => {
      throw transient("EMFILE");
    });

    expect(probeGitLayout("/home/me/dev/myrepo-wt1")).toEqual({
      status: "failed",
      reason: "EMFILE",
    });
  });

  it("treats an ordinary missing path as absence, with no retry", () => {
    lstatSync.mockImplementation(() => {
      throw transient("ENOENT");
    });

    expect(probeGitLayout("/nowhere/at/all")).toEqual({ status: "absent" });
  });
});

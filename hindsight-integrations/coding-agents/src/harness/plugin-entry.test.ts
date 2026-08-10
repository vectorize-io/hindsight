import { describe, expect, it } from "vitest";
import { resolveProjectDirectory } from "./plugin-entry";

describe("resolveProjectDirectory", () => {
  it("prefers a meaningful worktree", () => {
    expect(
      resolveProjectDirectory({
        worktree: "/Users/me/project",
        directory: "/Users/me/workspace",
      })
    ).toBe("/Users/me/project");
  });

  it("falls back from the POSIX root to the workspace directory", () => {
    expect(resolveProjectDirectory({ worktree: "/", directory: "/Users/me/workspace" })).toBe(
      "/Users/me/workspace"
    );
  });

  it.each(["C:\\", "C:/"])(
    "falls back from the Windows root %s to the workspace directory",
    (worktree) => {
      expect(resolveProjectDirectory({ worktree, directory: "C:\\Users\\me\\workspace" })).toBe(
        "C:\\Users\\me\\workspace"
      );
    }
  );

  it("falls back when worktree is empty", () => {
    expect(resolveProjectDirectory({ worktree: "", directory: "/Users/me/workspace" })).toBe(
      "/Users/me/workspace"
    );
  });
});

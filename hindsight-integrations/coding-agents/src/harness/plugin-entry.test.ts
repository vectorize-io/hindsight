import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { RuntimeCore } from "../core/runtime";
import { createPluginEntry, resolveProjectDirectory } from "./plugin-entry";

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

// opencode and Kilo read sessions back from the host's server, which keeps every one of them: the
// same call is what replays a session whose write-back failed before the host exited (#6559).
describe("createPluginEntry past-session source", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
  });

  it.each(["opencode", "kilo"])("%s replays through GET /session/{id}/message", async (harness) => {
    vi.stubEnv("HINDSIGHT_API_TOKEN", "test-key"); // Cloud without a token counts as disabled
    vi.spyOn(RuntimeCore.prototype, "seedIfCold").mockResolvedValue();
    const setPast = vi.spyOn(RuntimeCore.prototype, "setPastSessionSource");
    const messages = vi.fn(async () => ({ data: [] }));
    const dir = mkdtempSync(join(tmpdir(), "hs-plugin-entry-"));

    await createPluginEntry(harness)({
      directory: dir,
      worktree: dir,
      client: { session: { messages } },
    } as never);

    expect(setPast).toHaveBeenCalledOnce();
    await setPast.mock.calls[0]![0]("ses_old");
    expect(messages).toHaveBeenCalledWith({ path: { id: "ses_old" } });
  });
});

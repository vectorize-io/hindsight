import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { claudeProjectDir, importLocalHistory } from "./history";

let home: string;
afterEach(() => {
  if (home) rmSync(home, { recursive: true, force: true });
});

function newHome(): string {
  home = mkdtempSync(join(tmpdir(), "hs-history-"));
  return home;
}

const claudeLine = (role: string, text: string, cwd?: string) =>
  JSON.stringify({
    type: role,
    ...(cwd ? { cwd } : {}),
    message: { role, content: [{ type: "text", text }] },
  });

describe("local history import", () => {
  it("reads Claude sessions from the project directory for THIS repo only", () => {
    const h = newHome();
    const repo = "/Users/x/dev/myrepo";
    const dir = claudeProjectDir(repo, h);
    mkdirSync(dir, { recursive: true });
    writeFileSync(
      join(dir, "s1.jsonl"),
      `${claudeLine("user", "why retry 429?", repo)}\n${claudeLine("assistant", "because backpressure")}\n`
    );
    // A different project must not leak into this repo's import.
    const other = claudeProjectDir("/Users/x/dev/otherrepo", h);
    mkdirSync(other, { recursive: true });
    writeFileSync(
      join(other, "s2.jsonl"),
      `${claudeLine("user", "unrelated", "/Users/x/dev/otherrepo")}\n`
    );

    const r = importLocalHistory("claude-code", repo, h);
    expect(r.supported).toBe(true);
    expect(r.sessions).toHaveLength(1);
    expect(r.sessions[0].id).toBe("s1");
    expect(JSON.stringify(r.sessions[0].turns)).toContain("why retry 429?");
    expect(JSON.stringify(r.sessions)).not.toContain("unrelated");
  });

  // Claude Code encodes EVERY non-alphanumeric character as `-`, not just `/` and `.`. Encoding
  // only the separators left whole repositories invisible to `--import-conversations`: the install
  // reported "no past sessions found on disk" while the transcripts sat there under the real name.
  // A space is the common case in the field (`~/Documents/My Projects/...`), not just underscores.
  it.each([
    ["underscores", "/Users/x/dev/my_project", "-Users-x-dev-my-project"],
    ["spaces", "/Users/x/My Projects/app", "-Users-x-My-Projects-app"],
    ["mixed punctuation", "/Users/x/dev/hs+odd@repo v2", "-Users-x-dev-hs-odd-repo-v2"],
    ["dots", "/Users/x/dev/repo.git", "-Users-x-dev-repo-git"],
  ])("reads Claude sessions when the repository path contains %s", (_label, repo, encoded) => {
    const h = newHome();
    const dir = join(h, ".claude", "projects", encoded);
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, "s1.jsonl"), `${claudeLine("user", "found me", repo)}\n`);

    const r = importLocalHistory("claude-code", repo, h);

    expect(r.sessions).toHaveLength(1);
    expect(JSON.stringify(r.sessions)).toContain("found me");
  });

  // Case is preserved and runs are NOT collapsed, so the encoding stays a 1:1 substitution.
  it("preserves case and does not collapse runs of separators", () => {
    const h = newHome();
    expect(claudeProjectDir("/tmp/a-1/-crewAI", h)).toBe(
      join(h, ".claude", "projects", "-tmp-a-1--crewAI")
    );
  });

  // The lossy encoding must never be trusted on its own: `my_repo` and `my-repo` collide, so the
  // cwd recorded INSIDE the transcript is what actually attributes a session to a repository.
  it("does not import a sibling repository that encodes to the same directory name", () => {
    const h = newHome();
    const dir = join(h, ".claude", "projects", "-Users-x-dev-my-repo");
    mkdirSync(dir, { recursive: true });
    writeFileSync(
      join(dir, "s1.jsonl"),
      `${claudeLine("user", "the other one", "/Users/x/dev/my-repo")}\n`
    );

    const r = importLocalHistory("claude-code", "/Users/x/dev/my_repo", h);

    expect(r.sessions).toHaveLength(0);
    expect(JSON.stringify(r.sessions)).not.toContain("the other one");
  });

  it("matches Codex rollouts by the cwd in their session_meta header", () => {
    const h = newHome();
    const day = join(h, ".codex", "sessions", "2026", "08", "03");
    mkdirSync(day, { recursive: true });
    const rollout = (cwd: string, text: string) =>
      `${JSON.stringify({ type: "session_meta", payload: { id: "abc", cwd } })}\n` +
      `${JSON.stringify({ type: "response_item", payload: { type: "message", role: "user", content: [{ type: "input_text", text }] } })}\n`;
    writeFileSync(join(day, "mine.jsonl"), rollout("/repo/mine", "mine"));
    writeFileSync(join(day, "theirs.jsonl"), rollout("/repo/theirs", "theirs"));

    const r = importLocalHistory("codex", "/repo/mine", h);
    expect(r.sessions).toHaveLength(1);
    expect(JSON.stringify(r.sessions)).toContain("mine");
    expect(JSON.stringify(r.sessions)).not.toContain("theirs");
  });

  it("handles a session_meta header larger than one read chunk", () => {
    const h = newHome();
    const day = join(h, ".codex", "sessions", "2026", "08", "03");
    mkdirSync(day, { recursive: true });
    // Codex embeds the agent's full base instructions in this single line — tens of KB. Reading a
    // fixed-size prefix truncated it mid-JSON, so EVERY rollout was skipped and the import
    // silently returned nothing.
    const huge = "x".repeat(200_000);
    writeFileSync(
      join(day, "big.jsonl"),
      `${JSON.stringify({ type: "session_meta", payload: { id: "big", cwd: "/repo/mine", instructions: huge } })}\n` +
        `${JSON.stringify({ type: "response_item", payload: { type: "message", role: "user", content: [{ type: "input_text", text: "found me" }] } })}\n`
    );
    const r = importLocalHistory("codex", "/repo/mine", h);
    expect(r.sessions).toHaveLength(1);
    expect(JSON.stringify(r.sessions)).toContain("found me");
  });

  it("reports SQLite-backed harnesses as unsupported with a reason, not an empty success", () => {
    const h = newHome();
    for (const harness of ["opencode", "kilo", "cursor-cli", "cline-cli"]) {
      const r = importLocalHistory(harness, "/repo/mine", h);
      expect(r.supported).toBe(false);
      expect(r.reason).toMatch(/SQLite/);
      expect(r.sessions).toEqual([]);
    }
  });

  it("returns empty (not an error) when a supported harness has no history", () => {
    const h = newHome();
    const r = importLocalHistory("claude-code", "/repo/none", h);
    expect(r.supported).toBe(true);
    expect(r.sessions).toEqual([]);
  });
});

/**
 * `importLocalHistory` documents that it never throws, and its caller (`--import-conversations`)
 * takes it at its word. A stray entry must therefore cost that entry, not the whole run (#3771).
 */
describe("a junk entry costs itself, not the import", () => {
  it("skips a regular file sitting where a Claude project directory was expected", () => {
    const h = newHome();
    const repo = "/Users/x/dev/myrepo";
    const dir = claudeProjectDir(repo, h);
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, "s1.jsonl"), `${claudeLine("user", "the real session", repo)}\n`);
    // Named like the project dir of a SUBDIRECTORY, so the prefix filter hands it to the listing;
    // without the guard readdirSync threw ENOTDIR and nothing was imported.
    writeFileSync(claudeProjectDir("/Users/x/dev/myrepo/sub", h), "not a folder");

    const r = importLocalHistory("claude-code", repo, h);

    expect(r.supported).toBe(true);
    expect(r.sessions).toHaveLength(1);
    expect(JSON.stringify(r.sessions)).toContain("the real session");
  });

  it("skips a regular file where ~/.claude/projects itself was expected", () => {
    const h = newHome();
    mkdirSync(join(h, ".claude"), { recursive: true });
    writeFileSync(join(h, ".claude", "projects"), "not a folder");

    const r = importLocalHistory("claude-code", "/Users/x/dev/myrepo", h);

    expect(r.supported).toBe(true);
    expect(r.sessions).toEqual([]);
  });
});

describe("attribution must be proven, never guessed", () => {
  it("skips a Claude session that records no cwd instead of trusting the directory name", () => {
    const h = newHome();
    const repo = "/Users/x/dev/myrepo";
    const dir = claudeProjectDir(repo, h);
    mkdirSync(dir, { recursive: true });
    // Sits in this repo's project directory but proves nothing about where it ran.
    writeFileSync(join(dir, "nocwd.jsonl"), `${claudeLine("user", "ambiguous")}\n`);

    const r = importLocalHistory("claude-code", repo, h);
    expect(r.sessions).toEqual([]);
    expect(r.unattributed).toBe(1);
  });

  it("does not pull a SIBLING repo's sessions in via the ambiguous name encoding", () => {
    const h = newHome();
    // "/Users/x/dev/repo-sub" and "/Users/x/dev/repo/sub" both encode to the same prefix shape,
    // so only the recorded cwd can tell them apart.
    const sibling = claudeProjectDir("/Users/x/dev/repo-sub", h);
    mkdirSync(sibling, { recursive: true });
    writeFileSync(
      join(sibling, "s.jsonl"),
      `${claudeLine("user", "sibling repo work", "/Users/x/dev/repo-sub")}\n`
    );

    const r = importLocalHistory("claude-code", "/Users/x/dev/repo", h);
    expect(JSON.stringify(r.sessions)).not.toContain("sibling repo work");
  });

  it("includes a session run in a SUBDIRECTORY of the repo", () => {
    const h = newHome();
    const repo = "/Users/x/dev/repo";
    const sub = claudeProjectDir("/Users/x/dev/repo/packages/api", h);
    mkdirSync(sub, { recursive: true });
    writeFileSync(
      join(sub, "s.jsonl"),
      `${claudeLine("user", "work in a subpackage", "/Users/x/dev/repo/packages/api")}\n`
    );

    const r = importLocalHistory("claude-code", repo, h);
    expect(JSON.stringify(r.sessions)).toContain("work in a subpackage");
  });
});

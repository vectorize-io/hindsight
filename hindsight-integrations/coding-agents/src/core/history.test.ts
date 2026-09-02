import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { claudeProjectDir, importLocalHistory, piSessionDir } from "./history";

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

/** One entry of a stored pi/Prime Agent session log. */
const piHeader = (id: string, cwd?: string) =>
  JSON.stringify({
    type: "session",
    version: 3,
    id,
    timestamp: "2026-08-24T12:00:00.000Z",
    ...(cwd ? { cwd } : {}),
  });
const piMessage = (role: string, text: string) =>
  JSON.stringify({
    type: "message",
    id: "e1",
    timestamp: "2026-08-24T12:00:01.000Z",
    message: { role, content: [{ type: "text", text }] },
  });
const piLog = (id: string, cwd: string | undefined, ...turns: string[]) =>
  `${[piHeader(id, cwd), ...turns].join("\n")}\n`;

/** Write a stored session where each host actually keeps it: pi under an encoded per-cwd folder,
 *  Prime Agent flat. */
function writePiSession(home: string, cwd: string, id: string, ...turns: string[]): void {
  const dir = join(home, ".pi", "agent", "sessions", piSessionDir(cwd));
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, `2026-08-24T12-00-00-000Z_${id}.jsonl`), piLog(id, cwd, ...turns));
}
function writePrimeSession(
  home: string,
  cwd: string | undefined,
  id: string,
  ...turns: string[]
): void {
  const dir = join(home, ".prime", "agent", "sessions");
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, `${id}.jsonl`), piLog(id, cwd, ...turns));
}

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
    for (const harness of ["opencode", "opencode2", "kilo", "cursor-cli", "cline-cli"]) {
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

/**
 * pi and Prime Agent write the same session schema but lay the files out differently, so each host
 * needs its own enumeration proved — the shared body that reads a header is only reached if the
 * files were found at all.
 */
describe("pi-family history import", () => {
  it("reads pi sessions from the encoded per-directory folder for THIS repo only", () => {
    const h = newHome();
    const repo = "/Users/x/dev/myrepo";
    writePiSession(
      h,
      repo,
      "s1",
      piMessage("user", "why retry 429?"),
      piMessage("assistant", "backpressure")
    );
    writePiSession(h, "/Users/x/dev/otherrepo", "s2", piMessage("user", "unrelated"));

    const r = importLocalHistory("pi", repo, h);
    expect(r.supported).toBe(true);
    expect(r.sessions).toHaveLength(1);
    expect(r.sessions[0].id).toBe("s1");
    expect(JSON.stringify(r.sessions[0].turns)).toContain("why retry 429?");
    expect(JSON.stringify(r.sessions)).not.toContain("unrelated");
  });

  it("includes a pi session started in a SUBDIRECTORY of the repo", () => {
    const h = newHome();
    writePiSession(
      h,
      "/Users/x/dev/repo/packages/api",
      "s1",
      piMessage("user", "work in a subpackage")
    );
    expect(JSON.stringify(importLocalHistory("pi", "/Users/x/dev/repo", h).sessions)).toContain(
      "work in a subpackage"
    );
  });

  // The folder name is lossy in exactly the way the Claude encoding is: "/Users/x/dev/repo-sub"
  // matches the prefix scan for "/Users/x/dev/repo", so only the header cwd can separate them.
  it("does not pull a SIBLING repo's pi sessions in via the ambiguous folder name", () => {
    const h = newHome();
    writePiSession(h, "/Users/x/dev/repo-sub", "s1", piMessage("user", "sibling repo work"));
    expect(JSON.stringify(importLocalHistory("pi", "/Users/x/dev/repo", h).sessions)).not.toContain(
      "sibling repo work"
    );
  });

  // Regression: `~/.pi/agent/sessions` itself as a regular file passed the existsSync check and the
  // root readdirSync threw ENOTDIR out of importLocalHistory — which documents that it never throws,
  // and whose caller does not catch. One junk file killed the whole --import-conversations run.
  it("treats a stray file where the pi sessions ROOT was expected as no history", () => {
    const h = newHome();
    mkdirSync(join(h, ".pi", "agent"), { recursive: true });
    writeFileSync(join(h, ".pi", "agent", "sessions"), "not a folder");

    expect(importLocalHistory("pi", "/Users/x/dev/myrepo", h)).toEqual({
      supported: true,
      sessions: [],
      unattributed: 0,
    });
  });

  // Same shape one level down: a regular file named like a session folder must cost that one
  // entry, not the run.
  it("skips a stray file where a pi session folder was expected, keeping the real sessions", () => {
    const h = newHome();
    const repo = "/Users/x/dev/myrepo";
    writePiSession(h, repo, "s1", piMessage("user", "real work"));
    writeFileSync(join(h, ".pi", "agent", "sessions", piSessionDir(`${repo}/sub`)), "not a folder");

    const r = importLocalHistory("pi", repo, h);
    expect(r.sessions).toHaveLength(1);
    expect(JSON.stringify(r.sessions)).toContain("real work");
  });

  it("reads Prime Agent sessions out of its FLAT directory, attributing by header cwd", () => {
    const h = newHome();
    const repo = "/Users/x/dev/myrepo";
    writePrimeSession(h, repo, "p1", piMessage("user", "which statuses retry?"));
    writePrimeSession(h, "/Users/x/dev/otherrepo", "p2", piMessage("user", "unrelated"));

    const r = importLocalHistory("prime-agent", repo, h);
    expect(r.sessions).toHaveLength(1);
    expect(r.sessions[0].id).toBe("p1");
    expect(JSON.stringify(r.sessions)).not.toContain("unrelated");
  });

  it("skips a session whose header records no cwd rather than trusting where the file sits", () => {
    const h = newHome();
    writePrimeSession(h, undefined, "p1", piMessage("user", "ambiguous"));
    const r = importLocalHistory("prime-agent", "/Users/x/dev/myrepo", h);
    expect(r.sessions).toEqual([]);
    expect(r.unattributed).toBe(1);
  });

  it("reports no sessions, not an error, when the host has never run on this machine", () => {
    const h = newHome();
    for (const harness of ["pi", "prime-agent"]) {
      const r = importLocalHistory(harness, "/Users/x/dev/myrepo", h);
      expect(r.supported).toBe(true);
      expect(r.sessions).toEqual([]);
    }
  });
});

describe("piSessionDir", () => {
  // Verified against pi 0.84.2's SessionManager: one leading separator dropped, `/`, `\` and `:`
  // mapped to `-`, everything else — dots, spaces, case — left alone.
  it.each([
    ["/Users/x/dev/myrepo", "--Users-x-dev-myrepo--"],
    ["/private/tmp", "--private-tmp--"],
    ["/Users/x/dev/repo.git", "--Users-x-dev-repo.git--"],
    ["/Users/x/My Projects/App", "--Users-x-My Projects-App--"],
    // Windows: no leading separator to strip, and the drive colon maps to `-` like the
    // separators, so the drive letter survives as its own segment.
    ["C:\\Users\\x\\repo", "--C--Users-x-repo--"],
  ])("encodes %s", (repo, encoded) => {
    expect(piSessionDir(repo)).toBe(encoded);
  });
});

describe("importLocalHistory — dcode", () => {
  const REPO = "/Users/x/dev/repo";

  // The reader honours DEEPAGENTS_HOME (as the dsh reader honours DSH_HOME); a developer who has
  // it exported must not have it steer these fixtures.
  beforeEach(() => {
    vi.stubEnv("DEEPAGENTS_HOME", "");
    delete process.env.DEEPAGENTS_HOME;
  });
  afterEach(() => vi.unstubAllEnvs());

  /** Dcode names a transcript `<readable prefix>--<sha256 of the full thread id>.jsonl`. */
  function writeTranscript(home: string, threadId: string, lines: object[]): void {
    const dir = join(home, ".deepagents", "transcripts");
    mkdirSync(dir, { recursive: true });
    const digest = createHash("sha256").update(threadId, "utf8").digest("hex");
    writeFileSync(
      join(dir, `${threadId.slice(0, 32)}--${digest}.jsonl`),
      lines.map((l) => JSON.stringify({ schema_version: 1, ...l })).join("\n")
    );
  }

  const threadsJson = (rows: object[]): string =>
    JSON.stringify({ schema_version: 1, command: "threads list", data: rows });

  it("attributes threads to the repo via `dcode threads list --json`", () => {
    const h = newHome();
    writeTranscript(h, "01a05876-0764-7e33-a52f-8286acfe9e19", [
      { role: "user", content: "fix the retry policy" },
      { role: "assistant", content: [{ type: "text", text: "done" }] },
    ]);
    writeTranscript(h, "01a05879-5c46-7933-b094-c2a7c8f154a6", [
      { role: "user", content: "unrelated project work" },
    ]);

    const r = importLocalHistory("dcode", REPO, h, () =>
      threadsJson([
        { thread_id: "01a05876-0764-7e33-a52f-8286acfe9e19", cwd: REPO },
        { thread_id: "01a05879-5c46-7933-b094-c2a7c8f154a6", cwd: "/Users/x/dev/other" },
      ])
    );

    expect(r.supported).toBe(true);
    expect(r.sessions).toHaveLength(1);
    expect(r.sessions[0]!.id).toBe("01a05876-0764-7e33-a52f-8286acfe9e19");
    expect(r.sessions[0]!.turns.map((t) => t.text)).toEqual(["fix the retry policy", "done"]);
    expect(JSON.stringify(r.sessions)).not.toContain("unrelated project work");
  });

  it("includes a thread run in a SUBDIRECTORY of the repo", () => {
    const h = newHome();
    writeTranscript(h, "t-sub", [{ role: "user", content: "work in a subpackage" }]);
    const r = importLocalHistory("dcode", REPO, h, () =>
      threadsJson([{ thread_id: "t-sub", cwd: `${REPO}/packages/api` }])
    );
    expect(JSON.stringify(r.sessions)).toContain("work in a subpackage");
  });

  it("does not pull in a SIBLING repo whose path merely shares the prefix", () => {
    const h = newHome();
    writeTranscript(h, "t-sibling", [{ role: "user", content: "sibling repo work" }]);
    const r = importLocalHistory("dcode", REPO, h, () =>
      threadsJson([{ thread_id: "t-sibling", cwd: `${REPO}-sub` }])
    );
    expect(r.sessions).toHaveLength(0);
  });

  it("counts a thread the CLI could not attribute rather than guessing", () => {
    const h = newHome();
    writeTranscript(h, "t-nocwd", [{ role: "user", content: "orphan" }]);
    const r = importLocalHistory("dcode", REPO, h, () => threadsJson([{ thread_id: "t-nocwd" }]));
    expect(r.sessions).toHaveLength(0);
    expect(r.unattributed).toBe(1);
  });

  it("skips a listed thread whose transcript was never materialized", () => {
    const h = newHome();
    mkdirSync(join(h, ".deepagents", "transcripts"), { recursive: true });
    const r = importLocalHistory("dcode", REPO, h, () =>
      threadsJson([{ thread_id: "t-missing", cwd: REPO }])
    );
    expect(r.supported).toBe(true);
    expect(r.sessions).toHaveLength(0);
  });

  it("reports unsupported — with the reason — when the dcode CLI cannot be run", () => {
    const h = newHome();
    writeTranscript(h, "t1", [{ role: "user", content: "hi" }]);
    const r = importLocalHistory("dcode", REPO, h, () => {
      throw new Error("spawn dcode ENOENT");
    });
    expect(r.supported).toBe(false);
    expect(r.reason).toMatch(/dcode CLI is not runnable/);
  });

  it("refuses an unrecognized threads-list schema instead of importing nothing silently", () => {
    const h = newHome();
    writeTranscript(h, "t1", [{ role: "user", content: "hi" }]);
    const r = importLocalHistory("dcode", REPO, h, () =>
      JSON.stringify({ schema_version: 2, data: [{ thread_id: "t1", cwd: REPO }] })
    );
    expect(r.supported).toBe(false);
    expect(r.reason).toMatch(/schema_version 1/);
  });

  it("reports nothing to import when dcode was never used on this machine", () => {
    const h = newHome();
    const r = importLocalHistory("dcode", REPO, h, () => {
      throw new Error("must not be called — there is no transcripts dir to attribute");
    });
    expect(r).toEqual({ supported: true, sessions: [] });
  });
});

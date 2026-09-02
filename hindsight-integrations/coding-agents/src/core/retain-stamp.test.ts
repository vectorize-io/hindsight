import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { buildRetainStamp } from "./retain-stamp";

let repo: string;
const ctx = (over: Partial<Parameters<typeof buildRetainStamp>[1]> = {}) => ({
  directory: repo,
  harness: "codex",
  bankId: "shared-bank",
  sessionId: "sess-1",
  ...over,
});

beforeEach(() => {
  repo = mkdtempSync(join(tmpdir(), "hs-stamp-repo-"));
  execFileSync("git", ["init", "-q"], { cwd: repo });
});

afterEach(() => {
  rmSync(repo, { recursive: true, force: true });
  delete process.env.HINDSIGHT_USER_ID;
  delete process.env.HINDSIGHT_CHANNEL_ID;
});

describe("buildRetainStamp", () => {
  it("adds nothing when neither setting is configured", () => {
    expect(buildRetainStamp({}, ctx())).toEqual({ tags: [], metadata: {} });
    expect(buildRetainStamp({ retainTags: [], retainMetadata: {} }, ctx())).toEqual({
      tags: [],
      metadata: {},
    });
  });

  it("resolves {gitProject} to the repo name — the #3269 use case", () => {
    const { tags, metadata } = buildRetainStamp(
      { retainTags: ["project:{gitProject}"], retainMetadata: { repo: "{gitProject}" } },
      ctx()
    );
    const name = repo.split("/").pop()!;
    expect(tags).toEqual([`project:${name}`]);
    expect(metadata).toEqual({ repo: name });
  });

  it("resolves {gitProject} from the session root outside a repository", () => {
    // The stamp must name the project its BANK ID names, and outside a repo the bank id comes from
    // where the session started rather than the agent's live cwd (#3563).
    const plain = mkdtempSync(join(tmpdir(), "hs-stamp-plain-"));
    const sub = join(plain, "analysis");
    mkdirSync(sub);
    try {
      const { tags } = buildRetainStamp(
        { retainTags: ["project:{gitProject}", "dir:{project}"] },
        ctx({ directory: sub, sessionRoot: plain })
      );
      // {project} stays on the live working directory — it is documented as exactly that.
      expect(tags).toEqual([`project:${plain.split("/").pop()!}`, "dir:analysis"]);
    } finally {
      rmSync(plain, { recursive: true, force: true });
    }
  });

  it("resolves the rest of the vocabulary", () => {
    process.env.HINDSIGHT_USER_ID = "nico";
    process.env.HINDSIGHT_CHANNEL_ID = "team";
    const { tags } = buildRetainStamp(
      {
        retainTags: ["h:{harness}", "b:{bankId}", "s:{sessionId}", "u:{user}", "c:{channel}"],
      },
      ctx()
    );
    expect(tags).toEqual(["h:codex", "b:shared-bank", "s:sess-1", "u:nico", "c:team"]);
  });

  it("resolves {sessionId} to unknown for non-session documents", () => {
    expect(
      buildRetainStamp({ retainTags: ["session:{sessionId}"] }, ctx({ sessionId: undefined })).tags
    ).toEqual(["session:unknown"]);
  });

  it("resolves {project} from the directory basename, without git", () => {
    const { tags } = buildRetainStamp({ retainTags: ["p:{project}"] }, ctx());
    expect(tags).toEqual([`p:${repo.split("/").pop()}`]);
  });

  it("stamps a literal tag with no placeholders unchanged", () => {
    expect(buildRetainStamp({ retainTags: ["env:work"] }, ctx()).tags).toEqual(["env:work"]);
  });

  it("substitutes 'unknown' for a typo'd placeholder rather than an empty string", () => {
    // An empty substitution yields a tag like `project:` that looks configured but matches nothing;
    // applyTemplate also names the offending setting on stderr.
    expect(buildRetainStamp({ retainTags: ["project:{gitproject}"] }, ctx()).tags).toEqual([
      "project:unknown",
    ]);
  });

  it("drops tags in the plugin's own namespaces, so attribution cannot be forged", () => {
    // A configured `harness:x` would sit next to the real one and make the document match a
    // documents-list filter for an agent that never wrote it.
    const { tags } = buildRetainStamp(
      { retainTags: ["harness:not-codex", "source:elsewhere", "env:work"] },
      ctx()
    );
    expect(tags).toEqual(["env:work"]);
  });

  it("drops a tag that resolves to nothing", () => {
    expect(buildRetainStamp({ retainTags: ["   "] }, ctx()).tags).toEqual([]);
  });

  it("resolves {timestamp} to an ISO-8601 instant", () => {
    const { metadata } = buildRetainStamp({ retainMetadata: { at: "{timestamp}" } }, ctx());
    expect(metadata.at).toMatch(/^\d{4}-\d{2}-\d{2}T[\d:.]+Z$/);
  });

  it("gives one retain a single consistent {timestamp} across tags and metadata", () => {
    const { tags, metadata } = buildRetainStamp(
      { retainTags: ["t:{timestamp}"], retainMetadata: { at: "{timestamp}" } },
      ctx()
    );
    expect(tags[0]).toBe(`t:${metadata.at}`);
  });
});

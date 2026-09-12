import { execFileSync, spawnSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { buildSync } from "esbuild";
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";

// Observe the host process's actual stderr. A spy on console.error cannot see
// execFileSync forwarding a failed child's stderr before the caller catches it.
describe("background git stderr isolation", () => {
  let buildDir: string;
  let bundle: string;
  const repos: string[] = [];

  beforeAll(() => {
    buildDir = mkdtempSync(join(tmpdir(), "hs-git-stderr-build-"));
    bundle = pathToFileURL(join(buildDir, "probe.mjs")).href;
    buildSync({
      stdin: {
        contents: `
          export { gitHeadSha, hasGitHistory, commitsSince, gitLogText, gitLogNewestAuthorDate, ingestGit } from './git';
          export { syncStatus } from './status';
          export { syncGit } from './sync';
          export { buildSessionStartContext } from './session-start';
          export { resolveConfig } from './config';
        `,
        resolveDir: fileURLToPath(new URL(".", import.meta.url)),
        loader: "ts",
      },
      bundle: true,
      platform: "node",
      format: "esm",
      outfile: fileURLToPath(bundle),
      banner: {
        js: 'import { createRequire } from "node:module"; const require = createRequire(import.meta.url);',
      },
    });
  });

  afterEach(() => {
    for (const repo of repos.splice(0)) rmSync(repo, { recursive: true, force: true });
  });
  afterAll(() => rmSync(buildDir, { recursive: true, force: true }));

  function directory(): string {
    const repo = mkdtempSync(join(tmpdir(), "hs-git-stderr-repo-"));
    repos.push(repo);
    return repo;
  }

  function run(repo: string, code: string): unknown {
    const result = spawnSync(
      process.execPath,
      [
        "--input-type=module",
        "-e",
        `
      import * as api from ${JSON.stringify(bundle)};
      const repo = ${JSON.stringify(repo)};
      const client = {
        listDocumentIds: async () => new Set(),
        listPages: async () => ({ items: [] }),
        activeOperations: async () => 0,
      };
      ${code}
    `,
      ],
      {
        encoding: "utf8",
        timeout: 10_000,
        windowsHide: true,
        env: {
          ...process.env,
          GIT_CEILING_DIRECTORIES: repo,
          GIT_DIR: undefined,
          GIT_WORK_TREE: undefined,
        },
      }
    );
    expect(result.error).toBeUndefined();
    expect(result.status).toBe(0);
    expect(result.stderr).toBe("");
    return JSON.parse(result.stdout);
  }

  it("quietly returns the existing fallback values outside a repository", () => {
    expect(
      run(
        directory(),
        `console.log(JSON.stringify([
      api.gitHeadSha(repo), api.hasGitHistory(repo), api.commitsSince(repo, 'missing'),
      api.gitLogText(repo, 10), api.gitLogNewestAuthorDate(repo)
    ]));`
      )
    ).toEqual([null, false, null, "", null]);
  });

  it("keeps diagnostics on an uncaught ingestion failure without printing them", () => {
    expect(
      run(
        directory(),
        `
      try { await api.ingestGit(client, repo); throw new Error('expected git to fail'); }
      catch (error) { console.log(JSON.stringify({ status: error.status, stderr: String(error.stderr) })); }
    `
      )
    ).toEqual({ status: 128, stderr: expect.stringContaining("fatal:") });
  });

  it("returns an unknown sync target without leaking the commit-count error", () => {
    expect(
      run(
        directory(),
        `console.log(JSON.stringify((await api.syncStatus(client, 'bank', repo)).gitDiffTarget));`
      )
    ).toBeNull();
  });

  it("preserves best-effort sync when neither the preferred ref nor HEAD exists", () => {
    expect(
      run(
        directory(),
        `console.log(JSON.stringify(await api.syncGit(client, repo, { fetch: true })));`
      )
    ).toEqual({
      ref: null,
      total: 0,
      ingested: 0,
      failures: 0,
      inSync: true,
    });
  });

  it("does not leak a late commit-count failure while building the full-sync banner", () => {
    const repo = directory();
    const git = (...args: string[]) =>
      execFileSync("git", ["-C", repo, ...args], { stdio: "pipe" });
    git("init", "-q");
    git(
      "-c",
      "user.name=Test",
      "-c",
      "user.email=test@example.invalid",
      "commit",
      "--allow-empty",
      "-qm",
      "initial"
    );
    expect(
      run(
        repo,
        `
      import { renameSync } from 'node:fs';
      import { join } from 'node:path';
      client.listDocumentIds = async (tag) => {
        // The repository disappears after HEAD resolves but before rev-list.
        if (tag.startsWith('gitlog-head:')) renameSync(join(repo, '.git'), join(repo, '.git-hidden'));
        return new Set(['git:already-ingested']);
      };
      const result = await api.buildSessionStartContext({
        cwd: repo, bankId: 'bank', client, harness: 'pi',
        cfg: api.resolveConfig({ gitIngest: 'full', codebaseSurvey: false }),
        startSeed: () => {},
      });
      console.log(JSON.stringify(result.systemMessage));
    `
      )
    ).toContain("syncing git history (1/300)");
  });
});

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtemp, rm, writeFile, mkdir } from "fs/promises";
import { tmpdir } from "os";
import { join } from "path";
import { readFileSync, writeFileSync, existsSync, readdirSync, statSync } from "fs";
import { extname, relative, basename } from "path";

// ── Extracted helpers (mirroring cli.ts logic for unit testing) ──

const CONTENT_EXTS = new Set([".md", ".txt", ".html", ".json", ".csv", ".xml"]);
const IGNORED_FILES = new Set(["bank-template.json"]);

function findContentFiles(dir: string): string[] {
  const results: string[] = [];
  function walk(current: string) {
    for (const entry of readdirSync(current)) {
      const full = join(current, entry);
      if (statSync(full).isDirectory()) {
        walk(full);
      } else if (CONTENT_EXTS.has(extname(entry).toLowerCase()) && !IGNORED_FILES.has(entry)) {
        results.push(relative(dir, full));
      }
    }
  }
  walk(dir);
  return results.sort();
}

function isLocalPath(input: string): boolean {
  return (
    input.startsWith("./") ||
    input.startsWith("../") ||
    input.startsWith("/") ||
    input.startsWith("~")
  );
}

function parseAgentsJson(raw: string): any[] {
  const clean = raw.replace(/\n?\x1b\[[0-9;]*m[^\n]*/g, "").trim();
  const arrStart = clean.indexOf("\n[");
  const jsonStr = arrStart >= 0 ? clean.slice(arrStart + 1) : clean.startsWith("[") ? clean : "[]";
  return JSON.parse(jsonStr);
}

function resolveFromPluginConfig(
  agentId: string,
  pc: Record<string, any>
): { apiUrl: string; bankId: string; apiToken?: string } {
  const apiUrl = pc.hindsightApiUrl || `http://localhost:${pc.apiPort || 9077}`;
  const apiToken = pc.hindsightApiToken || undefined;

  let bankId: string;
  if (pc.dynamicBankId === false && pc.bankId) {
    bankId = pc.bankId;
  } else {
    const granularity: string[] = pc.dynamicBankGranularity || ["agent", "channel", "user"];
    const fieldMap: Record<string, string> = {
      agent: agentId,
      channel: "unknown",
      user: "anonymous",
      provider: "unknown",
    };
    const base = granularity.map((f) => encodeURIComponent(fieldMap[f] || "unknown")).join("::");
    bankId = pc.bankIdPrefix ? `${pc.bankIdPrefix}-${base}` : base;
  }

  return { apiUrl, bankId, apiToken };
}

// ── Tests ──

describe("findContentFiles", () => {
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), "sda-test-"));
  });

  afterEach(async () => {
    await rm(tmpDir, { recursive: true, force: true });
  });

  it("finds .md files recursively", async () => {
    await writeFile(join(tmpDir, "root.md"), "hello");
    await mkdir(join(tmpDir, "sub"));
    await writeFile(join(tmpDir, "sub", "nested.md"), "world");

    const files = findContentFiles(tmpDir);
    expect(files).toEqual(["root.md", "sub/nested.md"]);
  });

  it("finds multiple content extensions", async () => {
    await writeFile(join(tmpDir, "a.md"), "md");
    await writeFile(join(tmpDir, "b.txt"), "txt");
    await writeFile(join(tmpDir, "c.html"), "html");
    await writeFile(join(tmpDir, "d.csv"), "csv");

    const files = findContentFiles(tmpDir);
    expect(files).toEqual(["a.md", "b.txt", "c.html", "d.csv"]);
  });

  it("excludes bank-template.json", async () => {
    await writeFile(join(tmpDir, "bank-template.json"), "{}");
    await writeFile(join(tmpDir, "readme.md"), "hello");

    const files = findContentFiles(tmpDir);
    expect(files).toEqual(["readme.md"]);
  });

  it("excludes non-content files", async () => {
    await writeFile(join(tmpDir, "image.png"), "binary");
    await writeFile(join(tmpDir, "script.js"), "code");
    await writeFile(join(tmpDir, "doc.md"), "content");

    const files = findContentFiles(tmpDir);
    expect(files).toEqual(["doc.md"]);
  });

  it("handles deeply nested directories", async () => {
    await mkdir(join(tmpDir, "a", "b", "c"), { recursive: true });
    await writeFile(join(tmpDir, "a", "b", "c", "deep.md"), "deep");

    const files = findContentFiles(tmpDir);
    expect(files).toEqual(["a/b/c/deep.md"]);
  });

  it("returns empty for directory with no content", async () => {
    await writeFile(join(tmpDir, "bank-template.json"), "{}");
    await writeFile(join(tmpDir, "image.png"), "binary");

    const files = findContentFiles(tmpDir);
    expect(files).toEqual([]);
  });

  it("ignores bank-template.json in subdirectories too", async () => {
    await mkdir(join(tmpDir, "sub"));
    await writeFile(join(tmpDir, "sub", "bank-template.json"), "{}");
    await writeFile(join(tmpDir, "sub", "guide.md"), "content");

    const files = findContentFiles(tmpDir);
    expect(files).toEqual(["sub/guide.md"]);
  });
});

describe("isLocalPath", () => {
  it("detects relative paths", () => {
    expect(isLocalPath("./my-agent")).toBe(true);
    expect(isLocalPath("../parent/agent")).toBe(true);
  });

  it("detects absolute paths", () => {
    expect(isLocalPath("/Users/me/agent")).toBe(true);
  });

  it("detects home paths", () => {
    expect(isLocalPath("~/dev/agent")).toBe(true);
  });

  it("rejects GitHub-style references", () => {
    expect(isLocalPath("marketing")).toBe(false);
    expect(isLocalPath("org/repo/path")).toBe(false);
  });
});

describe("parseAgentsJson", () => {
  it("parses clean JSON array", () => {
    const agents = parseAgentsJson('[{"id": "main"}]');
    expect(agents).toEqual([{ id: "main" }]);
  });

  it("strips ANSI log lines before JSON", () => {
    const raw =
      "\x1b[38;5;103mhindsight:\x1b[0m plugin entry invoked\n" +
      '[\n  {"id": "main"},\n  {"id": "seo-writer", "name": "seo-writer"}\n]';
    const agents = parseAgentsJson(raw);
    expect(agents).toHaveLength(2);
    expect(agents[1].id).toBe("seo-writer");
  });

  it("returns empty array for unparseable output", () => {
    const agents = parseAgentsJson("some random text");
    expect(agents).toEqual([]);
  });

  it("handles multiple ANSI lines", () => {
    const raw = [
      "Config warnings:",
      "\x1b[35m[plugins]\x1b[39m registering plugin",
      "\x1b[35m[plugins]\x1b[39m hooks registered",
      '[{"id": "test"}]',
    ].join("\n");
    const agents = parseAgentsJson(raw);
    expect(agents).toEqual([{ id: "test" }]);
  });
});

describe("resolveFromPluginConfig", () => {
  it("uses external API URL when set", () => {
    const result = resolveFromPluginConfig("my-agent", {
      hindsightApiUrl: "https://api.example.com",
      hindsightApiToken: "tok-123",
      dynamicBankGranularity: ["agent"],
    });
    expect(result.apiUrl).toBe("https://api.example.com");
    expect(result.apiToken).toBe("tok-123");
    expect(result.bankId).toBe("my-agent");
  });

  it("falls back to localhost with apiPort", () => {
    const result = resolveFromPluginConfig("my-agent", {
      apiPort: 8888,
      dynamicBankGranularity: ["agent"],
    });
    expect(result.apiUrl).toBe("http://localhost:8888");
    expect(result.apiToken).toBeUndefined();
  });

  it("defaults to port 9077", () => {
    const result = resolveFromPluginConfig("my-agent", {});
    expect(result.apiUrl).toBe("http://localhost:9077");
  });

  it("computes bank ID with prefix", () => {
    const result = resolveFromPluginConfig("seo-writer", {
      bankIdPrefix: "nicolo",
      dynamicBankGranularity: ["agent"],
    });
    expect(result.bankId).toBe("nicolo-seo-writer");
  });

  it("computes bank ID without prefix", () => {
    const result = resolveFromPluginConfig("seo-writer", {
      dynamicBankGranularity: ["agent"],
    });
    expect(result.bankId).toBe("seo-writer");
  });

  it("uses multi-field granularity", () => {
    const result = resolveFromPluginConfig("my-agent", {
      dynamicBankGranularity: ["agent", "channel", "user"],
    });
    expect(result.bankId).toBe("my-agent::unknown::anonymous");
  });

  it("uses default granularity when not specified", () => {
    const result = resolveFromPluginConfig("my-agent", {});
    expect(result.bankId).toBe("my-agent::unknown::anonymous");
  });

  it("uses static bankId when dynamicBankId is false", () => {
    const result = resolveFromPluginConfig("my-agent", {
      dynamicBankId: false,
      bankId: "static-bank",
    });
    expect(result.bankId).toBe("static-bank");
  });
});

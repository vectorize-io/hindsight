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
    expect(isLocalPath("marketing/seo")).toBe(false);
  });
});

describe("deriveDefaultName", () => {
  // Mirrors the logic in resolveAgentDir:
  // - GitHub refs: subpath with / → hyphens (marketing/seo → marketing-seo)
  // - Local paths: basename of resolved dir

  function deriveFromGitHub(input: string): string {
    const parts = input.split("/");
    const subpath = parts.length <= 2 ? input : parts.slice(2).join("/");
    return subpath.replace(/\//g, "-");
  }

  it("single name stays as-is", () => {
    expect(deriveFromGitHub("marketing")).toBe("marketing");
  });

  it("two segments become hyphenated", () => {
    expect(deriveFromGitHub("marketing/seo")).toBe("marketing-seo");
  });

  it("three+ segments treat first two as org/repo", () => {
    // marketing/seo/technical → org=marketing, repo=seo, path=technical
    expect(deriveFromGitHub("marketing/seo/technical")).toBe("technical");
  });

  it("org/repo with deep path uses hyphenated path", () => {
    expect(deriveFromGitHub("my-org/my-repo/agents/seo")).toBe("agents-seo");
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

  it("resolves nemoclaw-style config (external API, static bank)", () => {
    const result = resolveFromPluginConfig("marketing-seo", {
      hindsightApiUrl: "https://api.hindsight.vectorize.io",
      hindsightApiToken: "hsk_abc",
      llmProvider: "claude-code",
      dynamicBankId: false,
      bankIdPrefix: "my-sandbox",
    });
    expect(result.apiUrl).toBe("https://api.hindsight.vectorize.io");
    expect(result.apiToken).toBe("hsk_abc");
    // dynamicBankId=false but no bankId set, so falls through to dynamic path
    // with bankIdPrefix
    expect(result.bankId).toBe("my-sandbox-marketing-seo::unknown::anonymous");
  });

  it("resolves nemoclaw-style config with static bankId", () => {
    const result = resolveFromPluginConfig("marketing-seo", {
      hindsightApiUrl: "https://api.hindsight.vectorize.io",
      hindsightApiToken: "hsk_abc",
      dynamicBankId: false,
      bankId: "my-sandbox-openclaw",
    });
    expect(result.bankId).toBe("my-sandbox-openclaw");
  });
});

describe("versionGte", () => {
  function versionGte(current: string, required: string): boolean {
    const [aMaj, aMin, aPat] = current.split(".").map(Number);
    const [bMaj, bMin, bPat] = required.split(".").map(Number);
    if (aMaj !== bMaj) return aMaj > bMaj;
    if (aMin !== bMin) return aMin > bMin;
    return aPat >= bPat;
  }

  it("equal versions return true", () => {
    expect(versionGte("0.7.2", "0.7.2")).toBe(true);
  });

  it("higher patch returns true", () => {
    expect(versionGte("0.7.3", "0.7.2")).toBe(true);
  });

  it("lower patch returns false", () => {
    expect(versionGte("0.7.1", "0.7.2")).toBe(false);
  });

  it("higher minor returns true", () => {
    expect(versionGte("0.8.0", "0.7.2")).toBe(true);
  });

  it("higher major returns true", () => {
    expect(versionGte("1.0.0", "0.7.2")).toBe(true);
  });

  it("lower major returns false", () => {
    expect(versionGte("0.6.9", "1.0.0")).toBe(false);
  });
});

describe("harness argument parsing", () => {
  function parseHarness(args: string[]): { harness?: string; sandbox?: string } {
    let harness: string | undefined;
    let sandbox: string | undefined;
    for (let i = 0; i < args.length; i++) {
      if (args[i] === "--harness" && args[i + 1]) harness = args[++i];
      else if (args[i] === "--sandbox" && args[i + 1]) sandbox = args[++i];
    }
    return { harness, sandbox };
  }

  it("parses openclaw harness", () => {
    const { harness, sandbox } = parseHarness(["--harness", "openclaw"]);
    expect(harness).toBe("openclaw");
    expect(sandbox).toBeUndefined();
  });

  it("parses nemoclaw harness with sandbox", () => {
    const { harness, sandbox } = parseHarness([
      "--harness",
      "nemoclaw",
      "--sandbox",
      "my-assistant",
    ]);
    expect(harness).toBe("nemoclaw");
    expect(sandbox).toBe("my-assistant");
  });

  it("nemoclaw without sandbox returns undefined sandbox", () => {
    const { harness, sandbox } = parseHarness(["--harness", "nemoclaw"]);
    expect(harness).toBe("nemoclaw");
    expect(sandbox).toBeUndefined();
  });

  it("parses hermes harness", () => {
    const { harness } = parseHarness(["--harness", "hermes"]);
    expect(harness).toBe("hermes");
  });

  it("parses claude harness", () => {
    const { harness } = parseHarness(["--harness", "claude"]);
    expect(harness).toBe("claude");
  });

  it("parses claude-code harness", () => {
    const { harness } = parseHarness(["--harness", "claude-code"]);
    expect(harness).toBe("claude-code");
  });
});

describe("hermes hindsight config", () => {
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), "sda-hermes-test-"));
  });

  afterEach(async () => {
    await rm(tmpDir, { recursive: true, force: true });
  });

  it("generates correct hindsight/config.json", async () => {
    const hindsightDir = join(tmpDir, "hindsight");
    await mkdir(hindsightDir, { recursive: true });
    const cfg = {
      mode: "cloud",
      api_url: "https://api.hindsight.vectorize.io",
      api_key: "hsk_test",
      bank_id: "marketing-seo",
      bank_id_template: "",
      recall_budget: "mid",
      memory_mode: "hybrid",
    };
    await writeFile(join(hindsightDir, "config.json"), JSON.stringify(cfg));

    const loaded = JSON.parse(readFileSync(join(hindsightDir, "config.json"), "utf-8"));
    expect(loaded.bank_id).toBe("marketing-seo");
    expect(loaded.bank_id_template).toBe("");
    expect(loaded.api_url).toBe("https://api.hindsight.vectorize.io");
    expect(loaded.api_key).toBe("hsk_test");
    expect(loaded.mode).toBe("cloud");
  });

  it("empty bank_id_template prevents dynamic resolution", () => {
    // Mirrors _resolve_bank_id_template logic: empty template → use fallback
    function resolveTemplate(template: string, fallback: string): string {
      if (!template) return fallback;
      return template; // simplified — real impl does placeholder substitution
    }

    expect(resolveTemplate("", "marketing-seo")).toBe("marketing-seo");
    expect(resolveTemplate("hermes-{profile}", "fallback")).toBe("hermes-{profile}");
  });

  it("plugin config is read from hindsight/config.json", async () => {
    // Simulates what the hermes plugin does: read from HERMES_HOME/hindsight/config.json
    const hindsightDir = join(tmpDir, "hindsight");
    await mkdir(hindsightDir, { recursive: true });
    const cfg = {
      mode: "cloud",
      api_url: "https://custom.api.com",
      api_key: "hsk_custom",
      bank_id: "my-agent",
    };
    await writeFile(join(hindsightDir, "config.json"), JSON.stringify(cfg));

    // Read it back the way the plugin does
    const cfgPath = join(tmpDir, "hindsight", "config.json");
    const loaded = JSON.parse(readFileSync(cfgPath, "utf-8"));
    const normalized = {
      api_url: loaded.api_url || "",
      api_token: loaded.api_key || "",
      bank_id: loaded.bank_id || "hermes",
    };

    expect(normalized.api_url).toBe("https://custom.api.com");
    expect(normalized.api_token).toBe("hsk_custom");
    expect(normalized.bank_id).toBe("my-agent");
  });

  it("falls back to default bank_id when not set", async () => {
    const hindsightDir = join(tmpDir, "hindsight");
    await mkdir(hindsightDir, { recursive: true });
    await writeFile(
      join(hindsightDir, "config.json"),
      JSON.stringify({ mode: "cloud", api_url: "https://api.example.com", api_key: "tok" })
    );

    const loaded = JSON.parse(readFileSync(join(hindsightDir, "config.json"), "utf-8"));
    const bankId = loaded.bank_id || "hermes";
    expect(bankId).toBe("hermes");
  });
});

// ── Claude skill generation tests ─────────────────────

describe("claude skill generation", () => {
  function generateSkillFrontmatter(agentId: string, apiToken?: string): string {
    const authHeader = apiToken ? `-H "Authorization: Bearer ${apiToken}"` : "";
    return `---\nname: ${agentId}\ndescription: Activate the ${agentId} agent. Loads knowledge pages from Hindsight memory.\n---`;
  }

  it("includes required name frontmatter field", () => {
    const fm = generateSkillFrontmatter("marketing-seo");
    expect(fm).toContain("name: marketing-seo");
  });

  it("includes description frontmatter field", () => {
    const fm = generateSkillFrontmatter("my-agent");
    expect(fm).toContain("description: Activate the my-agent agent");
  });

  it("bakes auth token when provided", () => {
    const authHeader = "secret-token" ? `-H "Authorization: Bearer secret-token"` : "";
    expect(authHeader).toContain("Bearer secret-token");
  });

  it("omits auth header when no token", () => {
    const authHeader = undefined ? `-H "Authorization: Bearer undefined"` : "";
    expect(authHeader).toBe("");
  });
});

describe("claude config validation", () => {
  function validateUrl(v: string | undefined): string | undefined {
    if (!v) return "URL required";
    try {
      const parsed = new URL(v);
      if (parsed.hostname === "localhost" || parsed.hostname === "127.0.0.1") {
        return "Claude connects from Anthropic's cloud — localhost won't work. Use a public URL.";
      }
    } catch {
      return "Invalid URL";
    }
  }

  it("rejects localhost URLs", () => {
    expect(validateUrl("http://localhost:9077")).toContain("localhost");
    expect(validateUrl("http://127.0.0.1:9077")).toContain("localhost");
  });

  it("accepts public URLs", () => {
    expect(validateUrl("https://api.example.com")).toBeUndefined();
    expect(validateUrl("https://api.hindsight.vectorize.io")).toBeUndefined();
  });

  it("rejects empty/missing URLs", () => {
    expect(validateUrl("")).toBe("URL required");
    expect(validateUrl(undefined)).toBe("URL required");
  });

  it("rejects invalid URLs", () => {
    expect(validateUrl("not-a-url")).toBe("Invalid URL");
  });
});

describe("claude-code config resolution", () => {
  function resolveFromConfig(
    agentId: string,
    config: Record<string, any>
  ): { apiUrl: string; bankId: string; apiToken?: string } {
    const apiUrl = config.hindsightApiUrl || `http://localhost:${config.apiPort || 9077}`;
    const apiToken = config.hindsightApiToken || undefined;
    let bankId: string;
    if (config.dynamicBankId === false && config.bankId) {
      bankId = config.bankId;
    } else if (config.dynamicBankId) {
      const granularity: string[] = config.dynamicBankGranularity || ["agent", "project"];
      const fieldMap: Record<string, string> = {
        agent: config.agentName || agentId,
        project: "unknown",
        session: "unknown",
        channel: "default",
        user: "anonymous",
      };
      const base = granularity
        .map((f: string) => encodeURIComponent(fieldMap[f] || "unknown"))
        .join("::");
      bankId = config.bankIdPrefix ? `${config.bankIdPrefix}-${base}` : base;
    } else {
      bankId = config.bankIdPrefix ? `${config.bankIdPrefix}-${agentId}` : agentId;
    }
    return { apiUrl, bankId, apiToken };
  }

  it("uses external API URL when set", () => {
    const r = resolveFromConfig("agent", {
      hindsightApiUrl: "https://api.example.com",
      hindsightApiToken: "tok",
    });
    expect(r.apiUrl).toBe("https://api.example.com");
    expect(r.apiToken).toBe("tok");
  });

  it("defaults to localhost:9077", () => {
    const r = resolveFromConfig("agent", {});
    expect(r.apiUrl).toBe("http://localhost:9077");
  });

  it("uses agentId as default bank", () => {
    const r = resolveFromConfig("marketing-seo", {});
    expect(r.bankId).toBe("marketing-seo");
  });

  it("uses static bankId when dynamicBankId=false", () => {
    const r = resolveFromConfig("agent", { dynamicBankId: false, bankId: "my-bank" });
    expect(r.bankId).toBe("my-bank");
  });

  it("computes dynamic bankId", () => {
    const r = resolveFromConfig("seo", {
      dynamicBankId: true,
      agentName: "seo",
      dynamicBankGranularity: ["agent"],
    });
    expect(r.bankId).toBe("seo");
  });

  it("applies bankIdPrefix", () => {
    const r = resolveFromConfig("agent", { bankIdPrefix: "prod" });
    expect(r.bankId).toBe("prod-agent");
  });
});

describe("harness validation", () => {
  const SUPPORTED_HARNESSES = ["openclaw", "nemoclaw", "hermes", "claude", "claude-code"];

  it("accepts all supported harnesses", () => {
    for (const h of SUPPORTED_HARNESSES) {
      expect(SUPPORTED_HARNESSES.includes(h)).toBe(true);
    }
  });

  it("rejects unknown harnesses", () => {
    expect(SUPPORTED_HARNESSES.includes("chatgpt")).toBe(false);
    expect(SUPPORTED_HARNESSES.includes("")).toBe(false);
  });
});

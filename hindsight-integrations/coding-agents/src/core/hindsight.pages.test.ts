import { afterEach, describe, expect, it, vi } from "vitest";
import { HindsightClient } from "./hindsight";
import {
  buildPageTrigger,
  KNOWLEDGE_LABELS,
  PAGE_MAX_TOKENS,
  pagesFor,
  RETAIN_STRATEGIES,
} from "./missions";
import { resolveConfig } from "./config";

/** What a client built with `bank: "repo-a"` and no `project` seeds — the bank id is the fallback. */
const PAGES = pagesFor("repo-a");

afterEach(() => vi.restoreAllMocks());

function stubFetch(calls: any[], jsonImpl: () => Promise<unknown> = async () => ({ ok: true })) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init: any) => {
      calls.push({
        url,
        method: init?.method,
        body: init?.body ? JSON.parse(init.body) : undefined,
      });
      return { ok: true, status: 200, json: jsonImpl } as any;
    })
  );
}

/** The knowledge-base surface is the ONLY page surface: nothing here may touch /mental-models,
 *  or ids stop resolving (search returns kp-… node ids) and seeded pages fall out of the search
 *  corpus (it joins through knowledge_pages). */
describe("HindsightClient knowledge-page reads", () => {
  it("recognizes an older server and exposes the missing capability", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: false,
        status: 404,
        json: async () => ({ detail: "not found" }),
      })) as any
    );
    const c = new HindsightClient({ apiUrl: "http://x", bank: "repo-a" });
    await expect(c.listPages()).rejects.toMatchObject({ code: "knowledge_pages_unavailable" });
    expect(c.knowledgePagesSupported).toBe(false);
    await expect(c.searchKnowledgePages("architecture")).rejects.toMatchObject({
      code: "knowledge_pages_unavailable",
    });
  });

  it("listPages reads the knowledge-base tree — never /mental-models", async () => {
    const calls: any[] = [];
    stubFetch(calls, async () => ({ roots: [] }));
    const c = new HindsightClient({ apiUrl: "http://x", bank: "repo-a" });
    await c.listPages();
    expect(calls).toHaveLength(1);
    expect(calls[0].method).toBe("GET");
    expect(calls[0].url).toContain("/v1/default/banks/repo-a/knowledge-base/tree");
    expect(calls.some((k) => k.url.includes("/mental-models"))).toBe(false);
  });

  it("listPages flattens nested pages, drops folders, and keeps the containing folder name", async () => {
    const calls: any[] = [];
    stubFetch(calls, async () => ({
      roots: [
        { id: "kp-1", kind: "page", name: "Component map", description: "what are the parts?" },
        {
          id: "kf-1",
          kind: "folder",
          name: "Initiatives",
          children: [{ id: "kp-2", kind: "page", name: "Retry backoff" }],
        },
      ],
    }));
    const c = new HindsightClient({ apiUrl: "http://x", bank: "repo-a" });
    expect(await c.listPages()).toEqual({
      items: [
        { id: "kp-1", name: "Component map", description: "what are the parts?" },
        { id: "kp-2", name: "Retry backoff", folder: "Initiatives" },
      ],
    });
  });

  it("listPages degrades to an empty roster when the tree body isn't JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        status: 200,
        json: async () => {
          throw new Error("Unexpected end of JSON input");
        },
      })) as any
    );
    const c = new HindsightClient({ apiUrl: "http://x", bank: "repo-a" });
    expect(await c.listPages()).toEqual({ items: [] });
  });

  it("getPage GETs knowledge-base/pages/{id}", async () => {
    const calls: any[] = [];
    stubFetch(calls, async () => ({ id: "kp-1" }));
    const c = new HindsightClient({ apiUrl: "http://x", bank: "repo-a" });
    const result = await c.getPage("kp-1");
    expect(result).toEqual({ id: "kp-1" });
    expect(calls[0].method).toBe("GET");
    expect(calls[0].url).toContain("/knowledge-base/pages/kp-1");
  });

  it("getPage resolves the id shape searchKnowledgePages hands back", async () => {
    const calls: any[] = [];
    stubFetchRouted(calls, [
      {
        match: (m, u) => m === "GET" && u.includes("/knowledge-base/search"),
        json: { results: [{ id: "kp-abc", name: "Component map", snippet: "…", score: 0.5 }] },
      },
      {
        match: (m, u) => m === "GET" && u.includes("/knowledge-base/pages/"),
        json: { id: "kp-abc" },
      },
    ]);
    const c = new HindsightClient({ apiUrl: "http://x", bank: "repo-a" });
    const [hit] = await c.searchKnowledgePages("components");
    await c.getPage(hit.id);
    // The read must land on the page endpoint for the very id search returned — the old
    // /mental-models read 404'd on every kp-… id search produced.
    expect(calls[1].url).toContain("/knowledge-base/pages/kp-abc");
  });

  it("URL-encodes pageId in the getPage suffix", async () => {
    const calls: any[] = [];
    stubFetch(calls, async () => ({ ok: true }));
    const c = new HindsightClient({ apiUrl: "http://x", bank: "repo-a" });
    await c.getPage("p 1/x");
    expect(calls[0].url).toContain(`/knowledge-base/pages/${encodeURIComponent("p 1/x")}`);
  });

  it("getPage throws on 404 instead of returning the error envelope", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () => ({ ok: false, status: 404, json: async () => ({ detail: "not found" }) }) as any
      )
    );
    const c = new HindsightClient({ apiUrl: "http://x", bank: "repo-a" });
    await expect(c.getPage("missing")).rejects.toThrow("knowledge page not found: missing");
  });
});

describe("HindsightClient.seedPages", () => {
  it("skips page writes when the server has no knowledge-pages API", async () => {
    const calls: any[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init: any) => {
        calls.push({ url, method: init?.method });
        if (url.endsWith("/knowledge-base/tree"))
          return { ok: false, status: 404, json: async () => ({}) } as any;
        return { ok: true, status: 200, json: async () => ({}) } as any;
      }) as any
    );
    const c = new HindsightClient({ apiUrl: "http://x", bank: "repo-a" });
    await expect(c.configureBank()).resolves.toBeUndefined();
    expect(c.knowledgePagesSupported).toBe(false);
    expect(calls.some((x) => x.url.endsWith("/import"))).toBe(true);
    expect(calls.some((x) => x.url.endsWith("/knowledge-base/pages"))).toBe(false);
  });

  it("creates every seeded page through /knowledge-base/pages on an empty bank", async () => {
    const calls: any[] = [];
    stubFetchRouted(calls, [
      { match: (m, u) => m === "GET" && u.endsWith("/knowledge-base/tree"), json: { roots: [] } },
    ]);
    const c = new HindsightClient({ apiUrl: "http://x", bank: "repo-a" });
    await c.seedPages();

    const posts = calls.filter(
      (k) => k.method === "POST" && k.url.endsWith("/knowledge-base/pages")
    );
    expect(posts).toHaveLength(PAGES.length);
    expect(posts.map((p) => p.body.name).sort()).toEqual(PAGES.map((p) => p.name).sort());
    for (const post of posts) {
      expect(post.body.tags).toHaveLength(1);
      expect(post.body.tags[0]).toMatch(
        /^knowledge:(feature-work|decision|convention|component|concept)$/
      );
      expect(post.body.max_tokens).toBe(PAGE_MAX_TOKENS);
      expect(post.body.trigger.refresh_after_consolidation).toBe(true);
      // NOT the server's `all_strict` default for a tagged model: that excludes untagged
      // memories, and every observation this plugin's banks consolidate is untagged.
      expect(post.body.trigger.tags_match).toBe("all");
      expect(post.body.trigger.fact_types).toContain("observation");
      expect(post.body.parent_id).toBeUndefined(); // seeded at the tree root
    }
    // Nothing on the mental-models surface.
    expect(calls.some((k) => k.url.includes("/mental-models"))).toBe(false);
  });

  // The trigger decides what these pages cost to keep current (#3506); it used to be hardcoded.
  it("stamps the configured refresh policy on every page it seeds", async () => {
    const calls: any[] = [];
    stubFetchRouted(calls, [
      { match: (m, u) => m === "GET" && u.endsWith("/knowledge-base/tree"), json: { roots: [] } },
    ]);
    const c = new HindsightClient({ apiUrl: "http://x", bank: "repo-a" });
    await c.configureBank({
      pageTrigger: buildPageTrigger(
        resolveConfig({ pageTriggerType: "cron", pageTriggerCron: "0 3 * * *" })
      ),
    });

    const posts = calls.filter(
      (k) => k.method === "POST" && k.url.endsWith("/knowledge-base/pages")
    );
    expect(posts).toHaveLength(PAGES.length);
    for (const post of posts) {
      expect(post.body.trigger.refresh_cron).toBe("0 3 * * *");
      expect(post.body.trigger.refresh_after_consolidation).toBeUndefined();
    }
  });

  it("is idempotent: an already-seeded bank issues no writes at all", async () => {
    const calls: any[] = [];
    stubFetchRouted(calls, [
      {
        match: (m, u) => m === "GET" && u.endsWith("/knowledge-base/tree"),
        json: {
          roots: PAGES.map((p, i) => ({
            id: `kp-${i}`,
            kind: "page",
            name: p.name,
            description: p.source_query,
            trigger: { tags_match: buildPageTrigger().tags_match },
          })),
        },
      },
    ]);
    const c = new HindsightClient({ apiUrl: "http://x", bank: "repo-a" });
    await c.seedPages();
    expect(calls).toHaveLength(1); // the tree GET only
    expect(calls.every((k) => k.method === "GET")).toBe(true);
  });

  it("tolerates a 409 from a concurrent run that seeded the same name first", async () => {
    const calls: any[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init: any) => {
        calls.push({ url, method: init?.method });
        const conflict = init?.method === "POST" && url.endsWith("/knowledge-base/pages");
        return {
          ok: !conflict,
          status: conflict ? 409 : 200,
          json: async () => ({ roots: [] }),
          text: async () => "already exists",
        } as any;
      })
    );
    const c = new HindsightClient({ apiUrl: "http://x", bank: "repo-a" });
    // A losing race must not fail the deepen run — the page exists either way.
    await expect(c.seedPages()).resolves.toBeUndefined();
    expect(
      calls.filter((k) => k.method === "POST" && k.url.endsWith("/knowledge-base/pages"))
    ).toHaveLength(PAGES.length);
  });

  it("matches by name case-insensitively and PATCHes a page whose source query drifted", async () => {
    const calls: any[] = [];
    const drifted = PAGES[0];
    stubFetchRouted(calls, [
      {
        match: (m, u) => m === "GET" && u.endsWith("/knowledge-base/tree"),
        json: {
          roots: PAGES.map((p, i) => ({
            id: `kp-${i}`,
            kind: "page",
            name: p.name.toUpperCase(),
            description: p === drifted ? "an older wording of the query" : p.source_query,
            trigger: { tags_match: buildPageTrigger().tags_match },
          })),
        },
      },
    ]);
    const c = new HindsightClient({ apiUrl: "http://x", bank: "repo-a" });
    await c.seedPages();

    // Case difference alone must NOT look like a missing page.
    expect(calls.some((k) => k.method === "POST")).toBe(false);
    const patches = calls.filter((k) => k.method === "PATCH");
    expect(patches).toHaveLength(1);
    expect(patches[0].url).toContain("/knowledge-base/nodes/kp-0");
    expect(patches[0].body).toEqual({
      source_query: drifted.source_query,
      tags: drifted.tags,
    });
  });

  // A page seeded before `tags_match` existed keeps the server's `all_strict` default, which
  // excludes the untagged shared observations these pages are meant to synthesize from. The
  // source query is unchanged on such a bank, so the trigger has to be its own drift signal.
  it("re-syncs the refresh trigger onto a page whose query did NOT drift", async () => {
    const calls: any[] = [];
    stubFetchRouted(calls, [
      {
        match: (m, u) => m === "GET" && u.endsWith("/knowledge-base/tree"),
        json: {
          roots: PAGES.map((p, i) => ({
            id: `kp-${i}`,
            kind: "page",
            name: p.name,
            description: p.source_query,
            trigger: { tags_match: "all_strict" },
          })),
        },
      },
    ]);
    const c = new HindsightClient({ apiUrl: "http://x", bank: "repo-a" });
    await c.seedPages();

    const patches = calls.filter((k) => k.method === "PATCH");
    expect(patches).toHaveLength(PAGES.length);
    for (const patch of patches) {
      // ONLY the trigger: sending `source_query` would schedule a full rebuild of every page on
      // a bank whose question never changed.
      expect(patch.body).toEqual({ trigger: buildPageTrigger() });
    }
  });

  // A server older than #3572 does not report a page's trigger. That makes the policy unknowable,
  // not divergent; sending a trigger-only PATCH would be rejected by servers older than #3549.
  it("does not write when a server does not report a trigger", async () => {
    const calls: any[] = [];
    stubFetchRouted(calls, [
      {
        match: (m, u) => m === "GET" && u.endsWith("/knowledge-base/tree"),
        json: {
          roots: PAGES.map((p, i) => ({
            id: `kp-${i}`,
            kind: "page",
            name: p.name,
            description: p.source_query,
          })),
        },
      },
    ]);
    const c = new HindsightClient({ apiUrl: "http://x", bank: "repo-a" });
    await c.seedPages();

    expect(calls).toHaveLength(1); // the tree GET only
    expect(calls.every((k) => k.method === "GET")).toBe(true);
  });

  it("still reconciles source-query drift when the server hides trigger", async () => {
    const calls: any[] = [];
    const drifted = PAGES[0];
    stubFetchRouted(calls, [
      {
        match: (m, u) => m === "GET" && u.endsWith("/knowledge-base/tree"),
        json: {
          roots: PAGES.map((p, i) => ({
            id: `kp-${i}`,
            kind: "page",
            name: p.name,
            description: p === drifted ? "an older wording of the query" : p.source_query,
          })),
        },
      },
    ]);
    const c = new HindsightClient({ apiUrl: "http://x", bank: "repo-a" });
    await c.seedPages();

    const patches = calls.filter((k) => k.method === "PATCH");
    expect(patches).toHaveLength(1);
    expect(patches[0].body).toEqual({
      source_query: drifted.source_query,
      tags: drifted.tags,
    });
  });

  it("names the repository in every seeded query, so synthesis can exclude a dependency's facts", async () => {
    const calls: any[] = [];
    stubFetchRouted(calls, [
      { match: (m, u) => m === "GET" && u.endsWith("/knowledge-base/tree"), json: { roots: [] } },
    ]);
    const c = new HindsightClient({
      apiUrl: "http://x",
      bank: "coding-agent::dotfiles",
      project: "dotfiles",
    });
    await c.seedPages();

    const posts = calls.filter(
      (k) => k.method === "POST" && k.url.endsWith("/knowledge-base/pages")
    );
    expect(posts).toHaveLength(PAGES.length);
    for (const post of posts) {
      // The repo is NAMED (not "this project"), and the exclusion is stated — the bank holds
      // facts about dependencies the repo merely discusses, and they are not its own (#3476).
      expect(post.body.source_query).toContain("dotfiles");
      expect(post.body.source_query).toMatch(/external tools, libraries and services/);
      expect(post.body.source_query).toMatch(/dependency/);
    }
  });

  it("falls back to the bank id when no project is supplied, never an unscoped query", async () => {
    const calls: any[] = [];
    stubFetchRouted(calls, [
      { match: (m, u) => m === "GET" && u.endsWith("/knowledge-base/tree"), json: { roots: [] } },
    ]);
    const c = new HindsightClient({ apiUrl: "http://x", bank: "coding-agent::dotfiles" });
    await c.seedPages();

    for (const post of calls.filter((k) => k.method === "POST")) {
      expect(post.body.source_query).toContain("coding-agent::dotfiles");
    }
  });
});

describe("pagesFor", () => {
  it("scopes every page in the taxonomy to the named repository", () => {
    const pages = pagesFor("dotfiles");
    expect(pages).toHaveLength(5);
    for (const page of pages) {
      expect(page.source_query).toContain("Scope this page to dotfiles ITSELF");
    }
  });

  it("is a pure function of the project, so a re-seed does not PATCH the same query back", () => {
    // seedPages() compares the live description against this text on every deepen run; anything
    // varying per call (a timestamp, a set iteration order) would re-PATCH all five pages forever.
    expect(pagesFor("dotfiles")).toEqual(pagesFor("dotfiles"));
    expect(pagesFor("dotfiles")).not.toEqual(pagesFor("other-repo"));
  });

  it("keeps the taxonomy's names and tier tags untouched", () => {
    expect(pagesFor("dotfiles").map((p) => p.name)).toEqual([
      "Component map",
      "Core concepts",
      "Conventions and patterns",
      "Key decisions and rationale",
      "Initiatives and enhancements",
    ]);
    expect(pagesFor("dotfiles").flatMap((p) => p.tags)).toEqual([
      "knowledge:component",
      "knowledge:concept",
      "knowledge:convention",
      "knowledge:decision",
      "knowledge:feature-work",
    ]);
  });
});

/** Route JSON responses by (method, url-substring) so multi-call flows can return distinct bodies. */
function stubFetchRouted(
  calls: any[],
  routes: { match: (method: string, url: string) => boolean; json: unknown }[]
) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init: any) => {
      const method = init?.method;
      calls.push({ url, method, body: init?.body ? JSON.parse(init.body) : undefined });
      const route = routes.find((r) => r.match(method, url));
      return { ok: true, status: 200, json: async () => route?.json ?? { ok: true } } as any;
    })
  );
}

describe("HindsightClient.ensureFolder", () => {
  it("returns an existing root folder's id (case-insensitive) and does NOT POST a duplicate", async () => {
    const calls: any[] = [];
    stubFetchRouted(calls, [
      {
        match: (m, u) => m === "GET" && u.endsWith("/knowledge-base/tree"),
        json: { roots: [{ id: "existing-1", kind: "folder", name: "initiatives" }] },
      },
    ]);
    const c = new HindsightClient({ apiUrl: "http://x", bank: "repo-a" });
    const id = await c.ensureFolder("Initiatives");
    expect(id).toBe("existing-1");
    expect(
      calls.some((k) => k.method === "POST" && k.url.endsWith("/knowledge-base/folders"))
    ).toBe(false);
  });

  it("creates the folder and returns its new id when the tree has no match", async () => {
    const calls: any[] = [];
    stubFetchRouted(calls, [
      { match: (m, u) => m === "GET" && u.endsWith("/knowledge-base/tree"), json: { roots: [] } },
      {
        match: (m, u) => m === "POST" && u.endsWith("/knowledge-base/folders"),
        json: { id: "new-folder" },
      },
    ]);
    const c = new HindsightClient({ apiUrl: "http://x", bank: "repo-a" });
    const id = await c.ensureFolder("Initiatives");
    expect(id).toBe("new-folder");
    const post = calls.find(
      (k) => k.method === "POST" && k.url.endsWith("/knowledge-base/folders")
    );
    expect(post.body).toEqual({ name: "Initiatives" });
  });
});

describe("HindsightClient.captureInitiative", () => {
  it("new initiative: POSTs a per-initiative page + a marker retain naming the same page id", async () => {
    const calls: any[] = [];
    stubFetchRouted(calls, [
      { match: (m, u) => m === "GET" && u.endsWith("/knowledge-base/tree"), json: { roots: [] } },
      {
        match: (m, u) => m === "POST" && u.endsWith("/knowledge-base/folders"),
        json: { id: "folder-abc" },
      },
      {
        match: (m, u) => m === "POST" && u.endsWith("/knowledge-base/pages"),
        json: { page_id: "pg" },
      },
      {
        match: (m, u) => m === "POST" && u.endsWith("/memories"),
        json: { operation_id: "op-1" },
      },
    ]);
    const c = new HindsightClient({ apiUrl: "http://x", bank: "repo-a" });
    const result = await c.captureInitiative({
      title: "Retry backoff for the uploader",
      summary: "Add exponential backoff so transient upload failures retry.",
    });

    // The returned id is the SERVER-ASSIGNED page id ("pg" from the mock), NOT the derived slug —
    // the /knowledge-base/pages endpoint mints its own id.
    expect(result).toEqual({ page_id: "pg" });

    // Page POST: name = title, nested under the Initiatives folder. The page itself carries NO
    // tags field at all (no tag taxonomy to maintain).
    const pagePost = calls.find(
      (k) => k.method === "POST" && k.url.endsWith("/knowledge-base/pages")
    );
    expect(pagePost).toBeDefined();
    expect(pagePost.body.name).toBe("Retry backoff for the uploader");
    expect(pagePost.body.parent_id).toBe("folder-abc");
    expect(pagePost.body.tags).toEqual(["knowledge:feature-work"]);

    // Marker retain POST to /memories. The page id rides on the metadata and on the context —
    // NEVER on a tag: tags are matched with exact set-ops against a fixed vocabulary, and one
    // more tag value per initiative both isolates nothing and pollutes every fact (#3641).
    const memPost = calls.find((k) => k.method === "POST" && k.url.endsWith("/memories"));
    expect(memPost).toBeDefined();
    const item = memPost.body.items[0];
    expect(item.tags).toEqual(["knowledge:feature-work"]);
    expect(item.metadata).toEqual({ relatedPageId: "pg" });
    // The context is the one channel a page synthesis reads back (reflect strips `metadata` from
    // its search results), so this is what lets the overview link to the REAL page id.
    expect(item.context).toBe("initiative marker for [[page:pg]]");
    expect(item.strategy).toBe("document");
    // A marker consolidates into the same single scope as everything else this plugin writes.
    expect(item.observation_scopes).toBe("shared");
    expect(memPost.body.async).toBe(true);
    // Unique per-marker document id (NOT the page id) so repeated captures accrue.
    expect(item.document_id).not.toBe("pg");
    expect(item.document_id).toContain("initiative-marker-retry-backoff-for-the-uploader-");

    // The returned page id and the id the marker names must be the same (the real page node id).
    expect(item.metadata.relatedPageId).toBe(result.page_id);
    expect(item.context).toContain(`[[page:${result.page_id}]]`);
  });

  it("enhancement (relatesToPageId): NO page POST; marker names the existing page id", async () => {
    const calls: any[] = [];
    stubFetchRouted(calls, [
      {
        match: (m, u) => m === "POST" && u.endsWith("/memories"),
        json: { operation_id: "op-1" },
      },
    ]);
    const c = new HindsightClient({ apiUrl: "http://x", bank: "repo-a" });
    const result = await c.captureInitiative({
      title: "More backoff tuning",
      summary: "Tweak the jitter window.",
      relatesToPageId: "initiative-x",
    });

    expect(result).toEqual({ page_id: "initiative-x" });

    // No new page created for enhancements.
    expect(calls.some((k) => k.method === "POST" && k.url.endsWith("/knowledge-base/pages"))).toBe(
      false
    );
    // No folder lookup/creation either.
    expect(calls.some((k) => k.method === "GET" && k.url.endsWith("/knowledge-base/tree"))).toBe(
      false
    );

    const memPost = calls.find((k) => k.method === "POST" && k.url.endsWith("/memories"));
    expect(memPost).toBeDefined();
    const item = memPost.body.items[0];
    expect(item.tags).toEqual(["knowledge:feature-work"]);
    expect(item.metadata).toEqual({ relatedPageId: "initiative-x" });
    expect(item.context).toBe("initiative marker for [[page:initiative-x]]");
    expect(item.content).toContain("Update to an existing initiative");
  });

  it("applies retain attribution to initiative markers", async () => {
    const calls: any[] = [];
    stubFetchRouted(calls, [
      {
        match: (m, u) => m === "POST" && u.endsWith("/memories"),
        json: { operation_id: "op-1" },
      },
    ]);
    const c = new HindsightClient({ apiUrl: "http://x", bank: "repo-a" });
    await c.captureInitiative({
      title: "Attributed initiative",
      summary: "Keep its project provenance.",
      relatesToPageId: "initiative-x",
      stamp: {
        tags: ["project:repo-a"],
        metadata: { project: "repo-a", source: "configured" },
      },
    });

    const item = calls.find((k) => k.url.endsWith("/memories")).body.items[0];
    expect(item.tags).toEqual(["project:repo-a", "knowledge:feature-work"]);
    // The configured attribution survives alongside the page id the marker adds.
    expect(item.metadata).toEqual({
      project: "repo-a",
      source: "configured",
      relatedPageId: "initiative-x",
    });
  });
});

describe("HindsightClient.configureBank template import", () => {
  it("POSTs the CODING_BANK_TEMPLATE manifest to /import, then seeds pages via the knowledge base", async () => {
    const calls: any[] = [];
    stubFetchRouted(calls, [
      { match: (m, u) => m === "GET" && u.endsWith("/knowledge-base/tree"), json: { roots: [] } },
    ]);
    const c = new HindsightClient({ apiUrl: "http://x", bank: "repo-a" });
    await c.configureBank();

    const importPosts = calls.filter((k) => k.method === "POST" && k.url.endsWith("/import"));
    expect(importPosts).toHaveLength(1);
    // Config before the knowledge base — page seeding needs the bank to exist. (The very first
    // call is the GET /config probe that decides whether the missions are still ours to seed.)
    expect(calls.indexOf(importPosts[0])).toBeLessThan(
      calls.findIndex((k) => k.url.includes("/knowledge-base/"))
    );

    const body = importPosts[0].body;
    expect(body.version).toBe("1");

    // bank config section
    expect(typeof body.bank.reflect_mission).toBe("string");
    expect(body.bank.retain_default_strategy).toBe("git");
    expect(Object.keys(body.bank.retain_strategies)).toEqual(
      expect.arrayContaining(["git", "gitlog", "conversation", "document"])
    );
    expect(body.bank.entity_labels).toEqual(
      expect.arrayContaining([expect.objectContaining({ key: "knowledge", tag: true })])
    );
    expect(body.bank.entities_allow_free_form).toBe(true);

    // The manifest must NOT carry pages: the template's mental_models key creates mental models
    // with no knowledge-base node, which are invisible to page search and unreadable by node id.
    expect(body).not.toHaveProperty("mental_models");
    expect(
      calls.filter((k) => k.method === "POST" && k.url.endsWith("/knowledge-base/pages"))
    ).toHaveLength(PAGES.length);
  });

  it("configureBank({reset: true}) DELETEs the bank before the import POST", async () => {
    const calls: any[] = [];
    stubFetchRouted(calls, [
      { match: (m, u) => m === "GET" && u.endsWith("/knowledge-base/tree"), json: { roots: [] } },
    ]);
    const c = new HindsightClient({ apiUrl: "http://x", bank: "repo-a" });
    await c.configureBank({ reset: true });

    expect(calls[0].method).toBe("DELETE");
    expect(calls[0].url.endsWith("/import")).toBe(false);
    expect(calls[1].method).toBe("POST");
    expect(calls[1].url.endsWith("/import")).toBe(true);
  });
});

describe("HindsightClient.configureBank — missions are seeded once (#2492)", () => {
  const routes = (overrides: Record<string, unknown> | undefined, ok = true) => [
    {
      match: (m: string, u: string) => m === "GET" && u.endsWith("/knowledge-base/tree"),
      json: { roots: [] },
    },
    {
      match: (m: string, u: string) => m === "GET" && u.endsWith("/config"),
      json: { bank_id: "repo-a", config: {}, overrides },
      ok,
    },
  ];

  const run = async (calls: any[], routeList: any[]) => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init: any) => {
        const method = init?.method;
        calls.push({ url, method, body: init?.body ? JSON.parse(init.body) : undefined });
        const route = routeList.find((r) => r.match(method, url));
        return {
          ok: route?.ok ?? true,
          status: route?.ok === false ? 404 : 200,
          json: async () => route?.json ?? { ok: true },
        } as any;
      })
    );
    await new HindsightClient({ apiUrl: "http://x", bank: "repo-a" }).configureBank();
    // Optional: a bank that already carries the whole structure is not imported to at all (#3927).
    return calls.find((k) => k.method === "POST" && k.url.endsWith("/import"))?.body;
  };

  it("seeds the full template — missions included — on a bank with no mission overrides", async () => {
    const body = await run([], routes({}));
    expect(typeof body.bank.reflect_mission).toBe("string");
    expect(body.bank.retain_strategies).toBeDefined();
  });

  it("leaves the missions alone once the bank carries its own", async () => {
    // The user rewrote reflect_mission in the control plane; a later seed pass must not stamp the
    // default back over it — the whole of #2492.
    const body = await run([], routes({ reflect_mission: "MY OWN MISSION" }));
    expect(body.bank.reflect_mission).toBeUndefined();
    expect(body.bank.retain_mission).toBeUndefined();
    expect(body.bank.observations_mission).toBeUndefined();
  });

  it("still adds the strategies and labels the plugin writes through", async () => {
    // Not preferences: without `conversation` the session write-back silently extracts under the
    // bank's own config (the server warns on an unknown strategy, it does not reject), so a
    // transcript would get whatever a commit diff gets. A bank that has none must still get them.
    const body = await run([], routes({ retain_mission: "mine" }));
    expect(Object.keys(body.bank.retain_strategies)).toEqual(
      expect.arrayContaining(["git", "gitlog", "conversation", "document"])
    );
    expect(body.bank.entity_labels).toEqual(
      expect.arrayContaining([expect.objectContaining({ key: "knowledge" })])
    );
  });

  it("treats a blank override as unset", async () => {
    const body = await run([], routes({ reflect_mission: "   " }));
    expect(typeof body.bank.reflect_mission).toBe("string");
  });

  it("seeds when the bank-config API is unavailable — nothing to preserve there", async () => {
    // With that API off a user cannot set per-bank missions at all, so there is no edit to protect.
    const body = await run([], routes(undefined, false));
    expect(typeof body.bank.reflect_mission).toBe("string");
  });

  it("makes no import call at all once the bank carries the whole structure (#3927)", async () => {
    // The settled steady state: deepen runs on every session start, and on a bank that already has
    // everything there is nothing left to write. Not a harmless no-op POST — an import of the
    // strategies IS the overwrite, since the server replaces the whole map.
    const calls: any[] = [];
    await run(
      calls,
      routes({
        reflect_mission: "mine",
        retain_default_strategy: "git",
        entities_allow_free_form: true,
        retain_strategies: RETAIN_STRATEGIES,
        entity_labels: [KNOWLEDGE_LABELS],
      })
    );
    expect(calls.some((k) => k.method === "POST" && k.url.endsWith("/import"))).toBe(false);
    // Pages are seeded either way — they are not bank configuration.
    expect(calls.some((k) => k.url.includes("/knowledge-base/"))).toBe(true);
  });

  it("manageBankConfig: false keeps the plugin out of the bank's config entirely", async () => {
    const calls: any[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init: any) => {
        calls.push({ url, method: init?.method });
        return { ok: true, status: 200, json: async () => ({ roots: [] }) } as any;
      })
    );
    await new HindsightClient({ apiUrl: "http://x", bank: "repo-a" }).configureBank({
      manage: false,
    });
    expect(calls.some((k) => k.url.endsWith("/import"))).toBe(false);
    // Not even the probe: the bank's configuration is none of this plugin's business.
    expect(calls.some((k) => k.url.endsWith("/config"))).toBe(false);
    expect(calls.some((k) => k.url.includes("/knowledge-base/"))).toBe(true);
  });

  it("re-seeds the missions after an explicit reset", async () => {
    const calls: any[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init: any) => {
        calls.push({
          url,
          method: init?.method,
          body: init?.body ? JSON.parse(init.body) : undefined,
        });
        return { ok: true, status: 200, json: async () => ({ roots: [] }) } as any;
      })
    );
    await new HindsightClient({ apiUrl: "http://x", bank: "repo-a" }).configureBank({
      reset: true,
    });
    const body = calls.find((k) => k.method === "POST" && k.url.endsWith("/import")).body;
    expect(typeof body.bank.reflect_mission).toBe("string");
    // reset deletes the bank, so there is nothing to probe
    expect(calls.some((k) => k.method === "GET" && k.url.endsWith("/config"))).toBe(false);
  });
});

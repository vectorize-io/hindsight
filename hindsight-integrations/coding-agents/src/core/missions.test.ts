import { describe, expect, it } from "vitest";
import { resolveConfig } from "./config";
import {
  type BankOverrides,
  buildPageTrigger,
  CODING_BANK_TEMPLATE,
  codingBankManifest,
  DEFAULT_PAGE_TRIGGER_CRON,
  expandCronHash,
  KNOWLEDGE_LABELS,
  PAGE_NAMES,
  pagesFor,
  pageScopeRule,
  pageTriggerDrifted,
  pageTriggerFor,
  pageTriggerPatch,
  PAGE_FACT_TYPES,
  REFLECT_MISSION,
  RETAIN_STRATEGIES,
} from "./missions";

/**
 * The page trigger is what a project's knowledge pages COST to keep current: auto-refresh means one
 * LLM synthesis per page per consolidation, which on a few auto-surveyed repos is real money
 * (#3506). It was hardcoded, so the only workaround was patching dist/ or fixing pages up after
 * the fact.
 */
describe("buildPageTrigger", () => {
  it("defaults to the hourly staggered schedule", () => {
    expect(buildPageTrigger()).toMatchObject({
      fact_types: PAGE_FACT_TYPES,
      refresh_cron: DEFAULT_PAGE_TRIGGER_CRON,
    });
    expect(buildPageTrigger(resolveConfig({}))).toEqual(buildPageTrigger());
  });

  /**
   * The server defaults a tagged model to `all_strict`, which EXCLUDES untagged memories — and
   * `DEFAULT_OBSERVATION_SCOPES = "shared"` makes every observation in these banks untagged
   * (#3564). Left at the default, a page asked for the `observation` fact type and could never
   * retrieve one (#3641).
   */
  it("admits the untagged shared observations these pages ask for", () => {
    expect(buildPageTrigger().tags_match).toBe("all");
    expect(PAGE_FACT_TYPES).toContain("observation");
    // Not the strict variants: those are the ones that drop untagged rows.
    for (const type of ["auto-refresh", "cron", "manual"] as const) {
      expect(buildPageTrigger(resolveConfig({ pageTriggerType: type })).tags_match).toBe("all");
    }
  });

  it("puts pages on a schedule", () => {
    const trigger = buildPageTrigger(
      resolveConfig({ pageTriggerType: "cron", pageTriggerCron: "0 3 * * *" })
    );
    expect(trigger.refresh_cron).toBe("0 3 * * *");
    // The API rejects a trigger carrying both — a page refreshes on one schedule or the other.
    expect(trigger.refresh_after_consolidation).toBeUndefined();
  });

  it("stops refreshing pages on request", () => {
    const trigger = buildPageTrigger(resolveConfig({ pageTriggerType: "manual" }));
    expect(trigger.refresh_after_consolidation).toBe(false);
    expect(trigger.refresh_cron).toBeUndefined();
  });

  /**
   * HOW a page refreshes belongs to the server: `create_knowledge_page` merges a client's fields
   * over KNOWLEDGE_PAGE_DEFAULT_TRIGGER (delta, no sibling pages in the reflect loop). Restating
   * those here would freeze a copy of someone else's defaults — so the trigger says nothing but
   * what this plugin actually decides.
   */
  it.each([
    ["auto-refresh", ["fact_types", "tags_match", "refresh_after_consolidation"]],
    ["cron", ["fact_types", "tags_match", "refresh_cron"]],
    ["manual", ["fact_types", "tags_match", "refresh_after_consolidation"]],
  ] as const)("states nothing the server owns under %s", (pageTriggerType, keys) => {
    const trigger = buildPageTrigger(
      resolveConfig({ pageTriggerType, pageTriggerCron: "0 3 * * *" })
    );
    expect(Object.keys(trigger).sort()).toEqual([...keys].sort());
  });
});

/**
 * One `pageTriggerCron` is shared by every page in every bank running this plugin, so a literal
 * expression schedules ALL of them on the one minute it names — a pile of LLM syntheses on the
 * worker pool that also serves retain. `H` is Jenkins' answer: hash the field per page.
 */
describe("hashed cron fields", () => {
  const cron = (raw: string, bank: string, page: string) =>
    pageTriggerFor(
      buildPageTrigger(resolveConfig({ pageTriggerType: "cron", pageTriggerCron: raw })),
      bank,
      page
    ).refresh_cron;

  it("leaves an expression without H exactly as written", () => {
    expect(cron("0 3 * * *", "repo-a", "Component map")).toBe("0 3 * * *");
    expect(cron("0 3 * * *", "repo-b", "Core concepts")).toBe("0 3 * * *");
  });

  it("resolves H to an ordinary cron expression the server can parse", () => {
    // `H` never leaves this package — `refresh_cron` is standard 5-field cron server-side.
    expect(cron("H H * * *", "repo-a", "Component map")).toMatch(
      /^(?:[0-9]|[1-5][0-9]) (?:[0-9]|1[0-9]|2[0-3]) \* \* \*$/
    );
    expect(cron("H * * * *", "repo-a", "Component map")).toMatch(
      /^(?:[0-9]|[1-5][0-9]) \* \* \* \*$/
    );
  });

  it("keeps the fields the operator wrote and hashes only the H", () => {
    // "spread within 03:00" — the hour is a decision, the minute is not.
    const daily = cron("H 3 * * *", "repo-a", "Component map");
    expect(daily).toMatch(/^\d+ 3 \* \* \*$/);
    // A range bounds where the hash may land: spread across the night only.
    const hours = new Set(
      Array.from({ length: 60 }, (_, i) =>
        Number(cron("H H(0-5) * * *", `repo-${i}`, "Component map")!.split(" ")[1])
      )
    );
    expect(Math.min(...hours)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...hours)).toBeLessThanOrEqual(5);
    expect(hours.size).toBeGreaterThan(1);
  });

  it("gives each page its own slot, stably", () => {
    const one = cron("H H * * *", "repo-a", "Component map");
    // Stable: a page keeps its slot across runs, machines and releases, or every session would
    // reschedule it (and the seed PATCH would report drift forever).
    expect(cron("H H * * *", "repo-a", "Component map")).toBe(one);
    expect(cron("H H * * *", "repo-a", "Core concepts")).not.toBe(one);
    expect(cron("H H * * *", "repo-b", "Component map")).not.toBe(one);
  });

  it("spreads a shared config across banks instead of piling them on one minute", () => {
    const slots = Array.from({ length: 50 }, (_, i) =>
      cron("H H * * *", `repo-${i}`, "Component map")
    );
    // The whole point: 50 banks copying the same setting do not collide. Hashing distributes
    // approximately — it does not partition — so a couple of collisions are expected, not a bug.
    expect(new Set(slots).size).toBeGreaterThan(45);
  });

  it("does not derive minute and hour from the same number", () => {
    // Hashing the field index alongside the seed is what keeps "H H * * *" worth 1440 slots
    // rather than 60 correlated ones.
    const minutes = new Set<number>();
    const hours = new Set<number>();
    for (let i = 0; i < 200; i++) {
      const [m, h] = cron("H H * * *", `repo-${i}`, "Component map")!.split(" ");
      minutes.add(Number(m));
      hours.add(Number(h));
    }
    expect(minutes.size).toBeGreaterThan(40);
    expect(hours.size).toBe(24);
  });

  it("leaves a malformed expression for the server to reject", () => {
    // Rewriting it here would invent a schedule nobody asked for; resolveConfig refuses it first.
    expect(expandCronHash("H(9-3) * * * *", "seed")).toBe("H(9-3) * * * *");
    expect(expandCronHash("H H", "seed")).toBe("H H");
  });

  it("passes a trigger with no cron through untouched", () => {
    const auto = buildPageTrigger(resolveConfig({ pageTriggerType: "auto-refresh" }));
    expect(pageTriggerFor(auto, "repo-a", "Component map")).toBe(auto);
  });
});

/**
 * An existing page is re-synced to whatever the config says, so a changed default reaches a bank
 * that was seeded under the old one — the point of the migration off auto-refresh.
 */
describe("pageTriggerDrifted", () => {
  const settled = (name: string) => pageTriggerFor(buildPageTrigger(), "repo-a", name);

  it("sees no drift in the policy it just wrote", () => {
    const desired = settled("Component map");
    // What the server reports back: the effective policy, with the exclusive counterpart at its
    // default rather than absent.
    expect(pageTriggerDrifted({ ...desired, refresh_after_consolidation: false }, desired)).toBe(
      false
    );
  });

  it("sees a page still on the old auto-refresh default as drifted", () => {
    expect(
      pageTriggerDrifted(
        { tags_match: "all", refresh_after_consolidation: true, refresh_cron: null },
        settled("Component map")
      )
    ).toBe(true);
  });

  it("sees a different schedule as drifted", () => {
    const desired = settled("Component map");
    expect(pageTriggerDrifted({ tags_match: "all", refresh_cron: "0 3 * * *" }, desired)).toBe(
      true
    );
    // Compared against the page's OWN resolved cron, never the shared `H * * * *` — otherwise
    // every page would look drifted on every session.
    expect(desired.refresh_cron).not.toBe(DEFAULT_PAGE_TRIGGER_CRON);
    expect(
      pageTriggerDrifted({ ...desired, refresh_cron: DEFAULT_PAGE_TRIGGER_CRON }, desired)
    ).toBe(true);
  });

  it("still sees the tags_match drift it was originally written for", () => {
    const desired = settled("Component map");
    expect(pageTriggerDrifted({ ...desired, tags_match: "all_strict" }, desired)).toBe(true);
  });
});

/**
 * The server drops the unstated counterpart of a TRUTHY refresh field, so a cron patch clears
 * auto-refresh by itself. `manual` is falsy and clears nothing — hence the explicit null.
 */
describe("pageTriggerPatch", () => {
  it("leaves a cron or auto-refresh patch alone", () => {
    for (const type of ["cron", "auto-refresh"] as const) {
      const desired = buildPageTrigger(resolveConfig({ pageTriggerType: type }));
      expect(pageTriggerPatch(desired)).toEqual(desired);
    }
  });

  it("clears an existing schedule when moving a page to manual", () => {
    const patch = pageTriggerPatch(buildPageTrigger(resolveConfig({ pageTriggerType: "manual" })));
    expect(patch.refresh_after_consolidation).toBe(false);
    expect(patch.refresh_cron).toBeNull();
  });
});

describe("page trigger config resolution", () => {
  // The default: hourly, each page on its own hashed minute. Auto-refresh — one LLM synthesis per
  // page per consolidation — is now opt-in.
  it("schedules an unconfigured repo's pages hourly and staggered", () => {
    expect(resolveConfig({}).pageTriggerType).toBe("cron");
    expect(resolveConfig({}).pageTriggerCron).toBe(DEFAULT_PAGE_TRIGGER_CRON);
    expect(DEFAULT_PAGE_TRIGGER_CRON).toBe("H * * * *");
    const trigger = buildPageTrigger(resolveConfig({}));
    expect(trigger.refresh_cron).toBe(DEFAULT_PAGE_TRIGGER_CRON);
    expect(trigger.refresh_after_consolidation).toBeUndefined();
    // And the `H` is resolved per page before it is sent — see "hashed cron fields" above.
    expect(pageTriggerFor(trigger, "repo-a", "Component map").refresh_cron).toMatch(
      /^(?:[0-9]|[1-5][0-9]) \* \* \* \*$/
    );
  });

  // The API rejects a cron trigger with no expression, so honouring an empty one literally would
  // fail page creation outright. The default schedule stands in; "manual" is how you ask for no
  // refreshes.
  it("uses the default schedule when cron is asked for without an expression", () => {
    for (const raw of [{}, { pageTriggerCron: "   " }] as const) {
      const cfg = resolveConfig({ pageTriggerType: "cron", ...raw });
      expect(cfg.pageTriggerType).toBe("cron");
      expect(cfg.pageTriggerCron).toBe(DEFAULT_PAGE_TRIGGER_CRON);
    }
  });

  /**
   * `expandCronHash` leaves an expression it cannot read alone, so an unchecked malformed `H`
   * would reach the server verbatim and fail page creation with a parse error naming syntax this
   * package invented. Only the H fields are checked — ordinary cron syntax is the server's.
   */
  it("falls back to the default schedule on a malformed hashed field", () => {
    for (const bad of ["H(9-3) * * * *", "H(0-99) * * * *", "H H", "Hx * * * *"]) {
      const cfg = resolveConfig({ pageTriggerType: "cron", pageTriggerCron: bad });
      expect(cfg.pageTriggerType).toBe("cron");
      expect(cfg.pageTriggerCron).toBe(DEFAULT_PAGE_TRIGGER_CRON);
    }
  });

  it("accepts a well-formed hashed cron", () => {
    for (const good of ["H H * * *", "H * * * *", "H 3 * * *", "H H(0-5) * * *"]) {
      const cfg = resolveConfig({ pageTriggerType: "cron", pageTriggerCron: good });
      expect(cfg.pageTriggerType).toBe("cron");
      expect(cfg.pageTriggerCron).toBe(good);
    }
  });

  it("keeps auto-refresh available for a repo that opts into it", () => {
    const cfg = resolveConfig({ pageTriggerType: "auto-refresh" });
    expect(cfg.pageTriggerType).toBe("auto-refresh");
    expect(cfg.pageTriggerCron).toBeUndefined();
    expect(buildPageTrigger(cfg).refresh_after_consolidation).toBe(true);
  });

  it("ignores a value that is not one of the three types", () => {
    const cfg = resolveConfig({ pageTriggerType: "whenever" as never });
    expect(cfg.pageTriggerType).toBe("cron");
    expect(cfg.pageTriggerCron).toBe(DEFAULT_PAGE_TRIGGER_CRON);
  });
});

/**
 * Applying the whole template on every pass is how a plugin takes a bank over. #1270 fixed it for
 * OpenClaw's missions, #2492 for this plugin's; #3927 is the same bug on the half both fixes left
 * un-guarded — the strategies and labels, re-sent wholesale on every session start. Because the
 * server stores each of those as ONE config value, re-sending them deleted a user's own strategy,
 * reverted their edits to the plugin's, and could leave `retain_default_strategy` pointing at a
 * strategy that no longer existed.
 */
describe("codingBankManifest (#3927)", () => {
  const bankOf = (overrides: BankOverrides | undefined) => codingBankManifest(overrides)?.bank;

  it("seeds the full template on a bank with no overrides of its own", () => {
    // Unreadable overrides (no bank yet, or the bank-config API switched off) seed the same lot:
    // nothing can have been customised through an API that is not there.
    for (const empty of [undefined, {}]) {
      expect(codingBankManifest(empty)).toEqual(CODING_BANK_TEMPLATE);
    }
  });

  it("adds a strategy a newer plugin release introduced, keeping the ones already there", () => {
    // The reason the re-apply exists at all: a bank seeded before `survey` shipped must still get
    // it, or the survey's documents retain under a strategy the bank does not have.
    const { survey: _survey, ...seededByAnOlderRelease } = RETAIN_STRATEGIES;
    const bank = bankOf({
      reflect_mission: "seeded",
      retain_strategies: { ...seededByAnOlderRelease, mycustom: { retain_chunk_size: 500 } },
    });
    expect(Object.keys(bank!.retain_strategies as object).sort()).toEqual([
      "conversation",
      "document",
      "git",
      "gitlog",
      "mycustom",
      "survey",
    ]);
    expect((bank!.retain_strategies as Record<string, unknown>).survey).toEqual(
      RETAIN_STRATEGIES.survey
    );
  });

  it("never deletes a strategy the user defined, nor reverts their edits to ours", () => {
    const mine = {
      ...RETAIN_STRATEGIES,
      // The user rewrote the conversation strategy's mission; that is theirs to decide.
      conversation: { retain_mission: "MINE", retain_extraction_mode: "concise" },
      mycustom: { retain_chunk_size: 500 },
    };
    // Nothing missing => the field is not written at all, so no import can revert it.
    expect(bankOf({ reflect_mission: "seeded", retain_strategies: mine })).not.toHaveProperty(
      "retain_strategies"
    );
  });

  it("re-syncs the extraction mode of the plugin's own strategies to the configured one (#4560)", () => {
    // A bank seeded by a release that defaulted to verbose, plus a user tweak and a user strategy.
    const seeded = Object.fromEntries(
      Object.entries(RETAIN_STRATEGIES).map(([n, d]) => [
        n,
        d.retain_extraction_mode === "custom" ? d : { ...d, retain_extraction_mode: "verbose" },
      ])
    );
    const current = {
      ...seeded,
      conversation: { ...seeded.conversation, retain_chunk_size: 500 },
      mycustom: { retain_extraction_mode: "verbose" },
    };
    const strategies = bankOf({ reflect_mission: "seeded", retain_strategies: current })!
      .retain_strategies as Record<string, Record<string, unknown>>;
    for (const name of ["git", "gitlog", "conversation", "document"])
      expect(strategies[name].retain_extraction_mode).toBe("concise");
    // Only the mode moves: the user's other edits, their own strategy and the survey stay put.
    expect(strategies.conversation.retain_chunk_size).toBe(500);
    expect(strategies.mycustom).toEqual({ retain_extraction_mode: "verbose" });
    expect(strategies.survey).toEqual(RETAIN_STRATEGIES.survey);

    // An explicit choice is honoured the same way, and seeds a new bank with it.
    const verbose = codingBankManifest(
      { reflect_mission: "seeded", retain_strategies: RETAIN_STRATEGIES },
      "verbose"
    )!.bank.retain_strategies as Record<string, Record<string, unknown>>;
    expect(verbose.git.retain_extraction_mode).toBe("verbose");
    expect(codingBankManifest(undefined, "chunks")!.bank.retain_extraction_mode).toBe("chunks");
  });

  it("leaves a bank that already carries the whole structure completely alone", () => {
    // Every session start calls this. On a settled bank it must be a no-op — no manifest, no POST.
    expect(
      codingBankManifest({
        reflect_mission: "seeded",
        retain_default_strategy: "mycustom",
        entities_allow_free_form: false,
        retain_strategies: RETAIN_STRATEGIES,
        entity_labels: [KNOWLEDGE_LABELS],
      })
    ).toBeUndefined();
  });

  it("keeps retain_default_strategy pointing where the user aimed it", () => {
    // The dangling half of #3927: the map was replaced (deleting `mycustom`) while the pointer to
    // it survived, leaving the bank naming a strategy that no longer existed.
    const bank = bankOf({
      retain_default_strategy: "mycustom",
      retain_strategies: { mycustom: { retain_chunk_size: 500 } },
    });
    expect(bank).not.toHaveProperty("retain_default_strategy");
    expect(bank!.retain_strategies).toHaveProperty("mycustom");
  });

  it("adds the knowledge label group alongside the user's own, and only once", () => {
    const mine = { key: "audience", type: "multi-values", values: [] };
    expect(bankOf({ reflect_mission: "seeded", entity_labels: [mine] })!.entity_labels).toEqual([
      mine,
      KNOWLEDGE_LABELS,
    ]);
    // Already present — including a version the user reworded — is left as it is.
    expect(
      bankOf({
        reflect_mission: "seeded",
        entity_labels: [{ ...KNOWLEDGE_LABELS, description: "my wording" }],
      })
    ).not.toHaveProperty("entity_labels");
  });

  it("still seeds the missions as a group, once (#2492)", () => {
    // Spelled out rather than imported: the contract is these three fields, whatever the
    // implementation happens to call its list of them.
    const missions = ["reflect_mission", "retain_mission", "observations_mission"] as const;
    expect(bankOf({})!.reflect_mission).toBe(REFLECT_MISSION);
    for (const field of missions) {
      // Any one mission present means the bank has been through here; none of the three is rewritten.
      const bank = bankOf({ [field]: "MY OWN MISSION" })!;
      for (const f of missions) expect(bank).not.toHaveProperty(f);
    }
    // A blank override is not a choice.
    expect(bankOf({ reflect_mission: "   " })!.reflect_mission).toBe(REFLECT_MISSION);
  });
});

/**
 * WHICH pages a repo gets and what each one asks. The config surface behind #4460, where a
 * `source_query` edited through the API was silently restored on the next session because the
 * taxonomy was the only wording `seedPages` would accept.
 */
describe("pagesFor page configuration", () => {
  const scope = pageScopeRule("repo-a");

  it("seeds the whole taxonomy with its built-in queries when no config is given", () => {
    const pages = pagesFor("repo-a");
    expect(pages.map((p) => p.name)).toEqual([...PAGE_NAMES]);
    for (const page of pages) expect(page.source_query.endsWith(scope)).toBe(true);
  });

  it("false skips a page and leaves the rest in taxonomy order", () => {
    const pages = pagesFor("repo-a", { "Component map": false });
    expect(pages.map((p) => p.name)).toEqual(PAGE_NAMES.filter((n) => n !== "Component map"));
  });

  it("a custom source_query replaces the built-in one, scoping clause still appended", () => {
    const pages = pagesFor("repo-a", { "Key decisions and rationale": { source_query: "why?" } });
    const page = pages.find((p) => p.name === "Key decisions and rationale")!;
    // The clause rides along with a reworded query: it is what keeps a dependency's decisions off
    // this project's page (#3476), which rewording the question is not a choice to take on.
    expect(page.source_query).toBe("why?" + scope);
    // The tier tag is the taxonomy's — it selects the facts the synthesis reads, so a reworded
    // query must not drop it.
    expect(page.tags).toEqual(["knowledge:decision"]);
    expect(pages.find((p) => p.name === "Core concepts")!.source_query).toContain("core concepts");
  });

  it("matches names case- and whitespace-insensitively, as seedPages matches live pages", () => {
    expect(pagesFor("repo-a", { "  component MAP  ": false }).map((p) => p.name)).not.toContain(
      "Component map"
    );
  });

  it("customPages are appended after the taxonomy, scoped, with their own tags", () => {
    const pages = pagesFor(
      "repo-a",
      {},
      {
        "Security posture": {
          source_query: "what are our security decisions?",
          tags: ["knowledge:decision"],
        },
      }
    );
    expect(pages).toHaveLength(PAGE_NAMES.length + 1);
    const custom = pages[pages.length - 1];
    expect(custom.name).toBe("Security posture");
    expect(custom.source_query).toBe("what are our security decisions?" + scope);
    expect(custom.tags).toEqual(["knowledge:decision"]);
  });

  it("a custom page without tags gets none — `all` then puts no tag constraint on it", () => {
    const pages = pagesFor("repo-a", {}, { Roadmap: { source_query: "where is this going?" } });
    expect(pages.find((p) => p.name === "Roadmap")!.tags).toEqual([]);
  });

  it("disabling a taxonomy page and adding one of your own compose", () => {
    const pages = pagesFor(
      "repo-a",
      { "Component map": false },
      { Roadmap: { source_query: "where is this going?" } }
    );
    expect(pages.map((p) => p.name)).toEqual([
      ...PAGE_NAMES.filter((n) => n !== "Component map"),
      "Roadmap",
    ]);
  });
});

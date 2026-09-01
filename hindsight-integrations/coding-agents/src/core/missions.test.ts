import { describe, expect, it } from "vitest";
import { resolveConfig } from "./config";
import {
  type BankOverrides,
  buildPageTrigger,
  CODING_BANK_TEMPLATE,
  codingBankManifest,
  KNOWLEDGE_LABELS,
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
  it("defaults to the auto-refresh policy every page shipped with", () => {
    expect(buildPageTrigger()).toMatchObject({
      fact_types: PAGE_FACT_TYPES,
      refresh_after_consolidation: true,
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

describe("page trigger config resolution", () => {
  it("keeps today's behaviour when nothing is configured", () => {
    expect(resolveConfig({}).pageTriggerType).toBe("auto-refresh");
    expect(resolveConfig({}).pageTriggerCron).toBeUndefined();
  });

  // The API rejects a cron trigger with no expression, so honouring this literally would fail page
  // creation outright. Falling back to the default keeps pages working; "manual" is how you ask
  // for no refreshes.
  it("falls back to auto-refresh when cron is asked for without an expression", () => {
    expect(resolveConfig({ pageTriggerType: "cron" }).pageTriggerType).toBe("auto-refresh");
    expect(resolveConfig({ pageTriggerType: "cron", pageTriggerCron: "   " }).pageTriggerType).toBe(
      "auto-refresh"
    );
  });

  it("ignores a value that is not one of the three types", () => {
    expect(resolveConfig({ pageTriggerType: "whenever" as never }).pageTriggerType).toBe(
      "auto-refresh"
    );
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
      // The user made the conversation strategy concise and small; that is theirs to decide.
      conversation: { retain_mission: "MINE", retain_extraction_mode: "concise" },
      mycustom: { retain_chunk_size: 500 },
    };
    // Nothing missing => the field is not written at all, so no import can revert it.
    expect(bankOf({ reflect_mission: "seeded", retain_strategies: mine })).not.toHaveProperty(
      "retain_strategies"
    );
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

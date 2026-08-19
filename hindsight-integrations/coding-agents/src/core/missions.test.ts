import { describe, expect, it } from "vitest";
import { resolveConfig } from "./config";
import { buildPageTrigger, PAGE_FACT_TYPES } from "./missions";

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
    ["auto-refresh", ["fact_types", "refresh_after_consolidation"]],
    ["cron", ["fact_types", "refresh_cron"]],
    ["manual", ["fact_types", "refresh_after_consolidation"]],
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

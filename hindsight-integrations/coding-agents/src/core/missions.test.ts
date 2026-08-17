import { describe, expect, it } from "vitest";
import { resolveConfig } from "./config";
import { buildPageTrigger, PAGE_FACT_TYPES } from "./missions";

/**
 * The page trigger is what a project's knowledge pages COST to keep current: reactive means one
 * LLM synthesis per page per consolidation, which on a few auto-surveyed repos is real money
 * (#3506). It was hardcoded, so the only workaround was patching dist/ or fixing pages up after
 * the fact.
 */
describe("buildPageTrigger", () => {
  it("defaults to the reactive policy every page shipped with", () => {
    expect(buildPageTrigger()).toEqual({
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

  it("carries delta mode, so a refresh edits the page instead of rebuilding it", () => {
    expect(buildPageTrigger(resolveConfig({ pageTriggerMode: "delta" })).mode).toBe("delta");
  });

  // Sending "full" would be a no-op against the server default, but it would also stamp the mode
  // onto a page whose owner set it in the control plane — so an unset mode says nothing at all.
  it("says nothing about the mode when it is unset", () => {
    expect(buildPageTrigger()).not.toHaveProperty("mode");
  });
});

describe("page trigger config resolution", () => {
  it("keeps today's behaviour when nothing is configured", () => {
    expect(resolveConfig({}).pageTriggerType).toBe("reactive");
    expect(resolveConfig({}).pageTriggerMode).toBeUndefined();
  });

  // The API rejects a cron trigger with no expression, so honouring this literally would fail page
  // creation outright. Falling back to the default keeps pages working; "manual" is how you ask
  // for no refreshes.
  it("falls back to reactive when cron is asked for without an expression", () => {
    expect(resolveConfig({ pageTriggerType: "cron" }).pageTriggerType).toBe("reactive");
    expect(resolveConfig({ pageTriggerType: "cron", pageTriggerCron: "   " }).pageTriggerType).toBe(
      "reactive"
    );
  });

  it("ignores a value that is not one of the three modes", () => {
    expect(resolveConfig({ pageTriggerType: "whenever" as never }).pageTriggerType).toBe(
      "reactive"
    );
    expect(resolveConfig({ pageTriggerMode: "surgical" as never }).pageTriggerMode).toBeUndefined();
  });
});

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { HindsightClient } from "@vectorize-io/hindsight-client";
import type { PluginConfig } from "./types.js";
import {
  applyConfiguredBankDefaults,
  buildBankConfigApiUpdates,
  buildCreateBankDefaults,
  hasConfiguredBankDefaults,
  normalizeDispositionTrait,
  normalizeEntityLabels,
  normalizeRetainExtractionMode,
  patchBankConfig,
} from "./bank-defaults.js";

const clientOpts = { baseUrl: "http://localhost:8888", apiKey: "test-token" };

describe("hasConfiguredBankDefaults", () => {
  it("returns false when no bank default options are set", () => {
    expect(hasConfiguredBankDefaults({})).toBe(false);
    expect(hasConfiguredBankDefaults({ autoRecall: true })).toBe(false);
  });

  it("returns true when any bank default or mission is configured", () => {
    expect(hasConfiguredBankDefaults({ retainMission: "Extract facts." })).toBe(true);
    expect(hasConfiguredBankDefaults({ retainExtractionMode: "verbose" })).toBe(true);
    expect(hasConfiguredBankDefaults({ enableAutoConsolidation: false })).toBe(true);
    expect(
      hasConfiguredBankDefaults({
        entityLabels: [{ name: "person", description: "A human" }],
      })
    ).toBe(true);
  });
});

describe("buildCreateBankDefaults", () => {
  it("includes only explicitly set fields", () => {
    const config: PluginConfig = {
      retainExtractionMode: "verbose",
      enableObservations: true,
      dispositionSkepticism: 4,
      dispositionLiteralism: 2,
      dispositionEmpathy: 5,
    };
    expect(buildCreateBankDefaults(config)).toEqual({
      retainExtractionMode: "verbose",
      enableObservations: true,
      dispositionSkepticism: 4,
      dispositionLiteralism: 2,
      dispositionEmpathy: 5,
    });
  });

  it("still includes missions when configured", () => {
    const config: PluginConfig = {
      bankMission: "Reflect mission.",
      retainMission: "Retain mission.",
      observationsMission: "Observations mission.",
      retainExtractionMode: "verbose",
    };
    expect(buildCreateBankDefaults(config)).toEqual({
      reflectMission: "Reflect mission.",
      retainMission: "Retain mission.",
      observationsMission: "Observations mission.",
      retainExtractionMode: "verbose",
    });
  });
});

describe("buildBankConfigApiUpdates", () => {
  it("includes entityLabels and enableAutoConsolidation when set", () => {
    const labels = [
      { name: "person", description: "Human user" },
      { name: "project", description: "Software project" },
    ];
    const config: PluginConfig = {
      entityLabels: labels,
      enableAutoConsolidation: true,
    };
    expect(buildBankConfigApiUpdates(config)).toEqual({
      entity_labels: labels,
      enable_auto_consolidation: true,
    });
  });

  it("omits unset config-api fields", () => {
    expect(buildBankConfigApiUpdates({ retainExtractionMode: "verbose" })).toEqual({});
  });
});

describe("normalizers", () => {
  it("validates retain extraction mode", () => {
    expect(normalizeRetainExtractionMode("verbose")).toBe("verbose");
    expect(normalizeRetainExtractionMode("INVALID")).toBeUndefined();
  });

  it("validates disposition traits", () => {
    expect(normalizeDispositionTrait(3)).toBe(3);
    expect(normalizeDispositionTrait(0)).toBeUndefined();
    expect(normalizeDispositionTrait(6)).toBeUndefined();
    expect(normalizeDispositionTrait(2.5)).toBeUndefined();
  });

  it("passes through the server's entity label shapes", () => {
    const list = [{ name: "person", description: "Human" }];
    expect(normalizeEntityLabels(list)).toEqual(list);
    const wrapped = { attributes: list };
    expect(normalizeEntityLabels(wrapped)).toEqual(wrapped);
  });

  it("drops empty or malformed entity label shapes", () => {
    expect(normalizeEntityLabels([])).toBeUndefined();
    expect(normalizeEntityLabels({ attributes: [] })).toBeUndefined();
    // A plain keyed object is not the server's wrapper shape — it would be
    // silently ignored server-side, so it's dropped rather than sent.
    expect(normalizeEntityLabels({ person: { description: "Human" } })).toBeUndefined();
  });
});

describe("applyConfiguredBankDefaults", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    global.fetch = vi.fn();
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("is a no-op when no defaults are configured", async () => {
    const createBank = vi.fn();
    const client = { createBank } as unknown as HindsightClient;

    await applyConfiguredBankDefaults(client, "agent-slack-U123", {}, clientOpts);

    expect(createBank).not.toHaveBeenCalled();
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it("sends createBank and config patch for a derived dynamic bank", async () => {
    const createBank = vi.fn().mockResolvedValue({});
    const client = { createBank } as unknown as HindsightClient;
    vi.mocked(global.fetch).mockResolvedValue(new Response("{}", { status: 200 }));

    const config: PluginConfig = {
      retainExtractionMode: "verbose",
      enableObservations: true,
      enableAutoConsolidation: true,
      entityLabels: ["person", "org"],
      retainMission: "Keep architectural decisions.",
    };

    await applyConfiguredBankDefaults(client, "agent-slack-U123", config, clientOpts);

    expect(createBank).toHaveBeenCalledWith("agent-slack-U123", {
      retainMission: "Keep architectural decisions.",
      retainExtractionMode: "verbose",
      enableObservations: true,
    });
    expect(global.fetch).toHaveBeenCalledWith(
      "http://localhost:8888/v1/default/banks/agent-slack-U123/config",
      expect.objectContaining({
        method: "PATCH",
        body: JSON.stringify({
          updates: {
            entity_labels: ["person", "org"],
            enable_auto_consolidation: true,
          },
        }),
      })
    );
  });

  it("applies missions via createBank without config patch when only missions are set", async () => {
    const createBank = vi.fn().mockResolvedValue({});
    const client = { createBank } as unknown as HindsightClient;

    await applyConfiguredBankDefaults(
      client,
      "openclaw",
      { bankMission: "Reflect.", retainMission: "Retain." },
      clientOpts
    );

    expect(createBank).toHaveBeenCalledWith("openclaw", {
      reflectMission: "Reflect.",
      retainMission: "Retain.",
    });
    expect(global.fetch).not.toHaveBeenCalled();
  });
});

describe("patchBankConfig", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    global.fetch = vi.fn();
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("throws on non-OK responses", async () => {
    vi.mocked(global.fetch).mockResolvedValue(new Response("nope", { status: 422 }));
    await expect(patchBankConfig("bank-a", { entity_labels: ["x"] }, clientOpts)).rejects.toThrow(
      /patchBankConfig failed \(422\)/
    );
  });
});

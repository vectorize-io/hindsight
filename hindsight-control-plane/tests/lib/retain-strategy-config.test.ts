import { describe, expect, it } from "vitest";
import {
  deserializeRetainStrategies,
  serializeRetainStrategies,
  type RetainStrategy,
  type RetainStrategyValues,
} from "../../src/lib/retain-strategy-config";

const parseEntityLabels = (raw: unknown) => (Array.isArray(raw) ? raw : null);
const baseValues = (overrides: Partial<RetainStrategyValues> = {}): RetainStrategyValues => ({
  retain_extraction_mode: null,
  retain_chunk_size: 3000,
  retain_structured_chunk_size: null,
  retain_mission: null,
  retain_custom_instructions: null,
  entities_allow_free_form: null,
  entity_labels: null,
  ...overrides,
});

describe("retain strategy config serialization", () => {
  it("omits null structured-chunk cap overrides", () => {
    const strategies = deserializeRetainStrategies(
      {
        jsonl: {
          retain_chunk_size: 8000,
          retain_structured_chunk_size: null,
        },
      },
      parseEntityLabels
    );

    expect(strategies[0]?.values.retain_structured_chunk_size).toBeNull();
    expect(serializeRetainStrategies(strategies)).toEqual({
      jsonl: {
        retain_chunk_size: 8000,
      },
    });
  });

  it("omits structured-chunk cap when the strategy inherits it", () => {
    const strategies = deserializeRetainStrategies(
      {
        inherited: {
          retain_chunk_size: 4000,
        },
      },
      parseEntityLabels
    );

    expect(strategies[0]?.values.retain_structured_chunk_size).toBeNull();
    expect(serializeRetainStrategies(strategies)).toEqual({
      inherited: {
        retain_chunk_size: 4000,
      },
    });
  });

  it("emits numeric structured-chunk cap overrides independently of chunk size", () => {
    const strategies: RetainStrategy[] = [
      {
        id: 1,
        name: "wide-jsonl",
        values: {
          retain_extraction_mode: null,
          retain_chunk_size: 4000,
          retain_structured_chunk_size: 2000,
          retain_mission: null,
          retain_custom_instructions: null,
          entities_allow_free_form: null,
          entity_labels: null,
        },
      },
    ];

    expect(serializeRetainStrategies(strategies)).toEqual({
      "wide-jsonl": {
        retain_chunk_size: 4000,
        retain_structured_chunk_size: 2000,
      },
    });
  });
});

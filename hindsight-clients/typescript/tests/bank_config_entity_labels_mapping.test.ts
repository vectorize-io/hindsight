/**
 * Unit tests for the entity-labels request mapping on updateBankConfig.
 *
 * Like the other *_mapping tests these do NOT require a running server: the
 * generated sdk layer is mocked so we can assert the camelCase options land on
 * the snake_case request body.
 *
 * updateBankConfig enumerates config fields rather than passing a dict through,
 * so a field the wrapper doesn't know about is silently dropped for every
 * consumer of the SDK — with no type error to catch it. `entityLabels` was
 * exactly that gap: the option did not exist, so TypeScript callers could not
 * configure a controlled vocabulary at all. The Python wrapper has the mirror of
 * this file in tests/test_bank_config_update_payload.py; the two must stay in step.
 */

import { HindsightClient } from "../src";
import * as sdk from "../generated/sdk.gen";
import type { LabelGroupInput } from "../generated/types.gen";

jest.mock("../generated/sdk.gen");

const mockedUpdateConfig = sdk.updateBankConfig as jest.MockedFunction<typeof sdk.updateBankConfig>;

describe("entity labels mapping", () => {
  let client: HindsightClient;

  beforeEach(() => {
    client = new HindsightClient({ baseUrl: "http://localhost:8888" });
    mockedUpdateConfig.mockReset();
    mockedUpdateConfig.mockResolvedValue({
      data: { bank_id: "bank", config: {}, overrides: {} },
    } as any);
  });

  test("updateBankConfig maps entity labels onto the updates map", async () => {
    const entityLabels: LabelGroupInput[] = [
      {
        key: "name",
        type: "multi-text",
        tag: true,
        description: "Every name the subject of this fact is known by.",
      },
      { key: "topic", type: "value", values: [{ value: "infra" }] },
    ];

    await client.updateBankConfig("bank", { entityLabels, entitiesAllowFreeForm: false });

    const body = mockedUpdateConfig.mock.calls[0][0].body as any;
    expect(body.updates).toEqual({
      entity_labels: entityLabels,
      entities_allow_free_form: false,
    });
  });

  test("updateBankConfig omits entity labels when unset so the vocabulary survives", async () => {
    await client.updateBankConfig("bank", { retainChunkSize: 1000 });

    const body = mockedUpdateConfig.mock.calls[0][0].body as any;
    expect(body.updates.entity_labels).toBeUndefined();
    expect(body.updates.entities_allow_free_form).toBeUndefined();
  });
});

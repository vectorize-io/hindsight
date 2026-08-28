/**
 * Unit tests for the recall pipeline toggles' request mapping.
 *
 * Like the other *_mapping tests these do NOT require a running server: the
 * generated sdk layer is mocked so we can assert the camelCase options land on
 * the snake_case request body.
 *
 * Both wrapper methods enumerate config fields rather than passing a dict
 * through, so a toggle added to the API but not to the wrapper is silently
 * dropped for every consumer of the SDK — with no type error to catch it. The
 * Python wrapper has the mirror of this file in
 * tests/test_bank_config_update_payload.py; the two must stay in step.
 */

import { HindsightClient } from "../src";
import * as sdk from "../generated/sdk.gen";

jest.mock("../generated/sdk.gen");

const mockedCreateBank = sdk.createOrUpdateBank as jest.MockedFunction<
  typeof sdk.createOrUpdateBank
>;
const mockedUpdateConfig = sdk.updateBankConfig as jest.MockedFunction<typeof sdk.updateBankConfig>;

describe("recall pipeline toggle mapping", () => {
  let client: HindsightClient;

  beforeEach(() => {
    client = new HindsightClient({ baseUrl: "http://localhost:8888" });
    mockedCreateBank.mockReset();
    mockedCreateBank.mockResolvedValue({
      data: { bank_id: "bank", name: "bank" },
    } as any);
    mockedUpdateConfig.mockReset();
    mockedUpdateConfig.mockResolvedValue({
      data: { bank_id: "bank", config: {}, overrides: {} },
    } as any);
  });

  test("createBank maps every toggle onto the body", async () => {
    await client.createBank("bank", {
      enableTextSearch: false,
      enableTemporalRetrieval: false,
      enableGraphRetrieval: false,
      enableReranking: false,
    });

    const body = mockedCreateBank.mock.calls[0][0].body as any;
    expect(body.enable_text_search).toBe(false);
    expect(body.enable_temporal_retrieval).toBe(false);
    expect(body.enable_graph_retrieval).toBe(false);
    expect(body.enable_reranking).toBe(false);
  });

  test("createBank omits unset toggles so the bank inherits the server default", async () => {
    await client.createBank("bank", {});

    const body = mockedCreateBank.mock.calls[0][0].body as any;
    expect(body.enable_text_search).toBeUndefined();
    expect(body.enable_temporal_retrieval).toBeUndefined();
    expect(body.enable_graph_retrieval).toBeUndefined();
    expect(body.enable_reranking).toBeUndefined();
  });

  test("updateBankConfig maps every toggle onto the updates map", async () => {
    await client.updateBankConfig("bank", {
      enableTextSearch: false,
      enableTemporalRetrieval: false,
      enableGraphRetrieval: false,
      enableReranking: false,
    });

    const updates = mockedUpdateConfig.mock.calls[0][0].body as any;
    expect(updates.updates).toEqual({
      enable_text_search: false,
      enable_temporal_retrieval: false,
      enable_graph_retrieval: false,
      enable_reranking: false,
    });
  });

  test("updateBankConfig sends only the toggles that were set", async () => {
    await client.updateBankConfig("bank", { enableTextSearch: false });

    const updates = mockedUpdateConfig.mock.calls[0][0].body as any;
    expect(updates.updates).toEqual({ enable_text_search: false });
  });
});

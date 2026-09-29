import { HindsightClient } from "../src";
import * as sdk from "../generated/sdk.gen";

function makeClient(): HindsightClient {
  return new HindsightClient({ baseUrl: "http://localhost:8888" });
}

describe("getVersion", () => {
  let spy: jest.SpyInstance;

  beforeEach(() => {
    spy = jest.spyOn(sdk, "getVersion").mockResolvedValue({
      data: {
        api_version: "0.8.2",
        features: {
          observations: true,
          mcp: true,
          document_upload: true,
          bank_config: true,
          directives: true,
          metrics: true,
          custom_llm_provider: true,
          bank_llm_health: true,
          file_conversion: true,
        },
      },
    } as any);
  });

  afterEach(() => {
    spy.mockRestore();
  });

  test("calls the version endpoint through the generated client", async () => {
    const version = await makeClient().getVersion();

    expect(version.api_version).toBe("0.8.2");
    expect(version.features.observations).toBe(true);
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy.mock.calls[0][0]).toHaveProperty("client");
  });
});

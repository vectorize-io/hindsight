/**
 * Tests that reflect() forwards include flags into the ReflectRequest body.
 *
 * The wrapper previously never sent `include`, so the reflect trace
 * (trace.tool_calls / trace.llm_calls) and based_on facts were unreachable
 * from the convenience layer. These tests pin the option -> request.include
 * mapping. No server required: sdk.reflect is mocked.
 */

import { HindsightClient } from "../src";
import * as sdk from "../generated/sdk.gen";

function makeClient(): HindsightClient {
  return new HindsightClient({ baseUrl: "http://localhost:8888" });
}

function capturedInclude(spy: jest.SpyInstance): any {
  return (spy.mock.calls[0][0] as any).body.include;
}

describe("reflect include options", () => {
  let spy: jest.SpyInstance;

  beforeEach(() => {
    spy = jest.spyOn(sdk, "reflect").mockResolvedValue({ data: { text: "ok" } } as any);
  });

  afterEach(() => {
    spy.mockRestore();
  });

  test("no include flags omits include", async () => {
    await makeClient().reflect("bank", "query");
    expect(capturedInclude(spy)).toBeUndefined();
  });

  test("includeToolCalls sets tool_calls with output on by default", async () => {
    await makeClient().reflect("bank", "query", { includeToolCalls: true });
    const include = capturedInclude(spy);
    expect(include).toBeDefined();
    expect(include.tool_calls).toEqual({ output: true });
    expect(include.facts).toBeUndefined();
  });

  test("includeToolCallOutput false requests inputs-only trace", async () => {
    await makeClient().reflect("bank", "query", {
      includeToolCalls: true,
      includeToolCallOutput: false,
    });
    expect(capturedInclude(spy).tool_calls).toEqual({ output: false });
  });

  test("includeFacts sets facts", async () => {
    await makeClient().reflect("bank", "query", { includeFacts: true });
    const include = capturedInclude(spy);
    expect(include.facts).toEqual({});
    expect(include.tool_calls).toBeUndefined();
  });

  test("includeFacts and includeToolCalls combine", async () => {
    await makeClient().reflect("bank", "query", {
      includeFacts: true,
      includeToolCalls: true,
    });
    const include = capturedInclude(spy);
    expect(include.facts).toEqual({});
    expect(include.tool_calls).toEqual({ output: true });
  });

  test("includeToolCallOutput is a no-op without includeToolCalls", async () => {
    await makeClient().reflect("bank", "query", { includeToolCallOutput: false });
    expect(capturedInclude(spy)).toBeUndefined();
  });
});

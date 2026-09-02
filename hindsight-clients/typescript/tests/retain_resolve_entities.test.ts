/**
 * Tests that retain()/retainBatch() forward resolve_entities into the request body.
 *
 * The wrapper builds each item field by field, so a new field is dropped unless it is
 * added here too — the gap #2975/#3042 closed for the mental-model methods. Dropping this
 * one would silently restore the entity substitution the flag exists to prevent (#3479).
 * No server required: sdk.retainMemories is mocked.
 */

import { HindsightClient } from "../src";
import * as sdk from "../generated/sdk.gen";

function makeClient(): HindsightClient {
  return new HindsightClient({ baseUrl: "http://localhost:8888" });
}

function capturedItems(spy: jest.SpyInstance): any[] {
  return (spy.mock.calls[0][0] as any).body.items;
}

describe("retain resolve_entities", () => {
  let spy: jest.SpyInstance;

  beforeEach(() => {
    spy = jest.spyOn(sdk, "retainMemories").mockResolvedValue({ data: { success: true } } as any);
  });

  afterEach(() => {
    spy.mockRestore();
  });

  test("retain forwards resolveEntities false", async () => {
    await makeClient().retain("bank", "The patient saw a specialist.", {
      entities: [{ text: "Dr. Waller", type: "PERSON" }],
      resolveEntities: false,
    });
    expect(capturedItems(spy)[0].resolve_entities).toBe(false);
  });

  test("omitting resolveEntities leaves the server default in place", async () => {
    await makeClient().retain("bank", "A fact.", { entities: [{ text: "Alice" }] });
    expect(capturedItems(spy)[0].resolve_entities).toBeUndefined();
  });

  test("retainBatch passes a per-item resolve_entities through", async () => {
    await makeClient().retainBatch("bank", [
      { content: "A fact.", entities: [{ text: "Alice" }], resolve_entities: false },
    ]);
    expect(capturedItems(spy)[0].resolve_entities).toBe(false);
  });
});

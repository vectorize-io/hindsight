/**
 * The maintained wrapper forwards multimodal content blocks unaltered.
 *
 * `content` widened from `string` to `string | ContentBlock[]` so an image can
 * sit inline where it appears. The wrapper is what most consumers call, so a
 * wrapper that coerced or dropped the list form would strip images for every
 * TypeScript user while the generated SDK happily supported them.
 *
 * The Python wrapper has the mirror of this file in
 * tests/test_retain_content_blocks.py; the two must stay in step.
 */

import { HindsightClient, type ContentBlock } from "../src";
import * as sdk from "../generated/sdk.gen";

jest.mock("../generated/sdk.gen");

const mockedRetain = sdk.retainMemories as jest.MockedFunction<typeof sdk.retainMemories>;

const IMAGE_BLOCK: ContentBlock = {
  type: "image",
  source: { type: "base64", media_type: "image/png", data: "aGVsbG8=" },
};

describe("retain content blocks", () => {
  let client: HindsightClient;

  beforeEach(() => {
    client = new HindsightClient({ baseUrl: "http://localhost:8888" });
    mockedRetain.mockReset();
    mockedRetain.mockResolvedValue({ data: { success: true } } as never);
  });

  const sentItems = () => (mockedRetain.mock.calls[0][0] as any).body.items;

  it("forwards a block list verbatim", async () => {
    const content: ContentBlock[] = [
      { type: "text", text: "click the button shown:" },
      IMAGE_BLOCK,
      { type: "text", text: "...then reconnect." },
    ];

    await client.retain("test-bank", content);

    expect(sentItems()[0].content).toEqual(content);
  });

  it("still forwards a plain string", async () => {
    // Widening the type must not disturb every existing caller.
    await client.retain("test-bank", "Alice joined the AI team");

    expect(sentItems()[0].content).toBe("Alice joined the AI team");
  });

  it("forwards block lists through retainBatch", async () => {
    await client.retainBatch("test-bank", [
      { content: [{ type: "text", text: "see:" }, IMAGE_BLOCK] },
      { content: "plain text item" },
    ]);

    expect(sentItems()[0].content).toEqual([{ type: "text", text: "see:" }, IMAGE_BLOCK]);
    expect(sentItems()[1].content).toBe("plain text item");
  });
});

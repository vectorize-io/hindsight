import { beforeEach, describe, expect, it, vi } from "vitest";
import { createHindsightProvider } from "../src/index.js";
import type { HindsightClient } from "../src/index.js";
import { USER_ID, mockClient, runtime, userMessage } from "./helpers.js";

describe("createHindsightProvider", () => {
  let client: HindsightClient;

  beforeEach(() => {
    client = mockClient();
  });

  it("exposes the elizaOS provider contract", () => {
    const provider = createHindsightProvider(client, undefined);
    expect(provider.name).toBe("HINDSIGHT_MEMORY");
    expect(provider.dynamic).toBe(false);
    expect(typeof provider.description).toBe("string");
  });

  it("renders the default heading above recalled memories", async () => {
    const provider = createHindsightProvider(client, undefined);
    const result = await provider.get(runtime, userMessage("q"), {} as never);
    expect(result.text.startsWith("# Relevant long-term memories\n")).toBe(true);
    expect(result.text).toContain("- User prefers dark mode");
    expect(result.text).toContain("- User lives in Berlin");
  });

  it("honours a custom heading", async () => {
    const provider = createHindsightProvider(client, undefined, { heading: "## Memory" });
    const result = await provider.get(runtime, userMessage("q"), {} as never);
    expect(result.text.startsWith("## Memory\n")).toBe(true);
  });

  it("returns empty text (no heading) when there are no results", async () => {
    const empty = mockClient({ results: [] });
    const provider = createHindsightProvider(empty, undefined);
    const result = await provider.get(runtime, userMessage("q"), {} as never);
    expect(result.text).toBe("");
    expect(result.values?.hindsightMemoryCount).toBe(0);
  });

  it("filters out results whose text is empty or whitespace", async () => {
    const noisy = mockClient({
      results: [
        { id: "1", text: "keep me" },
        { id: "2", text: "   " },
        { id: "3", text: "" },
        { id: "4", text: "keep me too" },
      ],
    });
    const provider = createHindsightProvider(noisy, undefined);
    const result = await provider.get(runtime, userMessage("q"), {} as never);
    expect(result.text).toBe("# Relevant long-term memories\n- keep me\n- keep me too");
    // The count reflects the raw recall size, not the rendered lines.
    expect(result.values?.hindsightMemoryCount).toBe(4);
  });

  it("trims whitespace around each memory line", async () => {
    const padded = mockClient({ results: [{ id: "1", text: "  spaced out  " }] });
    const provider = createHindsightProvider(padded, undefined);
    const result = await provider.get(runtime, userMessage("q"), {} as never);
    expect(result.text).toBe("# Relevant long-term memories\n- spaced out");
  });

  it("skips recall and returns empty text when the message has no content text", async () => {
    const provider = createHindsightProvider(client, undefined);
    const result = await provider.get(runtime, userMessage(undefined), {} as never);
    expect(client.recall).not.toHaveBeenCalled();
    expect(result.text).toBe("");
  });

  it("resolves the bank via a resolver function", async () => {
    const provider = createHindsightProvider(client, (m) => `room:${m.roomId}`);
    const message = userMessage("hi");
    await provider.get(runtime, message, {} as never);
    expect(client.recall).toHaveBeenCalledWith(`room:${message.roomId}`, "hi", expect.any(Object));
  });

  it("carries the full recall response through on data.hindsight", async () => {
    const provider = createHindsightProvider(client, undefined);
    const result = await provider.get(runtime, userMessage("q"), {} as never);
    expect(result.data?.hindsight).toEqual({
      results: [
        { id: "1", text: "User prefers dark mode" },
        { id: "2", text: "User lives in Berlin" },
      ],
    });
  });

  it("swallows recall errors and reports a zero count with the error message", async () => {
    (client.recall as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new Error("upstream 503"));
    const provider = createHindsightProvider(client, undefined);
    const result = await provider.get(runtime, userMessage("q"), {} as never);
    expect(result.text).toBe("");
    expect(result.values?.hindsightMemoryCount).toBe(0);
    expect(result.data?.hindsightError).toBe("upstream 503");
  });

  it("stringifies non-Error recall failures", async () => {
    (client.recall as ReturnType<typeof vi.fn>).mockRejectedValueOnce("plain string failure");
    const provider = createHindsightProvider(client, undefined);
    const result = await provider.get(runtime, userMessage("q"), {} as never);
    expect(result.data?.hindsightError).toBe("plain string failure");
  });

  it("uses the message entityId as the default bank", async () => {
    const provider = createHindsightProvider(client, undefined);
    await provider.get(runtime, userMessage("hi"), {} as never);
    expect(client.recall).toHaveBeenCalledWith(USER_ID, "hi", expect.any(Object));
  });
});

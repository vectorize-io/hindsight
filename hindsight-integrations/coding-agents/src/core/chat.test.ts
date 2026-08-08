import { describe, expect, it, vi } from "vitest";
import type { HindsightClient } from "./hindsight";
import { renderSessionJsonl, retainLiveSession, type TransportTurn } from "./chat";

describe("renderSessionJsonl", () => {
  const turns: TransportTurn[] = [
    { role: "user", content: "Add retry backoff", timestamp: "2026-01-01T00:00:00Z" },
    { role: "assistant", content: "On it.", timestamp: "2026-01-01T00:00:01Z" },
    { role: "action", content: "Edit uploader.ts" },
  ];

  it("renders JSONL (one JSON object per line) led by the REF-ID system turn, preserving roles/content/timestamps", () => {
    const jsonl = renderSessionJsonl("conversation:s1", turns, "2026-01-01T00:00:00Z");
    const parsed = jsonl.split("\n").map((line) => JSON.parse(line) as TransportTurn);
    expect(parsed).toHaveLength(4);
    expect(parsed[0]).toEqual({
      role: "system",
      content: "REF-ID: conversation:s1",
      timestamp: "2026-01-01T00:00:00Z",
    });
    expect(parsed[1]).toEqual({
      role: "user",
      content: "Add retry backoff",
      timestamp: "2026-01-01T00:00:00Z",
    });
    expect(parsed[2]).toEqual({
      role: "assistant",
      content: "On it.",
      timestamp: "2026-01-01T00:00:01Z",
    });
    // Compact action turns pass through untouched (no timestamp -> none serialized).
    expect(parsed[3]).toEqual({ role: "action", content: "Edit uploader.ts" });
  });

  it("empty turn list still yields the REF-ID system turn alone (exactly one line)", () => {
    const lines = renderSessionJsonl("r", [], "2026-01-01T00:00:00Z").split("\n");
    expect(lines).toHaveLength(1);
    expect(JSON.parse(lines[0]) as TransportTurn).toEqual({
      role: "system",
      content: "REF-ID: r",
      timestamp: "2026-01-01T00:00:00Z",
    });
  });
});

describe("retainLiveSession", () => {
  it("upserts the JSONL transcript under conversation:<id> with the unified conversation strategy", async () => {
    const retain = vi.fn().mockResolvedValue(undefined);
    const client = { retain } as unknown as HindsightClient;
    const turns: TransportTurn[] = [
      { role: "user", content: "hi", timestamp: "2026-01-01T00:00:00Z" },
    ];

    await retainLiveSession(client, "s2", turns, "2026-01-01T00:00:00Z");

    expect(retain).toHaveBeenCalledTimes(1);
    const [content, context, documentId, tags, strategy, opts] = retain.mock.calls[0];
    // The retained content IS the renderSessionJsonl transcript.
    expect(content).toBe(renderSessionJsonl("conversation:s2", turns, "2026-01-01T00:00:00Z"));
    const parsed = (content as string).split("\n").map((line) => JSON.parse(line) as TransportTurn);
    expect(parsed[0]).toEqual({
      role: "system",
      content: "REF-ID: conversation:s2",
      timestamp: "2026-01-01T00:00:00Z",
    });
    expect(parsed[1]).toEqual({ role: "user", content: "hi", timestamp: "2026-01-01T00:00:00Z" });
    expect(context).toBe("coding agent session");
    expect(documentId).toBe("conversation:s2");
    expect(tags).toEqual(["source:chat"]);
    expect(strategy).toBe("conversation");
    expect(opts).toMatchObject({
      async: true,
      idempotent: true,
      timestamp: "2026-01-01T00:00:00Z",
    });
    expect(opts.metadata).toMatchObject({
      source: "chat",
      session_id: "s2",
      ref_id: "conversation:s2",
    });
  });
});

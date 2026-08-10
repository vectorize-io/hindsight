import { describe, expect, it, vi } from "vitest";
import type { HindsightClient } from "./hindsight";
import { renderSessionJsonl, retainLiveSession, type TransportTurn } from "./chat";
import { memoryCursorStore, type RetainCursorStore } from "./retain-cursor";

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
    expect(opts).toMatchObject({ timestamp: "2026-01-01T00:00:00Z" });
    expect(opts.metadata).toMatchObject({
      source: "chat",
      session_id: "s2",
      ref_id: "conversation:s2",
    });
  });
});

describe("retainLiveSession — incremental write-back", () => {
  const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
  const turn = (i: number): TransportTurn => ({ role: "user", content: `turn ${i}` });
  const turns = (n: number) => Array.from({ length: n }, (_, i) => turn(i));

  /** Client double: `supported` is what GET /version would have told us about operation_id. */
  const stubClient = (supported = true) => {
    const retain = vi.fn().mockResolvedValue(undefined);
    return {
      retain,
      client: {
        retain,
        bank: "coding-agent::repo",
        supportsIdempotentRetain: async () => supported,
      } as unknown as HindsightClient,
    };
  };

  const write = (client: HindsightClient, turnList: TransportTurn[], cursors: RetainCursorStore) =>
    retainLiveSession(client, "s1", turnList, "2026-01-01T00:00:00Z", "codex", { cursors });

  it("replaces on the first write, then appends only the new turns", async () => {
    const { retain, client } = stubClient();
    const cursors = memoryCursorStore();

    await write(client, turns(2), cursors);
    const first = retain.mock.calls[0];
    expect(first[5].updateMode).toBeUndefined();
    expect(first[0]).toBe(renderSessionJsonl("conversation:s1", turns(2), "2026-01-01T00:00:00Z"));

    await write(client, turns(5), cursors);
    const second = retain.mock.calls[1];
    expect(second[5].updateMode).toBe("append");
    // Only the three new turns, and no REF-ID header: the document already carries one.
    expect((second[0] as string).split("\n").map((l) => JSON.parse(l) as TransportTurn)).toEqual([
      turn(2),
      turn(3),
      turn(4),
    ]);
    expect(second[2]).toBe("conversation:s1"); // same document id — append targets it
  });

  it("sends a stable v5 operation_id so a resubmitted write is not applied twice", async () => {
    const a = stubClient();
    const b = stubClient();
    await write(a.client, turns(3), memoryCursorStore());
    await write(b.client, turns(3), memoryCursorStore());
    const opId = a.retain.mock.calls[0][5].operationId;
    expect(opId).toMatch(UUID_RE);
    expect(b.retain.mock.calls[0][5].operationId).toBe(opId);
  });

  it("gives a different operation_id to a different payload", async () => {
    const { retain, client } = stubClient();
    const cursors = memoryCursorStore();
    await write(client, turns(2), cursors);
    await write(client, turns(5), cursors);
    expect(retain.mock.calls[0][5].operationId).not.toBe(retain.mock.calls[1][5].operationId);
  });

  it("skips the write entirely when no turn was added", async () => {
    const { retain, client } = stubClient();
    const cursors = memoryCursorStore();
    await write(client, turns(3), cursors);
    await write(client, turns(3), cursors);
    expect(retain).toHaveBeenCalledTimes(1);
  });

  it("replaces the whole document after a failed write, instead of appending onto an unknown state", async () => {
    const { retain, client } = stubClient();
    const cursors = memoryCursorStore();
    await write(client, turns(2), cursors);

    retain.mockRejectedValueOnce(new Error("timeout"));
    await expect(write(client, turns(4), cursors)).rejects.toThrow("timeout");
    expect(cursors.read("s1")?.dirty).toBe(true);

    await write(client, turns(6), cursors);
    const recovery = retain.mock.calls[2];
    expect(recovery[5].updateMode).toBeUndefined();
    expect(recovery[0]).toBe(
      renderSessionJsonl("conversation:s1", turns(6), "2026-01-01T00:00:00Z")
    );
    expect(cursors.read("s1")).toEqual({
      turns: 6,
      fingerprint: expect.any(String),
      bank: "coding-agent::repo",
      appends: 0, // the recovery replace re-established the document, restarting the re-sync count
    });
  });

  it("never appends against a server that ignores operation_id", async () => {
    const { retain, client } = stubClient(false);
    const cursors = memoryCursorStore();
    await write(client, turns(2), cursors);
    await write(client, turns(5), cursors);
    expect(retain.mock.calls.map((c) => c[5].updateMode)).toEqual([undefined, undefined]);
    expect(retain.mock.calls[1][0]).toBe(
      renderSessionJsonl("conversation:s1", turns(5), "2026-01-01T00:00:00Z")
    );
  });

  it("replaces (never appends) when no cursor store is supplied", async () => {
    const { retain, client } = stubClient();
    await retainLiveSession(client, "s1", turns(2), "2026-01-01T00:00:00Z", "codex");
    await retainLiveSession(client, "s1", turns(5), "2026-01-01T00:00:00Z", "codex");
    expect(retain.mock.calls.map((c) => c[5].updateMode)).toEqual([undefined, undefined]);
  });

  it("serialises overlapping write-backs so neither appends a slice the other already sent", async () => {
    // The runtime fires retains without awaiting them: a turn-driven one and an idle-driven one can
    // overlap. Unserialised, both planned an append from the same cursor position and submitted
    // overlapping slices, duplicating turns inside the document.
    const submitted: { mode: string; turns: number }[] = [];
    const gates: (() => void)[] = [];
    const retain = vi.fn((content: string, ...rest: unknown[]) => {
      const o = rest[4] as { updateMode?: string };
      submitted.push({ mode: o.updateMode ?? "replace", turns: content.split("\n").length });
      return new Promise<void>((resolve) => gates.push(resolve));
    });
    const client = {
      retain,
      bank: "b",
      supportsIdempotentRetain: async () => true,
    } as unknown as HindsightClient;
    const cursors = memoryCursorStore();

    const first = write(client, turns(5), cursors);
    await vi.waitFor(() => expect(gates).toHaveLength(1));
    gates[0]();
    await first;

    const a = write(client, turns(8), cursors);
    const b = write(client, turns(9), cursors);
    // Only ONE request is in flight: the second write-back waits for the first to be confirmed.
    await vi.waitFor(() => expect(gates).toHaveLength(2));
    expect(submitted).toHaveLength(2);
    gates[1]();
    await a;
    await vi.waitFor(() => expect(gates).toHaveLength(3));
    gates[2]();
    await b;

    // 6 = REF-ID + 5 turns, then turns 5-7, then turn 8 alone — every turn sent exactly once.
    expect(submitted).toEqual([
      { mode: "replace", turns: 6 },
      { mode: "append", turns: 3 },
      { mode: "append", turns: 1 },
    ]);
  });

  it("keeps two sessions in the same directory independent", async () => {
    // Same repo => same bank, but each session owns its own document and its own cursor, so two
    // agents running side by side in one checkout never append into each other's conversation.
    const { retain, client } = stubClient();
    const cursors = memoryCursorStore();
    const writeAs = (id: string, list: TransportTurn[]) =>
      retainLiveSession(client, id, list, "2026-01-01T00:00:00Z", "codex", { cursors });

    await writeAs("sess-a", turns(2));
    await writeAs("sess-b", turns(4));
    await writeAs("sess-a", turns(3));

    const docs = retain.mock.calls.map((c) => c[2]);
    expect(docs).toEqual(["conversation:sess-a", "conversation:sess-b", "conversation:sess-a"]);
    // sess-b's longer transcript did not advance sess-a's cursor: its append is turn 2 alone.
    expect(retain.mock.calls[1][5].updateMode).toBeUndefined(); // first write for sess-b
    expect(retain.mock.calls[2][5].updateMode).toBe("append");
    expect((retain.mock.calls[2][0] as string).split("\n")).toHaveLength(1);
    expect(cursors.read("sess-a")?.turns).toBe(3);
    expect(cursors.read("sess-b")?.turns).toBe(4);
  });

  it("replaces rather than appends when the session moved to another bank", async () => {
    const sent: { bank: string; mode: string }[] = [];
    const mk = (bank: string) =>
      ({
        bank,
        supportsIdempotentRetain: async () => true,
        retain: vi.fn(async (_c: string, ...rest: unknown[]) => {
          sent.push({ bank, mode: (rest[4] as { updateMode?: string }).updateMode ?? "replace" });
        }),
      }) as unknown as HindsightClient;
    const cursors = memoryCursorStore();

    await write(mk("repo-a"), turns(5), cursors);
    await write(mk("repo-b"), turns(8), cursors); // user cd'd into another repo mid-session
    expect(sent).toEqual([
      { bank: "repo-a", mode: "replace" },
      { bank: "repo-b", mode: "replace" },
    ]);
  });

  it("keeps the write-back when the capability probe itself fails", async () => {
    const retain = vi.fn().mockResolvedValue(undefined);
    const client = {
      retain,
      bank: "b",
      supportsIdempotentRetain: async () => {
        throw new Error("unreachable");
      },
    } as unknown as HindsightClient;
    await write(client, turns(2), memoryCursorStore());
    expect(retain).toHaveBeenCalledTimes(1);
    expect(retain.mock.calls[0][5].updateMode).toBeUndefined();
  });
});

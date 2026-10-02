import { describe, expect, it, vi } from "vitest";
import { opencodeAdapter } from "./opencode";
import { RuntimeCore } from "../core/runtime";
import { resolveConfig } from "../core/config";
import type { HindsightClient } from "../core/hindsight";
import type { OcMessage } from "../core/transcript-opencode";

function message(sessionID: string, role: string, text: string): OcMessage {
  return { info: { sessionID, role }, parts: [{ type: "text", text }] };
}

function transformFor(core: RuntimeCore) {
  const hooks = opencodeAdapter.createRuntime(core) as {
    "experimental.chat.messages.transform": (
      input: unknown,
      output: { messages: OcMessage[] }
    ) => Promise<void>;
  };
  return hooks["experimental.chat.messages.transform"];
}

function recordingCore(enabled = true) {
  const onTranscript = vi.fn(async () => {});
  const core = {
    harness: "opencode",
    toolSpecs: () => [],
    writeBackEnabled: enabled,
    onTranscript,
  } as unknown as RuntimeCore;
  return { transform: transformFor(core), onTranscript };
}

describe("OpenCode side-session retention isolation", () => {
  it("does not replace a parent document with a side conversation after a parent-context prefix", async () => {
    const documents = new Map<string, string>();
    const client = {
      retain: vi.fn(async (text: string, _context: string, documentId: string) => {
        documents.set(documentId, text);
      }),
    } as unknown as HindsightClient;
    const core = new RuntimeCore(client, "test-bank", resolveConfig({}), "opencode");
    const transform = transformFor(core);
    const parent = [
      message("parent", "user", "Original parent request"),
      message("parent", "assistant", "Original parent conclusion"),
    ];
    await transform({}, { messages: parent });
    await vi.waitFor(() => expect(documents.has("conversation:parent")).toBe(true));
    const original = documents.get("conversation:parent");
    // OmO prepends bounded parent context without changing its original session IDs.
    // See oh-my-openagent/packages/omo-opencode/src/features/btw-side/context-injector.ts.
    const mixed = [parent[0], message("side", "user", "Independent side question")];
    const before = structuredClone(mixed);
    await transform({}, { messages: mixed });
    await vi.waitFor(() => expect(client.retain).toHaveBeenCalledTimes(2));
    expect(documents.get("conversation:parent")).toBe(original);
    expect(documents.get("conversation:side")).toContain("Independent side question");
    expect(documents.get("conversation:side")).not.toContain("Original parent request");
    expect(mixed).toEqual(before);
  });

  it("keeps a normal session's complete transcript", async () => {
    const { transform, onTranscript } = recordingCore();
    await transform(
      {},
      {
        messages: [message("normal", "user", "Question"), message("normal", "assistant", "Answer")],
      }
    );
    expect(onTranscript).toHaveBeenCalledExactlyOnceWith(
      "normal",
      [
        { role: "user", content: "Question" },
        { role: "assistant", content: "Answer" },
      ],
      false
    );
  });

  it("does not merge nested ancestors into the current side's retained transcript", async () => {
    const { transform, onTranscript } = recordingCore();
    await transform(
      {},
      {
        messages: [
          message("root", "user", "Root"),
          message("parent", "assistant", "Parent"),
          message("child", "user", "Child"),
          message("child", "assistant", "Child reply"),
        ],
      }
    );
    expect(onTranscript).toHaveBeenCalledExactlyOnceWith(
      "child",
      [
        { role: "user", content: "Child" },
        { role: "assistant", content: "Child reply" },
      ],
      false
    );
  });

  it("does not retain unidentifiable or empty message lists", async () => {
    const { transform, onTranscript } = recordingCore();
    for (const messages of [
      [],
      [{ info: { role: "user" }, parts: [{ type: "text", text: "Unknown" }] }],
    ]) {
      await transform({}, { messages });
    }
    expect(onTranscript).not.toHaveBeenCalled();
  });

  it("still honors disabled session retention", async () => {
    const { transform, onTranscript } = recordingCore(false);
    await transform({}, { messages: [message("side", "user", "Private")] });
    expect(onTranscript).not.toHaveBeenCalled();
  });
});

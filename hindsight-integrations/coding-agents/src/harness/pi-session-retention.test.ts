import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPiExtension } from "./pi-extension";

const { onTranscript } = vi.hoisted(() => ({
  onTranscript: vi.fn(async (_session: string, _turns: unknown[], _complete: boolean) => {}),
}));
vi.mock("../core/host-client", () => ({
  resolveHostMemory: () => ({ cfg: {}, bankId: "test", client: {} }),
}));
vi.mock("../core/runtime", () => ({
  RuntimeCore: class {
    toolSpecs() {
      return [];
    }
    seedIfCold() {
      return Promise.resolve();
    }
    onTranscript = onTranscript;
  },
}));

const user = (text: string) => ({ type: "message", message: { role: "user", content: text } });
const assistant = (text: string) => ({
  type: "message",
  message: { role: "assistant", content: text },
});

function adapter() {
  const handlers: Record<string, Function> = {};
  createPiExtension("pi")({
    on: ((name: string, handler: Function) => {
      handlers[name] = handler;
    }) as never,
    registerTool: vi.fn(),
  });
  return handlers.agent_end;
}

beforeEach(() => onTranscript.mockClear());

describe("Pi session retention", () => {
  it("retains both runs when the second event contains only its latest run", async () => {
    const end = adapter();
    let branch = [user("Alpha"), assistant("173 items")];
    const ctx = { sessionManager: { getSessionId: () => "session", getBranch: () => branch } };
    await end({ messages: branch.map((entry) => entry.message) }, ctx);
    branch = [...branch, user("Beta"), assistant("47 seconds")];
    await end({ messages: branch.slice(2).map((entry) => entry.message) }, ctx);
    expect(onTranscript.mock.calls[1]).toEqual([
      "session",
      [
        { role: "user", content: "Alpha" },
        { role: "assistant", content: "173 items" },
        { role: "user", content: "Beta" },
        { role: "assistant", content: "47 seconds" },
      ],
      true,
    ]);
  });

  it("includes earlier messages after reopening a session", async () => {
    await adapter()(
      { messages: [assistant("new").message] },
      {
        sessionManager: {
          getSessionId: () => "resumed",
          getBranch: () => [user("old"), assistant("new")],
        },
      }
    );
    expect(onTranscript.mock.calls[0][1]).toEqual([
      { role: "user", content: "old" },
      { role: "assistant", content: "new" },
    ]);
  });

  it("keeps pre-compaction messages and ignores non-message entries", async () => {
    await adapter()(
      { messages: [assistant("after").message] },
      {
        sessionManager: {
          getSessionId: () => "session",
          getBranch: () => [
            user("before"),
            { type: "compaction", summary: "summary" },
            assistant("after"),
          ],
        },
      }
    );
    expect(onTranscript.mock.calls[0][1]).toEqual([
      { role: "user", content: "before" },
      { role: "assistant", content: "after" },
    ]);
  });

  it("uses the active branch rather than sibling entries or the event list", async () => {
    const sibling = assistant("sibling");
    await adapter()(
      { messages: [sibling.message] },
      {
        sessionManager: {
          getSessionId: () => "session",
          getBranch: () => [user("shared"), assistant("active")],
          getEntries: () => [user("shared"), sibling, assistant("active")],
        },
      }
    );
    expect(onTranscript.mock.calls[0][1]).toEqual([
      { role: "user", content: "shared" },
      { role: "assistant", content: "active" },
    ]);
  });

  it("does not retain an empty active branch", async () => {
    await adapter()(
      { messages: [assistant("stale").message] },
      {
        sessionManager: { getSessionId: () => "session", getBranch: () => [] },
      }
    );
    expect(onTranscript).not.toHaveBeenCalled();
  });
});

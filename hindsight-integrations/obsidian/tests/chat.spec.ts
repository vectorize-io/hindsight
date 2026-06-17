import { describe, expect, it, vi } from "vitest";
import type { HindsightClient } from "../src/client";
import { runChatTurn } from "../src/chat";
import type { ReflectResponse } from "../src/types";

function fakeClient(reflectResult: ReflectResponse) {
  return {
    retain: vi.fn(async () => {}),
    reflect: vi.fn(async () => reflectResult),
  };
}

const ANSWER: ReflectResponse = {
  text: "grounded answer",
  based_on: { memories: [{ id: "1", text: "fact", document_id: "Note.md" }] },
};

describe("runChatTurn", () => {
  it("§0.5: does NOT retain conversation when rememberConversations is off", async () => {
    const client = fakeClient(ANSWER);
    const res = await runChatTurn(
      {
        client: client as unknown as HindsightClient,
        bankId: "bank",
        budget: "low",
        rememberConversations: false,
      },
      "what are my open projects?"
    );

    expect(client.reflect).toHaveBeenCalledTimes(1);
    expect(client.reflect).toHaveBeenCalledWith(
      "bank",
      "what are my open projects?",
      expect.objectContaining({ includeCitations: true })
    );
    expect(client.retain).not.toHaveBeenCalled();
    expect(res.text).toBe("grounded answer");
  });

  it("forwards the scope tag_groups filter to reflect", async () => {
    const client = fakeClient(ANSWER);
    const tagGroups = [{ tags: ["vault:Personal"], match: "all_strict" as const }];
    await runChatTurn(
      {
        client: client as unknown as HindsightClient,
        bankId: "bank",
        budget: "low",
        rememberConversations: false,
        tagGroups,
      },
      "scoped question"
    );

    expect(client.reflect).toHaveBeenCalledWith(
      "bank",
      "scoped question",
      expect.objectContaining({ tagGroups })
    );
  });

  it("retains user + assistant turns only when explicitly enabled", async () => {
    const client = fakeClient(ANSWER);
    let n = 0;
    await runChatTurn(
      {
        client: client as unknown as HindsightClient,
        bankId: "bank",
        budget: "low",
        rememberConversations: true,
        newConversationDocId: (role) => `conversation/${(n += 1)}-${role}`,
      },
      "hello"
    );

    expect(client.retain).toHaveBeenCalledTimes(2);
    expect(client.retain).toHaveBeenNthCalledWith(
      1,
      "bank",
      "conversation/1-user",
      "hello",
      expect.objectContaining({ tags: ["conversation", "user"] })
    );
    expect(client.retain).toHaveBeenNthCalledWith(
      2,
      "bank",
      "conversation/2-assistant",
      "grounded answer",
      expect.objectContaining({ tags: ["conversation", "assistant"] })
    );
  });
});

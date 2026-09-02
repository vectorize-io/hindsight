import { beforeEach, describe, expect, it, vi } from "vitest";
import { createHindsightEvaluator } from "../src/index.js";
import type { HindsightClient } from "../src/index.js";
import { AGENT_ID, USER_ID, agentMessage, mockClient, runtime, userMessage } from "./helpers.js";

describe("createHindsightEvaluator", () => {
  let client: HindsightClient;

  beforeEach(() => {
    client = mockClient();
  });

  it("exposes the elizaOS evaluator contract", () => {
    const evaluator = createHindsightEvaluator(client, undefined);
    expect(evaluator.name).toBe("HINDSIGHT_RETAIN");
    expect(evaluator.alwaysRun).toBe(true);
    expect(evaluator.examples).toEqual([]);
  });

  it("validates only messages that carry non-empty text", async () => {
    const evaluator = createHindsightEvaluator(client, undefined);
    expect(await evaluator.validate(runtime, userMessage("hi"))).toBe(true);
    expect(await evaluator.validate(runtime, userMessage("   "))).toBe(false);
    expect(await evaluator.validate(runtime, userMessage(undefined))).toBe(false);
  });

  it("retains with async:true by default", async () => {
    const evaluator = createHindsightEvaluator(client, undefined);
    await evaluator.handler(runtime, userMessage("hi"), undefined, undefined, undefined, undefined);
    expect(client.retain).toHaveBeenCalledWith("00000000-0000-0000-0000-0000000000u5", "hi", {
      async: true,
      tags: undefined,
      metadata: undefined,
    });
  });

  it("does not block the turn on a slow retain in async mode (fire-and-forget)", async () => {
    // A retain that never settles must not hang the handler when async:true.
    (client.retain as ReturnType<typeof vi.fn>).mockImplementationOnce(() => new Promise(() => {}));
    const evaluator = createHindsightEvaluator(client, undefined, { async: true });
    await expect(
      evaluator.handler(runtime, userMessage("hi"), undefined, undefined, undefined, undefined)
    ).resolves.toBeUndefined();
    expect(client.retain).toHaveBeenCalledTimes(1);
  });

  it("passes tags and metadata through to retain", async () => {
    const evaluator = createHindsightEvaluator(client, undefined, {
      async: false,
      tags: ["source:eliza", "env:test"],
      metadata: { channel: "discord" },
    });
    await evaluator.handler(runtime, userMessage("hi"), undefined, undefined, undefined, undefined);
    expect(client.retain).toHaveBeenCalledWith(USER_ID, "hi", {
      async: false,
      tags: ["source:eliza", "env:test"],
      metadata: { channel: "discord" },
    });
  });

  it("resolves the bank via a resolver function", async () => {
    const evaluator = createHindsightEvaluator(client, (m) => `room:${m.roomId}`, {
      async: false,
    });
    const message = userMessage("hi");
    await evaluator.handler(runtime, message, undefined, undefined, undefined, undefined);
    expect(client.retain).toHaveBeenCalledWith(`room:${message.roomId}`, "hi", expect.any(Object));
  });

  it("skips the agent's own triggering message by default", async () => {
    const evaluator = createHindsightEvaluator(client, undefined, { async: false });
    await evaluator.handler(
      runtime,
      agentMessage("my reply"),
      undefined,
      undefined,
      undefined,
      undefined
    );
    expect(client.retain).not.toHaveBeenCalled();
  });

  it("retains the agent's own triggering message when includeAgentMessages is set", async () => {
    const evaluator = createHindsightEvaluator(client, undefined, {
      async: false,
      includeAgentMessages: true,
    });
    // The agent's own message defaults its bank to the agent's entityId.
    await evaluator.handler(
      runtime,
      agentMessage("my reply"),
      undefined,
      undefined,
      undefined,
      undefined
    );
    expect(client.retain).toHaveBeenCalledWith(AGENT_ID, "my reply", expect.any(Object));
  });

  it("retains the user message and every agent response when includeAgentMessages is set", async () => {
    const evaluator = createHindsightEvaluator(client, undefined, {
      async: false,
      includeAgentMessages: true,
    });
    const responses = [agentMessage("answer one"), agentMessage("answer two")];
    await evaluator.handler(
      runtime,
      userMessage("hello"),
      undefined,
      undefined,
      undefined,
      responses
    );
    expect(client.retain).toHaveBeenCalledWith(USER_ID, "hello", expect.any(Object));
    expect(client.retain).toHaveBeenCalledWith(USER_ID, "answer one", expect.any(Object));
    expect(client.retain).toHaveBeenCalledWith(USER_ID, "answer two", expect.any(Object));
    expect(client.retain).toHaveBeenCalledTimes(3);
  });

  it("does not retain agent responses unless includeAgentMessages is set", async () => {
    const evaluator = createHindsightEvaluator(client, undefined, { async: false });
    const responses = [agentMessage("answer one")];
    await evaluator.handler(
      runtime,
      userMessage("hello"),
      undefined,
      undefined,
      undefined,
      responses
    );
    expect(client.retain).toHaveBeenCalledTimes(1);
    expect(client.retain).toHaveBeenCalledWith(USER_ID, "hello", expect.any(Object));
  });

  it("skips agent responses that have no text", async () => {
    const evaluator = createHindsightEvaluator(client, undefined, {
      async: false,
      includeAgentMessages: true,
    });
    const responses = [agentMessage("real answer"), { ...userMessage("  "), entityId: AGENT_ID }];
    await evaluator.handler(
      runtime,
      userMessage("hello"),
      undefined,
      undefined,
      undefined,
      responses
    );
    expect(client.retain).toHaveBeenCalledWith(USER_ID, "hello", expect.any(Object));
    expect(client.retain).toHaveBeenCalledWith(USER_ID, "real answer", expect.any(Object));
    expect(client.retain).toHaveBeenCalledTimes(2);
  });

  it("does nothing when the triggering message has no text", async () => {
    const evaluator = createHindsightEvaluator(client, undefined, { async: false });
    await evaluator.handler(
      runtime,
      userMessage(undefined),
      undefined,
      undefined,
      undefined,
      undefined
    );
    expect(client.retain).not.toHaveBeenCalled();
  });

  it("swallows retain failures in sync mode without rejecting the turn", async () => {
    (client.retain as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("boom"));
    const evaluator = createHindsightEvaluator(client, undefined, { async: false });
    await expect(
      evaluator.handler(runtime, userMessage("hi"), undefined, undefined, undefined, undefined)
    ).resolves.toBeUndefined();
    expect(client.retain).toHaveBeenCalledTimes(1);
  });
});

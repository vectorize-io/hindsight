import { describe, expect, it } from "vitest";
import { resolveBank } from "../src/index.js";
import { USER_ID, userMessage } from "./helpers.js";

describe("resolveBank", () => {
  it("defaults to the message entityId when no bank is given", () => {
    expect(resolveBank(undefined, userMessage("hi"))).toBe(USER_ID);
  });

  it("uses a fixed string bank for every message", () => {
    expect(resolveBank("team-bank", userMessage("hi"))).toBe("team-bank");
  });

  it("falls back to entityId when the bank string is empty", () => {
    expect(resolveBank("", userMessage("hi"))).toBe(USER_ID);
  });

  it("derives the bank per message via a resolver function", () => {
    const message = userMessage("hi");
    const resolver = (m: typeof message) => `room:${m.roomId}`;
    expect(resolveBank(resolver, message)).toBe(`room:${message.roomId}`);
  });

  it("passes the actual message object to the resolver function", () => {
    const message = userMessage("hi");
    let received: unknown;
    resolveBank((m) => {
      received = m;
      return "b";
    }, message);
    expect(received).toBe(message);
  });
});

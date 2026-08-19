import type { IAgentRuntime, Memory } from "@elizaos/core";
import { vi } from "vitest";
import type { HindsightClient, RecallResponse, RetainResponse } from "../src/index.js";

export const AGENT_ID = "00000000-0000-0000-0000-0000000000a9" as const;
export const USER_ID = "00000000-0000-0000-0000-0000000000u5" as const;
export const ROOM_ID = "00000000-0000-0000-0000-0000000000r1" as const;

/** A mock client whose recall/retain can be overridden per call via vi.fn helpers. */
export function mockClient(recall?: RecallResponse): HindsightClient {
  return {
    recall: vi.fn(
      async (): Promise<RecallResponse> =>
        recall ?? {
          results: [
            { id: "1", text: "User prefers dark mode" },
            { id: "2", text: "User lives in Berlin" },
          ],
        }
    ),
    retain: vi.fn(
      async (bankId: string): Promise<RetainResponse> => ({
        success: true,
        bank_id: bankId,
        items_count: 1,
        async: true,
      })
    ),
  };
}

export function userMessage(text: string | undefined): Memory {
  return {
    entityId: USER_ID,
    agentId: AGENT_ID,
    roomId: ROOM_ID,
    content: text === undefined ? {} : { text },
  } as Memory;
}

export function agentMessage(text: string): Memory {
  return { ...userMessage(text), entityId: AGENT_ID } as Memory;
}

export const runtime = { agentId: AGENT_ID } as IAgentRuntime;

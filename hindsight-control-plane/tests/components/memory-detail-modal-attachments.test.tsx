// @vitest-environment jsdom
/**
 * The memory dialog shows the attachments behind the fact — in *both* layouts.
 *
 * The dialog renders two entirely separate trees, `isObservation ? … : …`, and
 * an addition to one is invisible in the other. That is exactly how the strip
 * came to work for observations and not for the world/experience memories that
 * make up nearly everything a user opens: the data was there, the markup was
 * there, and it was in the wrong branch.
 *
 * So this asserts the same thing twice, once per branch, which is the only shape
 * of test that would have caught it.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";

vi.mock("next-intl", () => ({
  useTranslations: () => Object.assign((key: string) => key, { rich: (key: string) => key }),
}));

vi.mock("@/lib/bank-context", () => ({
  useBank: () => ({ currentBank: "bank-a" }),
}));

const getMemory = vi.fn();
vi.mock("@/lib/api", () => ({
  client: {
    getMemory: (...args: unknown[]) => getMemory(...args),
    getChunk: vi.fn().mockResolvedValue(null),
    getDocument: vi.fn().mockResolvedValue(null),
    listLLMRequests: vi.fn().mockResolvedValue({ items: [] }),
    getObservationHistory: vi.fn().mockResolvedValue({ items: [] }),
    listMemories: vi.fn().mockResolvedValue({ items: [] }),
  },
}));

import { MemoryDetailModal } from "@/components/memory-detail-modal";

const ATTACHMENT = {
  id: "c414cd0e204d",
  hash: "c414cd0e204d0000",
  kind: "image",
  media_type: "image/png",
  byte_size: 10158,
};

function memoryOfType(type: string) {
  return {
    id: "mem-1",
    text: "To reset the VPN, click [image: image/png] then reconnect.",
    type,
    context: "",
    date: new Date().toISOString(),
    entities: [],
    chunk_id: "bank-a_doc_0",
    tags: [],
    attachments: [ATTACHMENT],
  };
}

beforeEach(() => getMemory.mockReset());
afterEach(cleanup);

describe("memory dialog attachments", () => {
  // "world" is the overwhelmingly common case and the one that was broken.
  it.each(["world", "experience", "observation"])("renders them for a %s memory", async (type) => {
    getMemory.mockResolvedValue(memoryOfType(type));

    render(<MemoryDetailModal memoryId="mem-1" onClose={() => {}} />);

    await waitFor(() => expect(getMemory).toHaveBeenCalled());
    const image = await screen.findByRole("img");
    expect(image.getAttribute("src")).toBe(`/api/banks/bank-a/attachments/${ATTACHMENT.id}`);
  });

  it("renders nothing extra for a memory with no attachments", async () => {
    getMemory.mockResolvedValue({ ...memoryOfType("world"), attachments: undefined });

    render(<MemoryDetailModal memoryId="mem-1" onClose={() => {}} />);

    await waitFor(() => expect(getMemory).toHaveBeenCalled());
    expect(screen.queryByRole("img")).toBeNull();
  });
});

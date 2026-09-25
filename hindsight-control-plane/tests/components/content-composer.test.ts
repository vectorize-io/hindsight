import { describe, it, expect } from "vitest";

import {
  hasComposedContent,
  toRetainContent,
  type ComposerBlock,
} from "@/components/content-composer";

const png: ComposerBlock = {
  kind: "attachment",
  name: "inv.png",
  mediaType: "image/png",
  data: "AAAA",
  size: 3,
};
const pdf: ComposerBlock = {
  kind: "attachment",
  name: "audit.pdf",
  mediaType: "application/pdf",
  data: "BBBB",
  size: 3,
};

describe("toRetainContent", () => {
  it("sends the plain string in text mode, whatever the blocks hold", () => {
    expect(toRetainContent("text", "hello", [png])).toBe("hello");
  });

  it("drops blank text blocks and types attachments by media type, in order", () => {
    expect(
      toRetainContent("blocks", "", [{ kind: "text", text: "  " }, png, { kind: "text", text: "after" }, pdf])
    ).toEqual([
      { type: "image", source: { type: "base64", media_type: "image/png", data: "AAAA" } },
      { type: "text", text: "after" },
      {
        type: "file",
        filename: "audit.pdf",
        source: { type: "base64", media_type: "application/pdf", data: "BBBB" },
      },
    ]);
  });
});

describe("hasComposedContent", () => {
  it("counts an attachment alone as content, but not blank text", () => {
    expect(hasComposedContent("blocks", "", [png])).toBe(true);
    expect(hasComposedContent("blocks", "typed", [{ kind: "text", text: " " }])).toBe(false);
    expect(hasComposedContent("text", " ", [png])).toBe(false);
  });
});

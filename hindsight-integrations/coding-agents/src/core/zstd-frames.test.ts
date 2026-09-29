import { zstdCompressSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import { scanZstdFrames, zstdDecompressFrames } from "./zstd-frames";

const frames = (...parts: string[]) =>
  Buffer.concat(parts.map((part) => zstdCompressSync(Buffer.from(part, "utf8"))));

describe("zstdDecompressFrames", () => {
  it("decodes every frame of a concatenated artifact", () => {
    // The whole point: Node's own one-shot decompress returns ONLY the first frame here, which is
    // how a dsh session log reads back as nothing but its header line.
    const buffer = frames(
      '{"type":"session"}\n',
      '{"type":"user/message"}\n',
      '{"type":"turn/end"}\n'
    );
    expect(zstdDecompressFrames(buffer)).toBe(
      '{"type":"session"}\n{"type":"user/message"}\n{"type":"turn/end"}\n'
    );
    expect(scanZstdFrames(buffer)).toHaveLength(3);
  });

  it("keeps the complete frames when the last one is torn by a crash", () => {
    const complete = frames("first\n", "second\n");
    const torn = Buffer.concat([complete, zstdCompressSync(Buffer.from("third\n")).subarray(0, 6)]);
    expect(zstdDecompressFrames(torn)).toBe("first\nsecond\n");
  });

  it("returns nothing for bytes that are not a zstd frame", () => {
    expect(scanZstdFrames(Buffer.from("plain text, not compressed"))).toEqual([]);
    expect(zstdDecompressFrames(Buffer.alloc(0))).toBe("");
  });

  it("round-trips a payload large enough to span several blocks", () => {
    const big = `${"x".repeat(300_000)}\n`;
    expect(zstdDecompressFrames(frames(big, "tail\n"))).toBe(`${big}tail\n`);
  });
});

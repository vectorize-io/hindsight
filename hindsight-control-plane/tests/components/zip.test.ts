import { describe, expect, it } from "vitest";
import { crc32, zipFiles } from "@/lib/zip";

describe("zipFiles", () => {
  it("computes the standard CRC-32", () => {
    expect(crc32(new TextEncoder().encode("hello"))).toBe(0x3610a686);
  });

  it("writes an archive whose directory points at each file's bytes", () => {
    const files = [
      { path: "index.md", content: "# Bank\n" },
      { path: "gemma/vllm.md", content: "---\ntype: Memory Topic\n---\nGemma é ✓\n" },
    ];
    const zip = zipFiles(files);
    const v = new DataView(zip.buffer);
    const end = zip.length - 22;
    expect(v.getUint32(end, true)).toBe(0x06054b50);
    expect(v.getUint16(end + 10, true)).toBe(2);
    let at = v.getUint32(end + 16, true);
    const dec = new TextDecoder();
    for (const file of files) {
      expect(v.getUint32(at, true)).toBe(0x02014b50);
      const size = v.getUint32(at + 24, true);
      const nameLen = v.getUint16(at + 28, true);
      const local = v.getUint32(at + 42, true);
      expect(dec.decode(zip.subarray(at + 46, at + 46 + nameLen))).toBe(file.path);
      const dataAt = local + 30 + v.getUint16(local + 26, true);
      expect(dec.decode(zip.subarray(dataAt, dataAt + size))).toBe(file.content);
      at += 46 + nameLen;
    }
  });
});

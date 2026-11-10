import { describe, expect, it } from "vitest";

import { parseByteRange } from "@/lib/byte-range";
import {
  buildUploadForm,
  errorMessage,
  fillTimeseries,
  firstPassageUrl,
  groupByMediaType,
  isPicturePassage,
  splitContext,
  formatBytes,
  formatMs,
  formatWindow,
  knowledgeApi,
  parseTags,
  proxiedMediaUrl,
  type KnowledgeSearchResult,
  type MediaType,
} from "@/lib/knowledge-banks";

describe("formatting", () => {
  it("formats bytes in 1024 steps", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(103)).toBe("103 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(182423)).toBe("178 KB");
    expect(formatBytes(524288000)).toBe("500 MB");
  });

  it("formats milliseconds as clock time", () => {
    expect(formatMs(0)).toBe("0:00");
    expect(formatMs(65_000)).toBe("1:05");
    expect(formatMs(3_725_000)).toBe("1:02:05");
  });

  it("formats a passage window, or nothing for an untimed passage", () => {
    expect(formatWindow({ start_ms: 60_000, end_ms: 120_000 })).toBe("1:00 – 2:00");
    expect(formatWindow({ start_ms: 0, end_ms: null })).toBe("0:00");
    expect(formatWindow({ start_ms: null, end_ms: null })).toBeNull();
  });
});

describe("input parsing", () => {
  it("splits tags and drops blanks", () => {
    expect(parseTags(" v2, onboarding ,,")).toEqual(["v2", "onboarding"]);
    expect(parseTags("")).toEqual([]);
  });

  it("reads FastAPI and proxy error bodies", () => {
    expect(errorMessage({ detail: "too big" })).toBe("too big");
    expect(errorMessage({ detail: [{ msg: "a" }, { msg: "b" }] })).toBe("a; b");
    expect(errorMessage({ error: "boom" })).toBe("boom");
    expect(errorMessage("nope")).toBeNull();
  });
});

describe("urls", () => {
  it("percent-encodes every path segment", () => {
    expect(knowledgeApi()).toBe("/api/knowledge-banks");
    expect(knowledgeApi("a/b", "documents", "x y")).toBe(
      "/api/knowledge-banks/a%2Fb/documents/x%20y"
    );
  });

  it("maps an API media path onto the proxy", () => {
    expect(proxiedMediaUrl("/v1/default/knowledge-banks/docs/documents/d/passages/3/media")).toBe(
      "/api/knowledge-banks/docs/documents/d/passages/3/media"
    );
  });
});

describe("buildUploadForm", () => {
  it("sends one files part per file and the options aligned by index", async () => {
    const a = new File(["aaa"], "a.pdf", { type: "application/pdf" });
    const b = new File(["bb"], "b.mp3", { type: "audio/mpeg" });
    const form = buildUploadForm([a, b], [{ tags: ["v2"] }, { tags: [], document_id: "b" }]);
    const files = form.getAll("files") as File[];
    expect(files.map((f) => f.name)).toEqual(["a.pdf", "b.mp3"]);
    expect(JSON.parse(form.get("request") as string)).toEqual({
      files: [{ tags: ["v2"] }, { document_id: "b", tags: [] }],
    });
  });
});

describe("firstPassageUrl", () => {
  it("points a document at its first passage's media", () => {
    expect(
      firstPassageUrl({ document_url: "/v1/default/knowledge-banks/docs/documents/d/file" })
    ).toBe("/v1/default/knowledge-banks/docs/documents/d/passages/0/media");
  });
});

describe("parseByteRange", () => {
  it("parses the three range forms", () => {
    expect(parseByteRange("bytes=0-99", 1000)).toEqual({ start: 0, end: 99 });
    expect(parseByteRange("bytes=900-", 1000)).toEqual({ start: 900, end: 999 });
    expect(parseByteRange("bytes=-100", 1000)).toEqual({ start: 900, end: 999 });
  });

  it("clamps an end past the body", () => {
    expect(parseByteRange("bytes=0-5000", 1000)).toEqual({ start: 0, end: 999 });
  });

  it("refuses a start past the body", () => {
    expect(parseByteRange("bytes=1000-", 1000)).toBe("unsatisfiable");
  });

  it("serves the whole body for absent, malformed or multi-range headers", () => {
    expect(parseByteRange(null, 1000)).toBeNull();
    expect(parseByteRange("items=0-1", 1000)).toBeNull();
    expect(parseByteRange("bytes=0-1,5-9", 1000)).toBeNull();
    expect(parseByteRange("bytes=9-1", 1000)).toBeNull();
  });
});

describe("fillTimeseries", () => {
  const now = new Date("2026-11-05T12:30:00Z");

  it("lays the server's sparse daily buckets on the whole period, oldest first", () => {
    const { trunc, buckets } = fillTimeseries(
      [
        { time: "2026-11-03T00:00:00Z", documents: 1, passages: 2 },
        { time: "2026-11-05T00:00:00Z", documents: 2, passages: 4 },
        { time: "2026-10-01T00:00:00Z", documents: 9, passages: 9 }, // outside 7d
      ],
      "7d",
      now
    );
    expect(trunc).toBe("day");
    expect(buckets).toHaveLength(7);
    expect(buckets[0]).toEqual({ time: "2026-10-30T00:00:00.000Z", documents: 0, passages: 0 });
    expect(buckets[4]).toEqual({ time: "2026-11-03T00:00:00.000Z", documents: 1, passages: 2 });
    expect(buckets[6]).toEqual({ time: "2026-11-05T00:00:00.000Z", documents: 2, passages: 4 });
    expect(buckets.reduce((s, b) => s + b.documents, 0)).toBe(3);
  });

  it("uses hourly buckets for one day", () => {
    const { trunc, buckets } = fillTimeseries(
      [{ time: "2026-11-05T12:00:00Z", documents: 1, passages: 3 }],
      "1d",
      now
    );
    expect(trunc).toBe("hour");
    expect(buckets).toHaveLength(24);
    expect(buckets[23]).toMatchObject({ time: "2026-11-05T12:00:00.000Z", passages: 3 });
  });
});

describe("groupByMediaType", () => {
  it("groups hits by media type, ordered by each group's best hit, keeping overall ranks", () => {
    const hit = (document_id: string, media_type: MediaType) =>
      ({ document_id, media_type }) as KnowledgeSearchResult;
    const groups = groupByMediaType([
      hit("a", "document"),
      hit("b", "audio"),
      hit("c", "document"),
      hit("d", "image"),
    ]);
    expect(
      groups.map((g) => [g.mediaType, g.hits.map((h) => `${h.result.document_id}#${h.rank}`)])
    ).toEqual([
      ["document", ["a#1", "c#3"]],
      ["audio", ["b#2"]],
      ["image", ["d#4"]],
    ]);
  });
});

describe("isPicturePassage", () => {
  it("treats images and rendered document pages with media as pictures", () => {
    expect(isPicturePassage({ media_type: "document", passage_url: "/p" })).toBe(true);
    expect(isPicturePassage({ media_type: "image", passage_url: "/p" })).toBe(true);
    expect(isPicturePassage({ media_type: "document", passage_url: null })).toBe(false);
    expect(isPicturePassage({ media_type: "video", passage_url: "/p" })).toBe(false);
  });
});

describe("splitContext", () => {
  it("splits the context around the hit", () => {
    expect(splitContext("A. B is the hit. C.", "B is the hit.")).toEqual({
      before: "A. ",
      hit: "B is the hit.",
      after: " C.",
    });
  });

  it("gives up when there is no context or the hit is not in it", () => {
    expect(splitContext(null, "x")).toBeNull();
    expect(splitContext("abc", null)).toBeNull();
    expect(splitContext("abc", "zzz")).toBeNull();
  });
});

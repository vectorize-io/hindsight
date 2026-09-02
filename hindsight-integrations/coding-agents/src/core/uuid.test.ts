import { describe, expect, it } from "vitest";
import { HINDSIGHT_NAMESPACE, uuidV5 } from "./uuid";

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const DNS_NAMESPACE = "6ba7b810-9dad-11d1-80b4-00c04fd430c8";

describe("uuidV5", () => {
  it("matches the published RFC 4122 vector", () => {
    // The canonical uuid5(NAMESPACE_DNS, "python.org") — proves the digest, version and variant
    // bits are laid out the way every other implementation (including the server's) expects.
    expect(uuidV5("python.org", DNS_NAMESPACE)).toBe("886313e1-3b8a-5372-9b90-0c9aee199e5d");
  });

  it("is a well-formed v5 UUID — the API rejects anything else as an operation_id", () => {
    expect(uuidV5("conversation:s1")).toMatch(UUID_RE);
  });

  it("is deterministic per name and separates distinct names", () => {
    expect(uuidV5("a")).toBe(uuidV5("a"));
    expect(uuidV5("a")).not.toBe(uuidV5("b"));
  });

  it("separates names across namespaces", () => {
    expect(uuidV5("a", HINDSIGHT_NAMESPACE)).not.toBe(uuidV5("a", DNS_NAMESPACE));
  });

  it("rejects a malformed namespace rather than silently hashing garbage", () => {
    expect(() => uuidV5("a", "not-a-uuid")).toThrow(/invalid UUID namespace/);
  });
});

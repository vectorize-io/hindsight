import { afterEach, describe, expect, it, vi } from "vitest";

import {
  assertValidControlPlaneAuthConfig,
  getControlPlaneAuthProvider,
  getExpectedDataplaneAuthProfile,
} from "@/lib/auth/provider";

describe("control-plane auth provider", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("defaults to disabled when no access key is configured", () => {
    vi.stubEnv("HINDSIGHT_CP_AUTH_PROVIDER", "");
    vi.stubEnv("HINDSIGHT_CP_ACCESS_KEY", "");

    expect(getControlPlaneAuthProvider()).toBe("disabled");
    expect(getExpectedDataplaneAuthProfile()).toBe("disabled");
  });

  it("defaults to access_key when an access key is configured", () => {
    vi.stubEnv("HINDSIGHT_CP_AUTH_PROVIDER", "");
    vi.stubEnv("HINDSIGHT_CP_ACCESS_KEY", "secret");

    expect(getControlPlaneAuthProvider()).toBe("access_key");
  });

  it("rejects access_key provider without an access key", () => {
    vi.stubEnv("HINDSIGHT_CP_AUTH_PROVIDER", "access_key");
    vi.stubEnv("HINDSIGHT_CP_ACCESS_KEY", "");

    expect(() => assertValidControlPlaneAuthConfig()).toThrow(/requires HINDSIGHT_CP_ACCESS_KEY/);
  });

  it("rejects unknown auth providers", () => {
    vi.stubEnv("HINDSIGHT_CP_AUTH_PROVIDER", "unknown_profile");

    expect(() => getControlPlaneAuthProvider()).toThrow(/Unsupported HINDSIGHT_CP_AUTH_PROVIDER/);
  });
});

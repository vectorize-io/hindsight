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

  it("uses supabase_org as an auth profile", () => {
    vi.stubEnv("HINDSIGHT_CP_AUTH_PROVIDER", "supabase_org");
    vi.stubEnv("HINDSIGHT_CP_ACCESS_KEY", "");

    expect(getControlPlaneAuthProvider()).toBe("supabase_org");
    expect(getExpectedDataplaneAuthProfile()).toBe("supabase_org");
  });

  it("rejects supabase_org combined with the access key provider secret", () => {
    vi.stubEnv("HINDSIGHT_CP_AUTH_PROVIDER", "supabase_org");
    vi.stubEnv("HINDSIGHT_CP_ACCESS_KEY", "secret");

    expect(() => assertValidControlPlaneAuthConfig()).toThrow(/must not be combined/);
  });
});

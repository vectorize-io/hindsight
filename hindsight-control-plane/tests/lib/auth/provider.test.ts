import { afterEach, describe, expect, it, vi } from "vitest";

import {
  assertValidControlPlaneAuthConfig,
  getControlPlaneAuthProvider,
  isSupabaseOrgSessionPresent,
} from "@/lib/auth/provider";

describe("control-plane auth provider", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("defaults to disabled when no access key or explicit provider is configured", () => {
    vi.stubEnv("HINDSIGHT_CP_AUTH_PROVIDER", "");
    vi.stubEnv("HINDSIGHT_CP_ACCESS_KEY", "");

    expect(getControlPlaneAuthProvider()).toBe("disabled");
  });

  it("keeps existing access-key behavior as the compatibility default", () => {
    vi.stubEnv("HINDSIGHT_CP_AUTH_PROVIDER", "");
    vi.stubEnv("HINDSIGHT_CP_ACCESS_KEY", "secret");

    expect(getControlPlaneAuthProvider()).toBe("access_key");
  });

  it("requires supabase_org to be explicitly configured", () => {
    vi.stubEnv("HINDSIGHT_CP_AUTH_PROVIDER", "supabase_org");
    vi.stubEnv("HINDSIGHT_CP_ACCESS_KEY", "");

    expect(getControlPlaneAuthProvider()).toBe("supabase_org");
  });

  it("rejects access_key provider without an access key", () => {
    vi.stubEnv("HINDSIGHT_CP_AUTH_PROVIDER", "access_key");
    vi.stubEnv("HINDSIGHT_CP_ACCESS_KEY", "");

    expect(() => assertValidControlPlaneAuthConfig()).toThrow(/requires HINDSIGHT_CP_ACCESS_KEY/);
  });

  it("rejects combining supabase_org with the legacy control-plane access key", () => {
    vi.stubEnv("HINDSIGHT_CP_AUTH_PROVIDER", "supabase_org");
    vi.stubEnv("HINDSIGHT_CP_ACCESS_KEY", "secret");

    expect(() => assertValidControlPlaneAuthConfig()).toThrow(/must not be combined/);
  });

  it("requires both Supabase token and selected org cookies for a supabase_org session", () => {
    expect(isSupabaseOrgSessionPresent("token", "org")).toBe(true);
    expect(isSupabaseOrgSessionPresent("token", undefined)).toBe(false);
    expect(isSupabaseOrgSessionPresent(undefined, "org")).toBe(false);
  });
});

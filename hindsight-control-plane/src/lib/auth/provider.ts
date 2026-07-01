export type ControlPlaneAuthProvider = "disabled" | "access_key" | "supabase_org";

export const SUPABASE_ORG_ACCESS_TOKEN_COOKIE = "hindsight_supabase_access_token";
export const SUPABASE_ORG_SELECTED_ORG_COOKIE = "hindsight_selected_org";
export const SUPABASE_ORG_REFRESH_TOKEN_COOKIE = "hindsight_supabase_refresh_token";

export function getControlPlaneAuthProvider(): ControlPlaneAuthProvider {
  const configured = process.env.HINDSIGHT_CP_AUTH_PROVIDER?.trim();
  if (configured === "disabled" || configured === "access_key" || configured === "supabase_org") {
    return configured;
  }
  return process.env.HINDSIGHT_CP_ACCESS_KEY ? "access_key" : "disabled";
}

export function assertValidControlPlaneAuthConfig(): void {
  const provider = getControlPlaneAuthProvider();
  if (provider === "access_key" && !process.env.HINDSIGHT_CP_ACCESS_KEY) {
    throw new Error("HINDSIGHT_CP_AUTH_PROVIDER=access_key requires HINDSIGHT_CP_ACCESS_KEY");
  }
  if (provider === "supabase_org" && process.env.HINDSIGHT_CP_ACCESS_KEY) {
    throw new Error(
      "HINDSIGHT_CP_AUTH_PROVIDER=supabase_org must not be combined with HINDSIGHT_CP_ACCESS_KEY"
    );
  }
}

export function isSupabaseOrgSessionPresent(
  accessToken: string | undefined,
  selectedOrgId: string | undefined
): boolean {
  return Boolean(accessToken && selectedOrgId);
}

export type OrgCreationPolicy = "open" | "direct_signup_only";

export function getOrgCreationPolicy(): OrgCreationPolicy {
  const configured = process.env.HINDSIGHT_AUTH_ORG_CREATION_POLICY?.trim();
  if (configured === "direct_signup_only") return configured;
  return "open";
}

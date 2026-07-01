/**
 * Next.js instrumentation file - runs exactly once at server startup.
 * https://nextjs.org/docs/app/building-your-application/optimizing/instrumentation
 */
export async function register() {
  const { assertValidControlPlaneAuthConfig, getControlPlaneAuthProvider } =
    await import("@/lib/auth/provider");
  assertValidControlPlaneAuthConfig();
  const authProvider = getControlPlaneAuthProvider();
  const dataplaneUrl = process.env.HINDSIGHT_CP_DATAPLANE_API_URL || "http://localhost:8888";
  const apiKey = process.env.HINDSIGHT_CP_DATAPLANE_API_KEY || "";

  console.log(`[Control Plane] Connecting to dataplane at: ${dataplaneUrl}`);
  console.log(`[Control Plane] Auth provider: ${authProvider}`);
  if (apiKey && authProvider !== "supabase_org") {
    console.log("[Control Plane] Using API key authentication");
  } else {
    console.log("[Control Plane] No API key configured (public access)");
  }
}

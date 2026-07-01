/**
 * Next.js instrumentation file - runs exactly once at server startup.
 * https://nextjs.org/docs/app/building-your-application/optimizing/instrumentation
 */
export async function register() {
  const dataplaneUrl = process.env.HINDSIGHT_CP_DATAPLANE_API_URL || "http://localhost:8888";
  const apiKey = process.env.HINDSIGHT_CP_DATAPLANE_API_KEY || "";

  console.log(`[Control Plane] Connecting to dataplane at: ${dataplaneUrl}`);
  if (apiKey) {
    console.log("[Control Plane] Using API key authentication");
  } else {
    console.log("[Control Plane] No API key configured (public access)");
  }

  // Warm the Langfuse client singleton so auth is verified at startup
  // instead of on first request. The SDK reads LANGFUSE_PUBLIC_KEY,
  // LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL from env automatically.
  if (process.env.LANGFUSE_PUBLIC_KEY || process.env.HINDSIGHT_CP_LANGFUSE_PUBLIC_KEY) {
    try {
      const { getLangfuse } = await import("@/lib/langfuse");
      getLangfuse();
    } catch (err) {
      console.warn("[Control Plane] Langfuse initialization skipped:", (err as Error).message);
    }
  }
}

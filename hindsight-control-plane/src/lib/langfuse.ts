/**
 * Langfuse server-side tracing client for the Control Plane.
 *
 * Initializes Langfuse at server startup and provides a shared client
 * for tracing API route handlers and server-side operations.
 *
 * Langfuse env vars (read automatically by the SDK):
 *   LANGFUSE_PUBLIC_KEY  — project public key
 *   LANGFUSE_SECRET_KEY  — project secret key
 *   LANGFUSE_BASE_URL    — Langfuse host URL (e.g. http://localhost:3002)
 *
 * The SDK honours the standard LANGFUSE_* env vars. To keep them alongside
 * the Hindsight-prefixed vars, both sets are supported below.
 */

let langfuseClient: ReturnType<typeof createLangfuseClient> | null = null;

function createLangfuseClient() {
  // Try Control-Plane specific vars first, then SDK-standard env vars.
  const publicKey =
    process.env.HINDSIGHT_CP_LANGFUSE_PUBLIC_KEY || process.env.LANGFUSE_PUBLIC_KEY || "";
  const secretKey =
    process.env.HINDSIGHT_CP_LANGFUSE_SECRET_KEY || process.env.LANGFUSE_SECRET_KEY || "";
  const baseUrl =
    process.env.HINDSIGHT_CP_LANGFUSE_HOST ||
    process.env.LANGFUSE_BASE_URL ||
    "http://localhost:3002";

  if (!publicKey || !secretKey) {
    console.warn("[Langfuse] Missing public/secret key — Langfuse tracing disabled");
    return null;
  }

  // Dynamic import to avoid pulling the SDK into client bundles

  const { Langfuse } = require("langfuse");

  const client = new Langfuse({
    publicKey,
    secretKey,
    baseUrl,
  });

  // Verify connectivity (non-blocking)
  client
    .authCheck()
    .then((result: { ok: boolean }) => {
      if (result.ok) {
        console.log(`[Langfuse] Connected to ${baseUrl}`);
      } else {
        console.warn(`[Langfuse] Auth check failed — host=${baseUrl}`);
      }
    })
    .catch((err: Error) => {
      console.warn(`[Langfuse] Connection error — host=${baseUrl}: ${err.message}`);
    });

  return client;
}

/** Get or lazily create the shared Langfuse client. */
export function getLangfuse() {
  if (!langfuseClient) {
    langfuseClient = createLangfuseClient();
  }
  return langfuseClient;
}

/**
 * Create a Langfuse trace for a Control Plane operation.
 * Returns null if Langfuse is not configured.
 */
export function createTrace(name: string, metadata?: Record<string, unknown>) {
  const client = getLangfuse();
  if (!client) return null;
  return client.trace({ name, metadata });
}

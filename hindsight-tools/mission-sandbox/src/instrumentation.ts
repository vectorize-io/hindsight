/** Runs once at server startup — load the deployment's .env before any route handler runs. */
export async function register() {
  if (process.env.NEXT_RUNTIME !== "nodejs") return;
  const { loadProjectEnv } = await import("@vectorize-io/hindsight-mission-sandbox/core");
  const loaded = loadProjectEnv();
  if (loaded) console.log(`[mission-sandbox] loaded env from ${loaded}`);
}

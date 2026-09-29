#!/usr/bin/env node
/**
 * Start the local `hindsight-embed` daemon. Spawned DETACHED by core/daemon.ts — never run inline
 * by a hook, because a cold start (uvx download + model load) outlives every hook timeout.
 *
 * Blocking here is the point: this process exists purely to own that wait, and exits once the
 * daemon answers /health. It writes only to the plugin log, so its stdio can be discarded.
 */
import { loadConfig } from "./core/config";
import { daemonEnv, isServerHealthy, preflightDaemon } from "./core/daemon";
import { diag } from "./core/diag";
import { log, setLogLevel } from "./core/log";

function arg(name: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : undefined;
}

async function main(): Promise<void> {
  const harness = arg("harness") ?? "coding-agent";
  const cfg = loadConfig({ harness });
  setLogLevel(cfg.logLevel);

  if (cfg.serverMode !== "daemon") return;
  // Several sessions can race to start one daemon; whoever loses simply finds it healthy.
  if (await isServerHealthy(cfg.apiUrl)) return;
  if (!preflightDaemon(cfg, harness)) return;

  const { HindsightServer } = await import("@vectorize-io/hindsight-all");
  const server = new HindsightServer({
    profile: cfg.daemonProfile,
    port: cfg.apiPort,
    env: daemonEnv(cfg),
    embedVersion: cfg.embedVersion,
    embedPackagePath: cfg.embedPackagePath,
    logger: {
      debug: (m) => log.debug(harness, m),
      info: (m) => log.info(harness, m),
      warn: (m) => log.warn(harness, m),
      error: (m) => log.error(harness, m),
    },
  });
  await server.start();
  diag(harness, "daemon_started", { apiUrl: cfg.apiUrl, profile: cfg.daemonProfile });
}

main().catch((error) => {
  diag(arg("harness") ?? "coding-agent", "daemon_start_failed", { error: String(error) });
});

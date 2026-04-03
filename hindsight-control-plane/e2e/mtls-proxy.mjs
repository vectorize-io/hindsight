#!/usr/bin/env node
/**
 * Local mTLS proxy for testing the control plane against a prod deployment
 * that requires mutual TLS (client certificates).
 *
 * Listens on LOCAL_PORT (default 18888) as plain HTTP and forwards all
 * requests to the remote HTTPS endpoint using the provided client certs.
 *
 * Usage:
 *   node e2e/mtls-proxy.mjs
 *
 * Environment:
 *   MTLS_CA_CERT     - path to CA certificate (default: ../openclaw-infra/ca.crt)
 *   MTLS_CLIENT_CERT - path to client certificate (default: ../openclaw-infra/client.crt)
 *   MTLS_CLIENT_KEY  - path to client key (default: ../openclaw-infra/client.key)
 *   MTLS_REMOTE_HOST - remote hostname (default: 34.208.169.77)
 *   MTLS_REMOTE_PORT - remote port (default: 443)
 *   MTLS_LOCAL_PORT  - local listen port (default: 18888)
 */

import http from "node:http";
import https from "node:https";
import tls from "node:tls";
import fs from "node:fs";
import path from "node:path";

const REMOTE_HOST = process.env.MTLS_REMOTE_HOST || "34.208.169.77";
const REMOTE_PORT = parseInt(process.env.MTLS_REMOTE_PORT || "443", 10);
const LOCAL_PORT = parseInt(process.env.MTLS_LOCAL_PORT || "18888", 10);

// e2e/ → control-plane/ → hindsight-contrib/ → code/ → openclaw-infra/
const infraDir = path.resolve(
  process.env.MTLS_INFRA_DIR || path.join(import.meta.dirname, "..", "..", "..", "openclaw-infra")
);

const caCert = fs.readFileSync(process.env.MTLS_CA_CERT || path.join(infraDir, "ca.crt"));
const clientCert = fs.readFileSync(process.env.MTLS_CLIENT_CERT || path.join(infraDir, "client.crt"));
const clientKey = fs.readFileSync(process.env.MTLS_CLIENT_KEY || path.join(infraDir, "client.key"));

// The server cert has CN=openclaw with no SAN — Node.js rejects it when
// connecting by IP. We verify the cert was signed by our CA (rejectUnauthorized
// + ca) but skip hostname matching since this is a self-signed internal CA.
const tlsOptions = {
  ca: caCert,
  cert: clientCert,
  key: clientKey,
  rejectUnauthorized: true,
  servername: "openclaw",
  checkServerIdentity: (hostname, cert) => {
    // Accept any cert signed by our CA — the CA check in rejectUnauthorized
    // ensures we're talking to the right server.
    return undefined;
  },
};

const server = http.createServer((req, res) => {
  const options = {
    hostname: REMOTE_HOST,
    port: REMOTE_PORT,
    path: req.url,
    method: req.method,
    headers: { ...req.headers, host: REMOTE_HOST },
    ...tlsOptions,
  };

  const proxy = https.request(options, (proxyRes) => {
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(res, { end: true });
  });

  proxy.on("error", (err) => {
    console.error(`[mtls-proxy] ${req.method} ${req.url} → error: ${err.message}`);
    if (!res.headersSent) {
      res.writeHead(502, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Proxy connection failed", detail: err.message }));
    }
  });

  req.pipe(proxy, { end: true });
});

server.listen(LOCAL_PORT, () => {
  console.log(`[mtls-proxy] Listening on http://localhost:${LOCAL_PORT}`);
  console.log(`[mtls-proxy] Forwarding to https://${REMOTE_HOST}:${REMOTE_PORT} with mTLS`);
  console.log(`[mtls-proxy] Certs from ${infraDir}`);
});

// Graceful shutdown
process.on("SIGINT", () => {
  console.log("\n[mtls-proxy] Shutting down...");
  server.close(() => process.exit(0));
});
process.on("SIGTERM", () => {
  server.close(() => process.exit(0));
});

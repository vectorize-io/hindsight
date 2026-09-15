#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// Next.js standalone uses HOSTNAME both to bind and as the origin its locale
// rewrite is checked against; a literal loopback address makes every page 307 to
// itself (#1926). Bind as `localhost` instead, pinning Node's DNS order so the
// address family stays the one requested. Mirrors docker/standalone/start-all.sh.
function resolveBindHost(hostname, nodeOptions) {
  const family = { '127.0.0.1': 'ipv4first', '::1': 'ipv6first', '[::1]': 'ipv6first' }[hostname];
  if (!family) return { hostname, nodeOptions };
  return {
    hostname: 'localhost',
    nodeOptions: [nodeOptions, `--dns-result-order=${family}`].filter(Boolean).join(' '),
  };
}

module.exports = { resolveBindHost };

if (require.main === module) {
  main();
}

function main() {
const args = process.argv.slice(2);

// Parse command line arguments
let port = process.env.PORT || 9999;
let hostname = process.env.HOSTNAME || '0.0.0.0';
let apiUrl = process.env.HINDSIGHT_CP_DATAPLANE_API_URL;

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--port' || args[i] === '-p') {
    port = args[++i];
  } else if (args[i] === '--hostname' || args[i] === '-H') {
    hostname = args[++i];
  } else if (args[i] === '--api-url' || args[i] === '-a') {
    apiUrl = args[++i];
  } else if (args[i] === '--help' || args[i] === '-h') {
    console.log(`
Hindsight Control Plane

Usage: hindsight-control-plane [options]

Options:
  -p, --port <port>       Port to listen on (default: 9999, env: PORT)
  -H, --hostname <host>   Hostname to bind to (default: 0.0.0.0, env: HOSTNAME)
  -a, --api-url <url>     Hindsight API URL (env: HINDSIGHT_CP_DATAPLANE_API_URL)
  -h, --help              Show this help message

Environment Variables:
  PORT                              Port to listen on
  HOSTNAME                          Hostname to bind to
  HINDSIGHT_CP_DATAPLANE_API_URL    URL of the Hindsight API server
`);
    process.exit(0);
  }
}

// Find the standalone server
const standaloneDir = path.join(__dirname, '..', 'standalone');
const serverPath = path.join(standaloneDir, 'server.js');

if (!fs.existsSync(serverPath)) {
  console.error('Error: Standalone server not found at', serverPath);
  console.error('This package may not have been built correctly.');
  process.exit(1);
}

// Set up environment
const bind = resolveBindHost(hostname, process.env.NODE_OPTIONS);
const env = {
  ...process.env,
  PORT: String(port),
  HOSTNAME: bind.hostname,
};
if (bind.nodeOptions) {
  env.NODE_OPTIONS = bind.nodeOptions;
}

if (apiUrl) {
  env.HINDSIGHT_CP_DATAPLANE_API_URL = apiUrl;
}

console.log(`Starting Hindsight Control Plane on http://${hostname}:${port}`);
if (apiUrl) {
  console.log(`API URL: ${apiUrl}`);
}

// Run the standalone server
const server = spawn('node', [serverPath], {
  cwd: standaloneDir,
  env,
  stdio: 'inherit',
});

server.on('error', (err) => {
  console.error('Failed to start server:', err.message);
  process.exit(1);
});

server.on('close', (code) => {
  process.exit(code || 0);
});

// Handle signals
process.on('SIGTERM', () => server.kill('SIGTERM'));
process.on('SIGINT', () => server.kill('SIGINT'));
}

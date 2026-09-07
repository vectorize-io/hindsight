#!/usr/bin/env bash
#
# Smoke test for the free-threaded API image (tag suffix -py3.14t).
#
#   ./docker/freethreaded-smoke.sh <image> [postgres-dsn]
#
# Separate from test-image.sh because this image is a different shape: it ships no
# local ML models, so embeddings and reranking must be remote, and test-image.sh
# assumes a provider API key. Everything here runs offline against a stub.
#
# What it proves, in order of what would actually go wrong:
#   1. the container starts and reaches /health/ready — i.e. it ran its migrations,
#      which on this image means the subprocess path, since psycopg2 would otherwise
#      take the GIL for the life of the process;
#   2. it serves a real retain and recall;
#   3. the SERVER process never lost free-threading. That is the failure this tag
#      exists to prevent, and it is silent: the image keeps working and merely
#      performs like the 3.11 one.
set -euo pipefail

IMAGE="${1:?usage: freethreaded-smoke.sh <image> [postgres-dsn]}"
DSN="${2:-postgresql://hindsight:hindsight@127.0.0.1:5432/hindsight}"
PORT="${SMOKE_PORT:-8899}"
TEI_PORT="${SMOKE_TEI_PORT:-8811}"
NAME="${SMOKE_CONTAINER_NAME:-hindsight-ft-smoke}"
TIMEOUT="${SMOKE_TEST_TIMEOUT:-180}"

cleanup() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  if [ -n "${TEI_PID:-}" ]; then kill "$TEI_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT

# A deterministic stand-in for a TEI embeddings server. The image ships no local
# models, so something has to answer /info and /embed; hashing the input keeps recall
# reproducible without downloading anything.
python3 - "$TEI_PORT" <<'PYEOF' &
import hashlib, json, math, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

DIM = 384


def vec(text):
    h = hashlib.blake2b(text.encode(), digest_size=64).digest()
    out = [((h[i % 64] << 8 | h[(i * 7 + 13) % 64]) / 65535.0) - 0.5 for i in range(DIM)]
    n = math.sqrt(sum(v * v for v in out)) or 1.0
    return [v / n for v in out]


class H(BaseHTTPRequestHandler):
    def _send(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"model_id": "smoke/deterministic", "max_input_length": 8192,
                    "model_dtype": "float32"})

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("content-length", 0)))
        inputs = json.loads(raw or b"{}").get("inputs", [])
        if isinstance(inputs, str):
            inputs = [inputs]
        self._send([vec(t) for t in inputs])

    def log_message(self, *a):
        pass


HTTPServer(("0.0.0.0", int(sys.argv[1])), H).serve_forever()
PYEOF
TEI_PID=$!
sleep 2

# Reach the runner from inside the container the same way on Linux and macOS:
# publish the port and add a host-gateway alias, rather than --network host (which
# on macOS does not give the container the host's network stack). A DSN pointing at
# the runner's loopback is rewritten to match.
CONTAINER_DSN="${DSN//127.0.0.1/host.docker.internal}"
CONTAINER_DSN="${CONTAINER_DSN//localhost/host.docker.internal}"

echo "==> starting $IMAGE"
docker run -d --name "$NAME" \
  --add-host=host.docker.internal:host-gateway \
  -p "${PORT}:8888" \
  -e HINDSIGHT_API_DATABASE_URL="$CONTAINER_DSN" \
  -e HINDSIGHT_API_LLM_PROVIDER=mock \
  -e HINDSIGHT_API_EMBEDDINGS_PROVIDER=tei \
  -e HINDSIGHT_API_EMBEDDINGS_TEI_URL="http://host.docker.internal:${TEI_PORT}" \
  -e HINDSIGHT_API_RERANKER_PROVIDER=rrf \
  -e HINDSIGHT_API_WORKER_ID=ft-smoke \
  "$IMAGE" >/dev/null

for _ in $(seq "$TIMEOUT"); do
  if curl -sf "http://127.0.0.1:${PORT}/health/ready" >/dev/null 2>&1; then break; fi
  sleep 1
done

if ! curl -sf "http://127.0.0.1:${PORT}/health/ready" >/dev/null 2>&1; then
  echo "FAIL: container never became ready"; docker logs "$NAME" | tail -50; exit 1
fi
echo "    ready (migrations applied)"

echo "==> retain + recall"
curl -sf -XPUT "http://127.0.0.1:${PORT}/v1/default/banks/smoke" \
  -H 'content-type: application/json' -d '{"description":"free-threaded smoke test"}' >/dev/null
curl -sf -XPOST "http://127.0.0.1:${PORT}/v1/default/banks/smoke/memories" \
  -H 'content-type: application/json' \
  -d '{"items":[{"content":"Hindsight runs on free-threaded CPython. Recall is I/O bound."}],"wait_for_completion":true}' >/dev/null

RESULTS=$(curl -sf -XPOST "http://127.0.0.1:${PORT}/v1/default/banks/smoke/memories/recall" \
  -H 'content-type: application/json' -d '{"query":"what does hindsight run on"}' \
  | python3 -c 'import sys,json; print(len(json.load(sys.stdin).get("results",[])))')
if [ "$RESULTS" -lt 1 ]; then
  echo "FAIL: recall returned no results"; docker logs "$NAME" | tail -50; exit 1
fi
echo "    recall returned $RESULTS result(s)"

# The point of the tag. A C extension without Py_MOD_GIL_NOT_USED re-enables the GIL
# for the whole process on import, and the only signal is a RuntimeWarning — so this
# has to be asserted rather than inferred from the container working.
echo "==> free-threading intact"
docker exec "$NAME" python -c "
import sys, sysconfig
assert sysconfig.get_config_var('Py_GIL_DISABLED'), 'not a free-threaded interpreter'
import hindsight_api.api.http
assert not sys._is_gil_enabled(), 'importing the API re-enabled the GIL'
print('    free-threaded:', sys.version.split()[0])
"

# The migration child takes the GIL on purpose and silences its own warning, so any
# occurrence here means the SERVER process lost free-threading.
if docker logs "$NAME" 2>&1 | grep -q "global interpreter lock"; then
  echo "FAIL: a GIL re-enable warning reached the server log"
  docker logs "$NAME" 2>&1 | grep -B2 -A2 "global interpreter lock" | head -20
  exit 1
fi
echo "    no GIL re-enable warnings"

echo "PASS"

#!/usr/bin/env bash
# Start a local Oracle 23ai Free container that mirrors what CI uses
# (`test-api-oracle`, `test-{python,typescript}-client-oracle`), bootstrap
# the HINDSIGHT_TEST user, and print the connection URL.
#
# Usage:
#   ./scripts/dev/start-oracle.sh           # start (idempotent) and print URL
#   ./scripts/dev/start-oracle.sh --reset   # drop and recreate the test user
#
# Stop with: ./scripts/dev/stop-oracle.sh

set -euo pipefail

CONTAINER_NAME="hindsight-oracle"
IMAGE="container-registry.oracle.com/database/free:latest"
PORT=1521
ORACLE_PWD="oracle"
TEST_USER="hindsight_test"
TEST_PASS="hindsight_test"

want_reset=0
for arg in "$@"; do
    case "$arg" in
        --reset) want_reset=1 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

# 1) Ensure the container is running.
if [ -z "$(docker ps -q -f name="^${CONTAINER_NAME}$")" ]; then
    if [ -n "$(docker ps -aq -f name="^${CONTAINER_NAME}$")" ]; then
        echo "→ Removing stopped container ${CONTAINER_NAME}"
        docker rm -f "${CONTAINER_NAME}" >/dev/null
    fi
    echo "→ Starting ${CONTAINER_NAME} (${IMAGE})"
    docker run -d \
        --name "${CONTAINER_NAME}" \
        -p ${PORT}:1521 \
        -e ORACLE_PWD="${ORACLE_PWD}" \
        "${IMAGE}" >/dev/null
fi

# 2) Wait until SQL*Plus inside the container can SELECT 1 FROM DUAL.
#    Same readiness probe CI uses for the service container's health-cmd.
echo "→ Waiting for FREEPDB1 to accept connections (~60-120s on a cold start) ..."
for i in $(seq 1 120); do
    if docker exec "${CONTAINER_NAME}" \
        bash -c "echo 'SELECT 1 FROM DUAL;' | sqlplus -s system/${ORACLE_PWD}@localhost:1521/FREEPDB1" \
        >/dev/null 2>&1; then
        echo "    ready after ${i}s"
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "FREEPDB1 not ready after 120s — recent container logs:" >&2
        docker logs --tail 60 "${CONTAINER_NAME}" >&2 || true
        exit 1
    fi
    sleep 1
done

# 3) Bootstrap (or reset) the HINDSIGHT_TEST user. Mirrors the CI step.
if [ "$want_reset" = "1" ]; then
    echo "→ --reset: dropping existing ${TEST_USER}"
    docker exec -i "${CONTAINER_NAME}" \
        sqlplus -s system/${ORACLE_PWD}@localhost:1521/FREEPDB1 <<SQL >/dev/null
DROP USER ${TEST_USER} CASCADE;
EXIT;
SQL
fi

echo "→ Ensuring ${TEST_USER} exists with ASSM tablespace (idempotent)"
# CREATE TABLESPACE / CREATE USER / GRANT statements are mirrored from the
# CI workflow steps (test-api-oracle, test-python-client-oracle, etc.).
# We tolerate "already exists" so the script can be re-run safely.
docker exec -i "${CONTAINER_NAME}" \
    sqlplus -s system/${ORACLE_PWD}@localhost:1521/FREEPDB1 <<SQL >/dev/null
WHENEVER SQLERROR EXIT FAILURE
SET ECHO OFF
SET FEEDBACK OFF

DECLARE
    e_already_exists EXCEPTION;
    PRAGMA EXCEPTION_INIT(e_already_exists, -1543);
BEGIN
    EXECUTE IMMEDIATE 'CREATE TABLESPACE hindsight_ts
        DATAFILE ''hindsight_ts.dbf'' SIZE 200M AUTOEXTEND ON NEXT 50M
        EXTENT MANAGEMENT LOCAL
        SEGMENT SPACE MANAGEMENT AUTO';
EXCEPTION WHEN e_already_exists THEN NULL;
END;
/

DECLARE
    e_user_exists EXCEPTION;
    PRAGMA EXCEPTION_INIT(e_user_exists, -1920);
BEGIN
    EXECUTE IMMEDIATE 'CREATE USER ${TEST_USER} IDENTIFIED BY ${TEST_PASS}
        DEFAULT TABLESPACE hindsight_ts
        TEMPORARY TABLESPACE temp
        QUOTA UNLIMITED ON hindsight_ts';
EXCEPTION WHEN e_user_exists THEN NULL;
END;
/

GRANT CONNECT, RESOURCE, CREATE TABLE, CREATE SEQUENCE, CREATE VIEW, CREATE PROCEDURE TO ${TEST_USER};
GRANT CTXAPP TO ${TEST_USER};
EXIT;
SQL

URL="oracle+oracledb://${TEST_USER}:${TEST_PASS}@localhost:${PORT}/FREEPDB1"
cat <<EOF

✓ Oracle is ready.

  Container : ${CONTAINER_NAME}
  URL       : ${URL}

  export HINDSIGHT_API_DATABASE_BACKEND=oracle
  export HINDSIGHT_API_DATABASE_URL='${URL}'

  Migrations : uv run --directory hindsight-api-slim hindsight-admin run-db-migration
  Stop       : ./scripts/dev/stop-oracle.sh
EOF

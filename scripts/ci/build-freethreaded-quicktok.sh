#!/usr/bin/env bash
# Build a free-threaded (cp3XXt) wheel of quicktok-v1 and install it.
#
# quicktok is on the recall hot path (token budgeting) and publishes no cp3XXt
# wheel. Building the published sdist as-is is NOT enough: the resulting extension
# would not declare `Py_MOD_GIL_NOT_USED`, so importing it re-enables the GIL for
# the whole process and free-threading is silently lost.
#
# Two changes are needed, both upstreamable:
#   1. pybind11 >= 2.13 -- the first release with free-threading support.
#   2. `py::mod_gil_not_used()` on the PYBIND11_MODULE declaration.
#
# The C++ core is already written for concurrent use: it releases the GIL around
# every entry point, keeps scratch buffers `thread_local`, and guards its shared
# validity memo with relaxed atomics that are self-validating on a torn read.
set -euo pipefail

QUICKTOK_REF="${QUICKTOK_REF:-main}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

git clone --depth 1 --branch "$QUICKTOK_REF" https://github.com/dmatth1/quicktok.git "$WORKDIR/quicktok"
cd "$WORKDIR/quicktok"

sed -i.bak 's/"pybind11>=2\.12"/"pybind11>=2.13"/' pyproject.toml
grep -q '"pybind11>=2.13"' pyproject.toml || { echo "failed to raise the pybind11 floor"; exit 1; }

"$PYTHON_BIN" - <<'PY'
import io
p = "python/src/_quicktok.cpp"
s = io.open(p).read()
if "mod_gil_not_used" in s:
    print("already declares mod_gil_not_used")
else:
    old = "PYBIND11_MODULE(_quicktok, m) {"
    new = "PYBIND11_MODULE(_quicktok, m, py::mod_gil_not_used()) {"
    if old not in s:
        raise SystemExit(f"PYBIND11_MODULE declaration not found in {p}")
    io.open(p, "w").write(s.replace(old, new, 1))
    print("declared the module free-threading safe")
PY

uv build --wheel --python "$PYTHON_BIN" -o "$WORKDIR/dist"
uv pip install --python "$PYTHON_BIN" --reinstall "$WORKDIR"/dist/*.whl

"$PYTHON_BIN" - <<'PY'
import sys, quicktok
assert not sys._is_gil_enabled(), "quicktok re-enabled the GIL; the wheel is not free-threaded"
assert quicktok.get_encoding("o200k_base").count("hello world") > 0
print("quicktok imports GIL-free and tokenizes:", quicktok.__file__)
PY

#!/usr/bin/env bash
# Build hindsight-api-slim wheels with per-platform mlx metadata.
#
# mlx ships wheels only for macOS arm64 and Linux glibc >= 2.35
# (manylinux_2_35). The py3-none-any wheel gates mlx to darwin/arm64 so
# Intel macs, musl and older-glibc Linux can resolve [all]/[local-ml];
# two manylinux_2_35-tagged wheel variants re-enable mlx on modern
# Linux. pip prefers the most specific compatible tag, and
# manylinux_2_35 tags are only selected on glibc >= 2.35, where mlx has
# wheels — so every platform keeps exactly the capabilities it has
# today, while previously-unresolvable platforms now install cleanly.
#
# Usage: build-api-slim-wheels.sh <package-dir> <out-dir>
#   e.g. build-api-slim-wheels.sh hindsight-api-slim hindsight-api-slim/dist
#
# Outputs:
#   hindsight_api_slim-<ver>-py3-none-any.whl                     (mlx on darwin/arm64)
#   hindsight_api_slim-<ver>-py3-none-manylinux_2_35_x86_64.whl   (mlx on linux)
#   hindsight_api_slim-<ver>-py3-none-manylinux_2_35_aarch64.whl  (mlx on linux)
#   hindsight_api_slim-<ver>.tar.gz
set -euo pipefail

pkg_dir="${1:?usage: build-api-slim-wheels.sh <package-dir> <out-dir>}"
out_dir="${2:?usage: build-api-slim-wheels.sh <package-dir> <out-dir>}"
pkg_dir=$(cd "$pkg_dir" && pwd)
mkdir -p "$out_dir"
out_dir=$(cd "$out_dir" && pwd)

# 1. sdist + base pure wheel from the pristine pyproject.toml
#    (mlx gated to darwin/arm64)
(cd "$pkg_dir" && uv build --out-dir "$out_dir")

# 2. linux-metadata wheel, retagged for both manylinux_2_35 architectures
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
tar --exclude=./dist --exclude=./.venv -cf - -C "$pkg_dir" . | tar -xf - -C "$tmp"
sed -i "s|sys_platform == 'darwin' and platform_machine == 'arm64'|sys_platform == 'linux'|" \
  "$tmp/pyproject.toml"
grep -q "sys_platform == 'linux'" "$tmp/pyproject.toml"  # sed must have matched

(cd "$tmp" && uv build --wheel --out-dir "$tmp/linux")

for arch in x86_64 aarch64; do
  (cd "$tmp/linux" && uv run --no-project --with wheel python -m wheel tags \
    --platform-tag "manylinux_2_35_${arch}" \
    hindsight_api_slim-*-py3-none-any.whl)
done
mv "$tmp"/linux/hindsight_api_slim-*-py3-none-manylinux_2_35_*.whl "$out_dir"/

echo "Built wheels:"
ls -1 "$out_dir"/*.whl

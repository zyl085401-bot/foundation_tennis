#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REQUIREMENTS="${SCRIPT_DIR}/requirements.jetson.mros.txt"
MROS_WHEEL="${REPO_ROOT}/mrospy/dist/aarch64/mros-2.3.1-py3-none-any.whl"
OUTPUT_DIR="${SCRIPT_DIR}/jetson_offline_jp61/mros"
PYTHON_BIN="${PYTHON_BIN:-python3}"

for required_file in "${REQUIREMENTS}" "${MROS_WHEEL}"; do
  if [[ ! -s "${required_file}" ]]; then
    echo "ERROR: Missing required file: ${required_file}" >&2
    exit 1
  fi
done

if ! "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} does not provide pip." >&2
  exit 1
fi

TEMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

# Resolve binary dependencies for the Jetson Python 3.10 ARM64 runtime while
# running this preparation step on an x86_64 development machine.
"${PYTHON_BIN}" -m pip download \
  --requirement "${REQUIREMENTS}" \
  --dest "${TEMP_DIR}" \
  --only-binary=:all: \
  --platform manylinux2014_aarch64 \
  --platform manylinux_2_17_aarch64 \
  --implementation cp \
  --python-version 310 \
  --abi cp310

cp "${MROS_WHEEL}" "${TEMP_DIR}/"

python_gnupg_count="$(find "${TEMP_DIR}" -maxdepth 1 -type f -iname 'python_gnupg-*.whl' | wc -l)"
lz4_count="$(find "${TEMP_DIR}" -maxdepth 1 -type f -iname 'lz4-*aarch64.whl' | wc -l)"
crypto_count="$(find "${TEMP_DIR}" -maxdepth 1 -type f -iname 'pycryptodomex-*aarch64.whl' | wc -l)"
mros_count="$(find "${TEMP_DIR}" -maxdepth 1 -type f -name 'mros-2.3.1-py3-none-any.whl' | wc -l)"
for item in \
  "python-gnupg:${python_gnupg_count}" \
  "lz4 ARM64:${lz4_count}" \
  "pycryptodomex ARM64:${crypto_count}" \
  "mROS:${mros_count}"; do
  name="${item%%:*}"
  count="${item##*:}"
  if [[ "${count}" != 1 ]]; then
    echo "ERROR: Expected exactly one ${name} wheel, found ${count}." >&2
    exit 1
  fi
done

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
cp "${TEMP_DIR}"/*.whl "${OUTPUT_DIR}/"
(
  cd "${OUTPUT_DIR}"
  sha256sum ./*.whl > SHA256SUMS
)

cat <<EOF
Prepared Jetson mROS offline bundle:
  output: ${OUTPUT_DIR}
  target: CPython 3.10 / Linux aarch64
EOF
find "${OUTPUT_DIR}" -maxdepth 1 -type f -printf '  %f\n' | sort

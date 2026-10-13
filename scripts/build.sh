#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-all}"
ARCH="${ARCH:-linux_a35_uclibc_32bit}"
RNE_RT_ROOT="${RNE_RT_ROOT:-${ROOT_DIR}/RNE-RT-Lib}"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build/${ARCH}}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-}"

usage() {
    echo "Usage: $0 [all|resnet18|yolov5s]"
    echo "Environment: ARCH, RNE_RT_ROOT, TOOLCHAIN_ROOT, BUILD_DIR, BUILD_TYPE, USE_DYNAMIC_LIB, JOBS"
    echo "ARCH: linux_a35_uclibc_32bit (default) | linux_a53_uclibc"
}

fail() {
    echo "Error: $*" >&2
    exit 1
}

case "${MODEL}" in
    all|resnet18|yolov5s) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; fail "unsupported model: ${MODEL}" ;;
esac

PARALLEL_ARGS=(--parallel)
if [[ -n "${JOBS:-}" ]]; then
    PARALLEL_ARGS+=("${JOBS}")
fi

if [[ ! -d "${RNE_RT_ROOT}/include" || ! -d "${RNE_RT_ROOT}/open_source" ]]; then
    fail "RNE-RT-Lib not found at ${RNE_RT_ROOT}; set RNE_RT_ROOT to its path"
fi
if [[ -n "${TOOLCHAIN_ROOT}" ]]; then
    [[ -d "${TOOLCHAIN_ROOT}/bin" ]] || fail "toolchain bin directory not found: ${TOOLCHAIN_ROOT}/bin"
    export PATH="${TOOLCHAIN_ROOT}/bin:${PATH}"
fi

case "${USE_DYNAMIC_LIB:-OFF}" in
    1|ON|on|true|TRUE) USE_DYNAMIC_LIB=ON ;;
    0|OFF|off|false|FALSE|'') USE_DYNAMIC_LIB=OFF ;;
    *) fail "USE_DYNAMIC_LIB must be ON/OFF or 1/0" ;;
esac

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DARCH="${ARCH}" \
    -DRNE_RT_ROOT="${RNE_RT_ROOT}" \
    -DUSE_DYNAMIC_LIB="${USE_DYNAMIC_LIB}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

case "${MODEL}" in
    all)
        cmake --build "${BUILD_DIR}" "${PARALLEL_ARGS[@]}"
        ;;
    resnet18)
        cmake --build "${BUILD_DIR}" --target resnet18_demo.elf "${PARALLEL_ARGS[@]}"
        ;;
    yolov5s)
        cmake --build "${BUILD_DIR}" --target yolov5s_demo.elf "${PARALLEL_ARGS[@]}"
        ;;
esac

echo "Build complete: ${BUILD_DIR}/bin"

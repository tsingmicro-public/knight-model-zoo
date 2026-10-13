#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-all}"
ACTION="${2:-simulate}"
INPUT_IMAGE="${3:-}"
KNIGHT_BIN="${KNIGHT_BIN:-Knight}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CHIP="${CHIP:-TX5326DV500}"

usage() {
    echo "Usage: $0 [all|resnet18|yolov5s] [simulate|profiling|all] [input-image]"
    echo "Environment: KNIGHT_BIN, PYTHON_BIN, CHIP, MODEL_FILE"
    echo "Run inside the Knight container with the repository mounted at /data."
}

if [[ "${MODEL}" == "-h" || "${MODEL}" == "--help" ]]; then
    usage
    exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

fail() {
    echo "Error: $*" >&2
    exit 1
}

case "${MODEL}" in
    all) MODELS=(resnet18 yolov5s) ;;
    resnet18|yolov5s) MODELS=("${MODEL}") ;;
    *) usage >&2; fail "unsupported model: ${MODEL}" ;;
esac
case "${ACTION}" in
    simulate|profiling|all) ;;
    *) usage >&2; fail "unsupported action: ${ACTION}" ;;
esac

[[ "${ROOT_DIR}" == "/data" ]] || fail "run inside the Knight container with this repository mounted at /data"
command -v "${KNIGHT_BIN}" >/dev/null 2>&1 || fail "Knight command not found: ${KNIGHT_BIN}"

run_simulation() {
    local model="$1"
    local model_dir="${ROOT_DIR}/models/${model}"
    local output_dir="${model_dir}/output"
    local model_file="${MODEL_FILE:-${model_dir}/${model}_quantize_r.tsmodel}"
    local image="${INPUT_IMAGE}"

    if [[ -z "${image}" ]]; then
        if [[ "${model}" == "resnet18" ]]; then
            image="${ROOT_DIR}/dataset/imagenet_random_10/ILSVRC2012_val_00000056_n02930766.JPEG"
        else
            image="${ROOT_DIR}/dataset/bus.jpg"
        fi
    elif [[ "${image}" != /* ]]; then
        image="${ROOT_DIR}/${image}"
    fi

    [[ -f "${model_file}" ]] || fail "model file not found: ${model_file}; set MODEL_FILE if using a different tsmodel"
    [[ -f "${image}" ]] || fail "input image not found: ${image}"

    echo "Simulating ${model} on ${image}..."
    if [[ "${model}" == "resnet18" ]]; then
        local pre_dir="${output_dir}/sim_input"
        local sim_dir="${output_dir}/sim"
        "${PYTHON_BIN}" "${model_dir}/make_image_input.py" --input "${image}" --outpath "${pre_dir}"
        "${KNIGHT_BIN}" --chip "${CHIP}" run \
            --input "${pre_dir}/model_input.bin" \
            --model "${model_file}" --format nchw --save-dir "${sim_dir}"
        "${PYTHON_BIN}" "${model_dir}/post_process.py" --input "${sim_dir}"
    else
        local pre_dir="${output_dir}/sim_input"
        local sim_dir="${output_dir}/sim"
        "${PYTHON_BIN}" "${model_dir}/yolov5_pre_post.py" \
            --input "${image}" --outpath "${pre_dir}" --pre_processing
        "${KNIGHT_BIN}" --chip "${CHIP}" run \
            --input "${pre_dir}/model_input.bin" \
            --model "${model_file}" --save-dir "${sim_dir}" -fmt nchw
        "${PYTHON_BIN}" "${model_dir}/yolov5_pre_post.py" \
            --input "${sim_dir}" --outpath "${sim_dir}" \
            --post_processing --data "${image}"
    fi
}

run_profiling() {
    local model="$1"
    local model_dir="${ROOT_DIR}/models/${model}"
    local model_file="${MODEL_FILE:-${model_dir}/${model}_quantize_r.tsmodel}"
    [[ -f "${model_file}" ]] || fail "model file not found: ${model_file}; set MODEL_FILE if using a different tsmodel"
    echo "Profiling ${model}..."
    "${KNIGHT_BIN}" profiling --chip "${CHIP}" --model "${model_file}" \
        --save-dir "${model_dir}/output/profiling"
}

for model in "${MODELS[@]}"; do
    case "${ACTION}" in
        simulate) run_simulation "${model}" ;;
        profiling) run_profiling "${model}" ;;
        all)
            run_simulation "${model}"
            run_profiling "${model}"
            ;;
    esac
done

echo "Model test workflow complete."

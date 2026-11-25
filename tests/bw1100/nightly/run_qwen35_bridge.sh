#!/usr/bin/env bash
# Nightly Qwen3.5-9B text pretrain entry for the Megatron-bridge-das framework.
#
# Megatron-bridge-das is a separate repository, checked out by the workflow at
# BRIDGE_ROOT. Its example script is the documented single-node entry point and
# resolves the bridge repo root from $0 before sourcing env/env.sh, so this
# wrapper delegates to it instead of duplicating that setup (sourcing env.sh
# from here would compute MEGATRON_PATH from this script's path and break the
# PYTHONPATH it exports).
#
# env:
#   BRIDGE_ROOT   Megatron-bridge-das checkout (the workflow pins the commit)
#   LOCAL_HF_PATH Qwen3.5-9B HF snapshot; the recipe reads config.json and the
#                 tokenizer from here, and loads no weights (mock dataset)
#   BRIDGE_RECIPE recipe name; defaults to the 8-GPU BW1100 9B config
#   TRAIN_ITERS   training iterations; defaults to 50. The bridge example
#                 hardcodes 50, so this wrapper runs a patched copy.
set -euo pipefail

bridge_root="${BRIDGE_ROOT:-}"
[[ -n "${bridge_root}" ]] || { echo "ERROR: BRIDGE_ROOT is required" >&2; exit 1; }
train_script="${bridge_root}/examples/qwen/train_qwen35_text_pretrain.sh"
[[ -f "${train_script}" ]] || {
    echo "ERROR: ${train_script} not found; BRIDGE_ROOT must point at a Megatron-bridge-das checkout" >&2
    exit 1
}

local_hf_path="${LOCAL_HF_PATH:-}"
[[ -n "${local_hf_path}" ]] || { echo "ERROR: LOCAL_HF_PATH is required" >&2; exit 1; }
[[ -f "${local_hf_path}/config.json" ]] || {
    echo "ERROR: ${local_hf_path}/config.json not found; LOCAL_HF_PATH must be a Qwen3.5 HF snapshot" >&2
    exit 1
}
export LOCAL_HF_PATH="${local_hf_path}"

recipe="${BRIDGE_RECIPE:-qwen35_text_9b_pretrain_8gpu_bw1100_bf16_config}"
train_iters="${TRAIN_ITERS:-50}"
[[ "${train_iters}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: TRAIN_ITERS must be a positive integer, got '${train_iters}'" >&2
    exit 1
}

# The bridge example hardcodes TRAIN_ITERS=50, so run a patched copy. Keep the
# copy in the example's own directory: the example derives CURRENT_DIR and
# MEGATRON_PATH from $0. Never edit the tracked checkout.
train_dir="$(cd "$(dirname "${train_script}")" && pwd)"
patched_script="${train_dir}/.train_qwen35_text_pretrain.ci.$$.sh"
finish() {
    rc=$?
    trap - EXIT
    rm -f -- "${patched_script}" || true
    exit "${rc}"
}
trap finish EXIT

cp -- "${train_script}" "${patched_script}"
if grep -qx 'TRAIN_ITERS=50' "${patched_script}"; then
    sed -i "s/^TRAIN_ITERS=50$/TRAIN_ITERS=${train_iters}/" "${patched_script}"
    grep -qx "TRAIN_ITERS=${train_iters}" "${patched_script}" || {
        echo "ERROR: cannot override TRAIN_ITERS in ${train_script}" >&2
        exit 1
    }
elif grep -Eq '^TRAIN_ITERS=\$\{TRAIN_ITERS:-[0-9]+\}$' "${patched_script}"; then
    export TRAIN_ITERS="${train_iters}"
else
    echo "ERROR: unsupported TRAIN_ITERS definition in ${train_script}" >&2
    exit 1
fi

echo "=== bridge nightly launch ==="
echo "bridge=${bridge_root}"
echo "recipe=${recipe}"
echo "iters=${train_iters}"
echo "model=${LOCAL_HF_PATH}"

bash "${patched_script}" "${recipe}" train

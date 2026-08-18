#!/usr/bin/env bash
# Workspace preparation (source this script):
#   1. register submodule safe.directory entries
#   2. sync + init top-level submodules (non-recursive; see below)
#   3. verify pinned submodule SHAs
#   4. export PYTHONPATH (repo root + 3rdparty submodules)
set -euo pipefail

prepare_das_workspace() {
    local script_dir repo_root
    local -a python_paths

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(cd "${script_dir}/../../.." && pwd)"

    for safe_path in \
        "${repo_root}" \
        "${repo_root}/3rdparty/Megatron-LM" \
        "${repo_root}/3rdparty/Megatron-Energon" \
        "${repo_root}/3rdparty/Megatron-Bridge"; do
        if ! git config --global --get-all safe.directory 2>/dev/null |
            grep -Fqx -- "${safe_path}"; then
            git config --global --add safe.directory "${safe_path}"
        fi
    done

    # Top-level submodules only (non-recursive): Megatron-Bridge contains the
    # nested 3rdparty/Megatron-LM submodule whose gitlink cannot be resolved,
    # and das code does not use Bridge. Switch to --recursive only if needed.
    git -C "${repo_root}" submodule sync
    git -C "${repo_root}" submodule update --init

    # Pinned submodule verification (gitlink + checkout), guards against drift.
    python3 "${script_dir}/verify_submodules.py" --repo-root "${repo_root}"

    # Optional: prebuilt hcu-megatron wheel (provides compiled ops such as
    # fused_weight_gradient_mlp_cuda). DAS_HCU_MEGATRON_WHEEL accepts any
    # pip-installable path or URL. Without it, sitecustomize.py injects an
    # import stub (calls raise NotImplementedError).
    if [[ -n "${DAS_HCU_MEGATRON_WHEEL:-}" ]]; then
        echo "Installing hcu-megatron wheel: ${DAS_HCU_MEGATRON_WHEEL}"
        pip install "${DAS_HCU_MEGATRON_WHEEL}"
    fi

    python_paths=(
        "${repo_root}"
        "${repo_root}/3rdparty/Megatron-LM"
        "${repo_root}/3rdparty/Megatron-Energon"
        "${repo_root}/3rdparty/Megatron-Bridge"
        "${script_dir}"  # sitecustomize.py: python3.10 typing.override shim
    )
    joined_python_path="$(IFS=:; printf '%s' "${python_paths[*]}")"
    export PYTHONPATH="${joined_python_path}${PYTHONPATH:+:${PYTHONPATH}}"

    echo "DAS workspace prepared at ${repo_root}"
}

prepare_das_workspace

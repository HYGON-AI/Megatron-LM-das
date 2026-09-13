#!/usr/bin/env bash
# Workspace preparation (source this script):
#   1. register submodule safe.directory entries
#   2. sync + init top-level submodules (non-recursive; see below)
#   3. verify pinned submodule SHAs
#   4. export PYTHONPATH (repo root + 3rdparty submodules)
set -euo pipefail

reset_das_submodule() {
    local repo_root="$1" git_dir="$2" submodule_path="$3"
    local submodule_worktree submodule_git_dir

    submodule_worktree="${repo_root}/${submodule_path}"
    submodule_git_dir="${git_dir}/modules/${submodule_path}"
    case "${submodule_worktree}" in
        "${repo_root}"/3rdparty/*) ;;
        *) echo "Refusing to reset unsafe submodule worktree" >&2; return 1 ;;
    esac
    case "${submodule_git_dir}" in
        "${git_dir}"/modules/3rdparty/*) ;;
        *) echo "Refusing to reset unsafe submodule gitdir" >&2; return 1 ;;
    esac

    echo "Resetting CI submodule: ${submodule_path}"
    git -C "${repo_root}" submodule deinit -f -- "${submodule_path}" >/dev/null 2>&1 || true
    rm -rf -- "${submodule_worktree}" "${submodule_git_dir}"
}

prepare_das_workspace() {
    local script_dir repo_root git_dir submodule_path submodule_prefix
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
    # and repo code does not use Bridge. Switch to --recursive only if needed.
    git -C "${repo_root}" submodule sync

    # A persistent runner can retain an interrupted submodule checkout. Git
    # treats its directory as initialized even when it has no valid HEAD, so
    # `submodule update --init` cannot repair it without removing its stale
    # worktree and gitdir first.
    local -a top_level_submodules
    top_level_submodules=(
        "3rdparty/Megatron-LM"
        "3rdparty/Megatron-Energon"
        "3rdparty/Megatron-Bridge"
    )
    git_dir="$(git -C "${repo_root}" rev-parse --absolute-git-dir)"
    for submodule_path in "${top_level_submodules[@]}"; do
        if [[ -e "${repo_root}/${submodule_path}" ]]; then
            # Without its .git file, Git falls back to the parent worktree.
            # Require this directory to be the Git worktree before checking HEAD.
            submodule_prefix="$(git -C "${repo_root}/${submodule_path}" rev-parse --show-prefix 2>/dev/null || true)"
            if [[ -n "${submodule_prefix}" ]] ||
                ! git -C "${repo_root}/${submodule_path}" rev-parse --verify -q HEAD >/dev/null 2>&1; then
                reset_das_submodule "${repo_root}" "${git_dir}" "${submodule_path}"
            fi
        fi
    done

    if ! git -C "${repo_root}" -c http.version=HTTP/1.1 submodule update --init --force; then
        # A valid HEAD does not prove the pinned object is available, so retry
        # once from clean top-level submodule state after an update failure.
        echo "Reinitializing CI submodules after update failure"
        for submodule_path in "${top_level_submodules[@]}"; do
            reset_das_submodule "${repo_root}" "${git_dir}" "${submodule_path}"
        done
        git -C "${repo_root}" -c http.version=HTTP/1.1 submodule update --init --force
    fi

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

    # Energon and Bridge are src-layout: the megatron.* packages live under
    # src/, so the repo root alone does not make megatron.bridge importable.
    python_paths=(
        "${repo_root}"
        "${repo_root}/3rdparty/Megatron-LM"
        "${repo_root}/3rdparty/Megatron-Energon/src"
        "${repo_root}/3rdparty/Megatron-Bridge/src"
        "${script_dir}"  # sitecustomize.py: python3.10 typing.override shim
    )
    joined_python_path="$(IFS=:; printf '%s' "${python_paths[*]}")"
    export PYTHONPATH="${joined_python_path}${PYTHONPATH:+:${PYTHONPATH}}"

    echo "HCU CI workspace prepared at ${repo_root}"
}

prepare_das_workspace

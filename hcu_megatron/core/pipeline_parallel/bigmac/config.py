# Copyright (c) 2026 Dots-Infra
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Some of this code was adopted from https://github.com/Dots-Infra/BigMac.git

# BigMac switches and Qwen3-VL launch dimensions.
import os
import argparse
import json
import sys
from pathlib import Path


def env_flag(name, fallback=None):
    value = os.getenv(name)
    if value is None and fallback:
        value = os.getenv(fallback)
    return (value or "").lower() in ("1", "true", "yes", "on")


def executor_enabled():
    return env_flag("MEGATRON_USE_PP_EXECUTOR", "USE_PP_EXECUTOR")


def encoder_enabled():
    return env_flag("MEGATRON_PP_EXECUTOR_WITH_VE", "USE_BIGMAC_STYLE")


def trace_options():
    iteration = os.getenv("MEGATRON_PP_EXECUTOR_TRACE_ITER")
    rank = os.getenv("MEGATRON_PP_EXECUTOR_TRACE_DP_RANK", "0")
    return dict(
        trace_enabled=env_flag("MEGATRON_PP_EXECUTOR_TRACE"),
        trace_dir=os.getenv("MEGATRON_PP_EXECUTOR_TRACE_DIR", "/tmp/pp_traces"),
        trace_iter_id=int(iteration) if iteration else None,
        trace_dp_rank=None if rank.lower() == "all" else int(rank),
    )


def _positive_int(config, name):
    value = config.get(name)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _pipeline_arguments(layers, pipeline_parallel_size, first_pipeline_num_layers):
    if pipeline_parallel_size < 1:
        raise ValueError("PP must be a positive integer")
    if first_pipeline_num_layers is None:
        if layers % pipeline_parallel_size:
            raise ValueError(
                f"num_hidden_layers={layers} is not divisible by PP={pipeline_parallel_size}; "
                "choose a compatible PP or set MP_PP0_LAYERS for an uneven first stage"
            )
        return []
    if pipeline_parallel_size == 1:
        raise ValueError("MP_PP0_LAYERS requires PP > 1")
    remaining = layers - first_pipeline_num_layers
    if first_pipeline_num_layers < 1 or remaining < pipeline_parallel_size - 1:
        raise ValueError("MP_PP0_LAYERS must leave at least one layer on every pipeline stage")
    if remaining % (pipeline_parallel_size - 1):
        raise ValueError("(num_hidden_layers - MP_PP0_LAYERS) must be divisible by (PP - 1)")
    return ["--decoder-first-pipeline-num-layers", str(first_pipeline_num_layers)]


def model_arguments(model_dir, pipeline_parallel_size=1, first_pipeline_num_layers=None):
    config_path = Path(model_dir).expanduser() / "config.json"
    with config_path.open(encoding="utf-8") as source:
        config = json.load(source)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    text = config.get("text_config")
    vision = config.get("vision_config")
    if not isinstance(text, dict) or not isinstance(vision, dict):
        raise ValueError(f"{config_path} must contain Qwen3-VL text_config and vision_config")

    layers = _positive_int(text, "num_hidden_layers")
    hidden = _positive_int(text, "hidden_size")
    is_moe = (
        config.get("model_type") == "qwen3_vl_moe"
        or text.get("model_type") == "qwen3_vl_moe_text"
        or text.get("num_experts") is not None
    )
    # All-expert Qwen3-VL configs can omit the dense MLP size. Do not fabricate
    # a dense architecture override; Bridge retains its own fallback in that case.
    ffn = None
    if not is_moe or text.get("intermediate_size") is not None:
        ffn = _positive_int(text, "intermediate_size")
    heads = _positive_int(text, "num_attention_heads")
    kv_heads = _positive_int(text, "num_key_value_heads")
    if text.get("head_dim") is not None:
        head_dim = _positive_int(text, "head_dim")
    else:
        if hidden % heads:
            raise ValueError("text_config must specify head_dim when hidden_size is not divisible by heads")
        head_dim = hidden // heads
    if heads % kv_heads:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    vision_hidden = _positive_int(vision, "out_hidden_size")
    if vision_hidden != hidden:
        raise ValueError(
            f"{config_path}: vision out_hidden_size={vision_hidden} differs from text hidden_size={hidden}"
        )

    # DAS overlays explicit CLI dimensions onto the HF-derived Bridge provider.
    # They must describe the same model as the untouched HF vision config.
    arguments = [
        "--num-layers", str(layers),
        "--hidden-size", str(hidden),
        "--num-attention-heads", str(heads),
        "--kv-channels", str(head_dim),
        "--group-query-attention", "--num-query-groups", str(kv_heads),
    ]
    if ffn is not None:
        arguments.extend(["--ffn-hidden-size", str(ffn)])
    tied_embeddings = config.get("tie_word_embeddings", False)
    if not isinstance(tied_embeddings, bool):
        raise ValueError("tie_word_embeddings must be a boolean")
    if not tied_embeddings:
        arguments.append("--untie-embeddings-and-output-weights")

    moe_description = ""
    if is_moe:
        experts = _positive_int(text, "num_experts")
        expert_ffn = _positive_int(text, "moe_intermediate_size")
        topk = _positive_int(text, "num_experts_per_tok")
        if topk > experts:
            raise ValueError("num_experts_per_tok must not exceed num_experts")
        # The native Qwen3-VL MoE provider builds one MoE spec for every layer,
        # as used by the reference A3B/A22B models. Do not silently misbuild a
        # custom dense/MoE mixture that this provider cannot represent.
        if text.get("mlp_only_layers", []) or text.get("decoder_sparse_step", 1) != 1:
            raise ValueError("The current Qwen3-VL Bridge requires all decoder layers to use MoE")
        arguments.extend([
            "--num-experts", str(experts),
            "--moe-ffn-hidden-size", str(expert_ffn),
            "--moe-router-topk", str(topk),
            "--moe-layer-freq", "1",
        ])
        moe_description = f", experts={experts}, expert_ffn={expert_ffn}, topk={topk}"
    arguments.extend(_pipeline_arguments(layers, pipeline_parallel_size, first_pipeline_num_layers))
    print(
        f"[Qwen3-VL config] {config_path}: layers={layers}, hidden={hidden}, "
        f"ffn={ffn}, heads={heads}, kv_heads={kv_heads}, head_dim={head_dim}, "
        f"tied_embeddings={tied_embeddings}{moe_description}",
        file=sys.stderr,
    )
    return arguments


def main():
    parser = argparse.ArgumentParser(description="BigMac switches and Qwen3-VL launch dimensions.")
    parser.add_argument("model_dir", help="Local HuggingFace Qwen3-VL model directory")
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--first-pipeline-num-layers", type=int)
    args = parser.parse_args()
    try:
        arguments = model_arguments(
            args.model_dir, args.pipeline_parallel_size, args.first_pipeline_num_layers,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Qwen3-VL model configuration error: {error}\n")
    print(" ".join(arguments))


if __name__ == "__main__":
    main()

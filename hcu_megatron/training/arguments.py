# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import os
import sys
import argparse

from typing import Union
from functools import wraps
from megatron.training.arguments import add_megatron_arguments
from megatron.core.msc_utils import MultiStorageClientFeature
from megatron.training.utils import warn_rank_0

from hcu_megatron.features_manager import ADAPTOR_FEATURES


def remove_original_params(parser, param_names: Union[list, str]):
    if isinstance(param_names, str):
        param_names = [param_names]

    for action in parser._actions:
        if action.dest in param_names:
            parser._actions.remove(action)
            for option_string in action.option_strings:
                if option_string in parser._option_string_actions:
                    del parser._option_string_actions[option_string]


def process_adaptor_args(parser):
    # add extra arguments
    parser = _add_extra_network_size_args(parser)
    parser = _add_extra_training_args(parser)
    parser = _add_extra_initialization_args(parser)
    parser = _add_extra_tokenizer_args(parser)
    parser = _add_extra_mbridge_args(parser)

    for feature in ADAPTOR_FEATURES:
        feature.register_args(parser)

    return parser


def parse_args(extra_args_provider=None, ignore_unknown_args=False):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Megatron-LM Arguments',
                                     allow_abbrev=False)

    parser = add_megatron_arguments(parser)

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    # add adaptor args
    parser = process_adaptor_args(parser)

    # Parse.
    explicit_args = {
        action.dest
        for action in parser._actions
        for option_string in action.option_strings
        if option_string and any(arg == option_string or arg.startswith(f"{option_string}=") for arg in sys.argv[1:])
    }
    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()
    args._explicit_args = explicit_args

    args._is_global_batch_size_explicitly_specified = args.global_batch_size is not None

    # Experimental yaml
    if args.yaml_cfg is not None:
        from megatron.training.yaml_arguments import load_yaml

        args = load_yaml(args.yaml_cfg)
        args._explicit_args = explicit_args

    # Args from environment
    if os.getenv("LAUNCH_BACKEND", "mpirun") == "torchrun":
        args.rank = int(os.getenv('RANK', '0'))
        args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    else:
        # set rank that safe_get_rank can use
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)

    # Args to enable MSC (opt-in: disabled by default)
    if args.disable_msc_deprecated:
        warn_rank_0(
            '--disable-msc is deprecated and will be removed in a future release. '
            'MSC is now disabled by default; pass --enable-msc to opt in.'
        )
        # Preserve legacy semantics: --disable-msc forces MSC off, even if
        # --enable-msc was also passed.
        args.enable_msc = False

    if args.enable_msc:
        MultiStorageClientFeature.enable()
        if not MultiStorageClientFeature.is_enabled():
            raise RuntimeError(
                "--enable-msc was passed but the multistorageclient package is not "
                "installed. Install it with `pip install multi-storage-client`."
            )
        warn_rank_0('The MSC feature is enabled.')
    else:
        MultiStorageClientFeature.disable()
        assert MultiStorageClientFeature.is_enabled() is False

    return args


def _add_extra_network_size_args(parser):
    # 删除原参数
    remove_original_params(parser, ["normalization"])

    # 重定义参数
    group = parser.add_argument_group(title='extra network size args')
    group.add_argument('--normalization', default='LayerNorm',
                       choices=['LayerNorm', 'RMSNorm', 'LightopRMSNorm'],
                       help='Which normalization technique to use.')

    return parser


def _add_extra_training_args(parser):
    remove_original_params(parser, ["recompute_modules"])

    group = parser.add_argument_group(title='extra training args')
    group.add_argument('--use-hip-profiler', action='store_true',
                       help='Use HIP PROFILER',
                       dest='use_hip_profiler')
    group.add_argument('--profile-dir', type=str, default="./",
                       help='profile dir to save.')
    group.add_argument('--recompute-modules', nargs='*', type=str, default=None,
                        help='The submodules to recompute. '
                        'choices: "core_attn", "moe_act", "layernorm", "mla_up_proj", "mlp", "moe", "experts", "router", "mhc".. '
                        'default: ["core_attn"].'
                        '"core_attn": recompute the core attention part of the transformer layer. '
                        '"moe_act": recompute the MoE MLP activation function. '
                        '"layernorm": recompute the input_layernorm and pre_mlp_layernorm. '
                        '"mla_up_proj": recompute the MLA up projection and RoPE applying parts.'
                        '"mlp": recompute the dense MLP layer.'
                        '"moe": recompute the MoE layer.'
                        '"experts: recompute the Experts layer"'
                        '"router: recompute the Router layer"'
                        '"mhc": recompute HyperConnection intermediate activations via CheckpointWithoutOutput + CheckpointManager. Requires enable_hyper_connections=True. Cannot be used with "mlp".'
                        '"moe_act", "layernorm", "mla_up_proj", and "mhc" use output-discarding checkpointing'
                        '"core_attn", "mlp", and "moe" uses normal checkpointing.')
    group.add_argument('--pipe-sp-splits', type=int, default=1,
                       help='num micro sequence')
    group.add_argument('--pipe-sp-strategy', type=str, default="average", choices=['average', 'uniform_comp'],
                       help='how to split sequence exactly')
    group.add_argument('--recompute-layer-ids', nargs='+', type=int, default=None,
                       help='Specify the exact IDs of transformer layers to recompute, enabling more flexible memory reduction.')
    group.add_argument('--recompute-mtp-layer-ids', nargs='+', type=int, default=None,
                       help='Specify the exact IDs of mtp layers to recompute, enabling more flexible memory reduction.')

    return parser


def _add_extra_initialization_args(parser):
    group = parser.add_argument_group(title='extra initialization args')
    group.add_argument('--reproduce', action='store_true',
                       help='reproduce train loss, need set --seed > 0.')

    return parser


def _add_extra_tokenizer_args(parser):
    # 删除原参数
    remove_original_params(parser, ["tokenizer_type"])

    # 重定义参数
    group = parser.add_argument_group(title='extra tokenizer args')
    group.add_argument('--extra-vocab-size', type=int, default=0,
                       help="--extra-vocab-size")
    group.add_argument('--tokenizer-type', type=str,
                       default=None,
                       choices=['BertWordPieceLowerCase',
                                'BertWordPieceCase',
                                'GPT2BPETokenizer',
                                'SentencePieceTokenizer',
                                'GPTSentencePieceTokenizer',
                                'HuggingFaceTokenizer',
                                'Llama2Tokenizer',
                                'Llama3Tokenizer',
                                'QwenTokenizer',
                                'TikTokenizer',
                                'MultimodalTokenizer',
                                'NullTokenizer',
                                'DeepSeekV2Tokenizer',
                                'SFTTokenizer',
                                'Qwen2VLTokenizer'],
                       help='What type of tokenizer to use.')
    return parser


def _add_extra_mbridge_args(parser):
    group = parser.add_argument_group(title='extra mbridge args')
    group.add_argument('--use-bridge', action='store_true',
                       help='Whether to use bridge module in the model')
    group.add_argument('--bridge-hf-model', type=str, default=None,
                       help='The HuggingFace model path for initializing the bridge module')
    group.add_argument('--load-weights', action='store_true',
                       help='Whether to load weights for the bridge module when initializing the model')
    group.add_argument('--bridge-language-model-only', action='store_true',
                       help='For VLM bridge providers, build only the language model (MCoreGPTModel) '
                            'via provider.provide_language_model, skipping ViT / vision projector. '
                            'No-op for pure LLM providers.')
    group.add_argument('--image-tokens-per-sample', type=int, default=None,
                       help='Number of image (post-merger) tokens per sample used to include '
                            'ViT + patch-embed + projector FLOPs in throughput accounting. '
                            'Only takes effect together with --use-bridge and a VLM provider that '
                            'exposes vision_config. Leave unset (default None) to keep the '
                            'LLM-only FLOPs baseline.')
    return parser


ORIGIN_ARG_VALUES = dict()

def validate_args_func_decorator(validate_args_func):
    @wraps(validate_args_func)
    def wrapper(args, defaults=None):
        if defaults is None:
            defaults = {}

        for feature in ADAPTOR_FEATURES:
            args = feature.pre_validate_args(args)

        # set num_layers. Otherwise validate_args will raise an error with msg 'Number of layers should be divisible by the pipeline-model-parallel size'
        if args.schedule_method == "dualpipev":
            args.encoder_num_layers = args.num_layers
            args.num_layers = None

        # delay_wgrad_compute supports overlap_grad_reduce
        global ORIGIN_ARG_VALUES
        ORIGIN_ARG_VALUES["delay_wgrad_compute"] = args.delay_wgrad_compute
        args.delay_wgrad_compute = False

        # set cross_entropy_fusion_impl to native. Otherwise validate_args will raise an error with msg
        # 'Transformer Engine cross entropy loss fusion is disabled due to stability issues.'
        ORIGIN_ARG_VALUES["cross_entropy_fusion_impl"] = args.cross_entropy_fusion_impl
        args.cross_entropy_fusion_impl = "native"

        args = validate_args_func(args, defaults)

        # print env vars
        _print_env_vars("env vars", exclude_vars=["BASH_FUNC", "OMPI"])

        args_dict = vars(args)
        for key, value in ORIGIN_ARG_VALUES.items():
            if key in args_dict:
                setattr(args, key, value)

        for feature in ADAPTOR_FEATURES:
            args = feature.validate_args(args)

        return args

    return wrapper


def _print_args(title, args):
    """Print arguments."""
    from megatron.training.utils import is_rank0

    global ORIGIN_ARG_VALUES

    args_dict = vars(args)
    for key, value in ORIGIN_ARG_VALUES.items():
        if key in args_dict:
            args_dict[key] = value
    args = argparse.Namespace(**args_dict)

    if is_rank0() and not int(os.getenv("UNIT_TEST_MODE", "0")):
        print(f'------------------------ {title} ------------------------', flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print(f'-------------------- end of {title} ---------------------', flush=True)


def _print_env_vars(title, exclude_vars=None):
    """Print arguments."""
    def _is_exclude_var(key):
        if exclude_vars is None:
            return False

        for var in exclude_vars:
            if key.startswith(var):
                return True

        return False

    from megatron.training.utils import is_rank0

    if is_rank0() and not int(os.getenv("UNIT_TEST_MODE", "0")):
        print(f'------------------------ {title} ------------------------', flush=True)
        str_list = []
        for key in os.environ.keys():
            if _is_exclude_var(key):
                continue
            dots = '.' * (48 - len(key))
            str_list.append('  {} {} {}'.format(key, dots, os.environ[key]))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print(f'-------------------- end of {title} ---------------------', flush=True)


_ADAPTOR_ARGS = None

def add_args(args, key, value):
    if key is not None:
        key = key[2:].replace('-', '_')
        if value is None:
            value = True
        elif len(value) == 1:
            value = value[0]
        setattr(args, key, value)


def parser_unknown_args(args, unknown):
    i = 0
    key = value = None
    while i < len(unknown):
        if unknown[i].startswith("--"):
            add_args(args, key, value)
            key = unknown[i]
            value = None
        else:
            if value is None:
                value = [unknown[i]]
            else:
                value.append(unknown[i])
        i += 1
    add_args(args, key, value)


def get_adaptor_args():
    global _ADAPTOR_ARGS
    if _ADAPTOR_ARGS is None:
        parser = argparse.ArgumentParser(description='Adaptor Arguments', allow_abbrev=False)
        _ADAPTOR_ARGS, unknown = process_adaptor_args(parser).parse_known_args()
        parser_unknown_args(_ADAPTOR_ARGS, unknown)
    return _ADAPTOR_ARGS


def destroy_adaptor_args():
    global _ADAPTOR_ARGS
    _ADAPTOR_ARGS = None

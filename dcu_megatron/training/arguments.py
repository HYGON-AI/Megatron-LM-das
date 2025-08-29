import argparse

from typing import Union
from functools import wraps
from megatron.training.arguments import add_megatron_arguments
from megatron.core.msc_utils import MultiStorageClientFeature

from dcu_megatron.adaptor.features_manager import ADAPTOR_FEATURES


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
    parser = _add_extra_distributed_args(parser)
    parser = _add_extra_tokenizer_args(parser)
    parser = _add_flux_args(parser)
    parser = _add_extra_checkpointing_args(parser)

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
    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    # Experimental yaml
    if args.yaml_cfg is not None:
        from megatron.training.yaml_arguments import load_yaml
        assert args.yaml_cfg and not args.use_legacy_models, \
            "Yaml config is not supported with legacy models."
        args = load_yaml(args.yaml_cfg)

    # Args from environment
    # args.rank = int(os.getenv('RANK', '0'))
    # args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    # Args to disable MSC
    if not args.enable_msc:
        MultiStorageClientFeature.disable()
        assert MultiStorageClientFeature.is_enabled() is False
        print('WARNING: The MSC feature is disabled.')

    return args


def _add_extra_network_size_args(parser):
    # 删除原参数
    remove_original_params(parser, ["normalization"])

    # 重定义参数
    group = parser.add_argument_group(title='extra network size args')
    group.add_argument('--normalization', default='LayerNorm',
                       choices=['LayerNorm', 'RMSNorm', 'LightopRMSNorm'],
                       help='Which normalization technique to use.')
    group.add_argument('--use-qk-norm', action='store_true',default=False,
                       help='Enable RMSNorm on Q, K before RoPE')
    return parser


def _add_extra_distributed_args(parser):
    group = parser.add_argument_group(title='extra distributed args')
    group.add_argument('--rank', default=-1, type=int,
                       help='node rank for distributed training')
    group.add_argument('--world-size', type=int, default=8,
                       help='number of nodes for distributed training')
    group.add_argument('--dist-url',
                       help='Which master node url for distributed training.')
    return parser


def _add_extra_training_args(parser):
    remove_original_params(parser, ["recompute_modules"])

    group = parser.add_argument_group(title='extra training args')
    group.add_argument('--use-hip-profiler', action='store_true',
                       help='Use HIP PROFILER',
                       dest='use_hip_profiler')
    group.add_argument('--profile-dir', type=str, default="./",
                       help='profile dir to save.')
    group.add_argument('--comm-time-log-iter', type=int, default=None,
                       help='iter to log communication time')
    group.add_argument('--recompute-modules', nargs='*', type=str, default=None,
                        help='The submodules to recompute. '
                        'choices: "core_attn", "moe_act", "layernorm", "mla_up_proj", "mlp", "moe", "experts", "router". '
                        'default: ["core_attn"].'
                        '"core_attn": recompute the core attention part of the transformer layer. '
                        '"moe_act": recompute the MoE MLP activation function. '
                        '"layernorm": recompute the input_layernorm and pre_mlp_layernorm. '
                        '"mla_up_proj": recompute the MLA up projection and RoPE applying parts.'
                        '"mlp": recompute the dense MLP layer.'
                        '"moe": recompute the MoE layer.'
                        '"experts: recompute the Experts layer"'
                        '"router: recompute the Router layer"'
                        '"moe_act", "layernorm", and "mla_up_proj" use output-discarding checkpointing, '
                        '"core_attn", "mlp", and "moe" uses normal checkpointing.')
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
                                'DeepSeekV2Tokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--use-quantize-comm',
                       default=False,
                       action="store_true",
                       help='use quantized communication')
    return parser


def _add_flux_args(parser):
    group = parser.add_argument_group(title='flux args')
    group.add_argument('--flux-transpose-weight', action='store_true', default=False,
                       help='Whether to transpose weight when using flux kernel')
    return parser


def _add_extra_checkpointing_args(parser):
    group = parser.add_argument_group(title='extra checkpointing args')
    group.add_argument('--use-ckpt-memory-cache', action='store_true', default=False,
                   help='Whether to enable memory caching for checkpoints (to speed up access by keeping checkpoints in memory)')
    return parser


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
        origin_delay_wgrad_compute = args.delay_wgrad_compute
        args.delay_wgrad_compute = False

        args = validate_args_func(args, defaults)

        # Check delay_wgrad_compute compatibility
        args.delay_wgrad_compute = origin_delay_wgrad_compute
        if args.delay_wgrad_compute:
            assert not args.moe_use_legacy_grouped_gemm, \
                'delay_wgrad_compute is not supported with legacy groupedgemm implementation'
            assert args.transformer_impl == 'transformer_engine', \
                'delay_wgrad_compute is only supported with transformer_engine implementation'

        for feature in ADAPTOR_FEATURES:
            feature.validate_args(args)

        return args

    return wrapper

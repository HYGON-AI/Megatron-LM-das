import os
import argparse

from typing import Union
from megatron.training.arguments import (
    _add_network_size_args,
    _add_regularization_args,
    _add_training_args,
    _add_initialization_args,
    _add_learning_rate_args,
    _add_checkpointing_args,
    _add_mixed_precision_args,
    _add_distributed_args,
    _add_validation_args,
    _add_data_args,
    _add_tokenizer_args,
    _add_autoresume_args,
    _add_biencoder_args,
    _add_vision_args,
    _add_moe_args,
    _add_mla_args,
    _add_logging_args,
    _add_straggler_detector_args,
    _add_inference_args,
    _add_transformer_engine_args,
    _add_retro_args,
    _add_experimental_args,
    _add_one_logger_args,
    _add_ft_package_args,
    _add_config_logger_args,
    _add_rerun_machine_args,
)


def remove_original_params(parser, param_names: Union[list, str]):
    if isinstance(param_names, str):
        param_names = [param_names]

    for action in parser._actions:
        if action.dest in param_names:
            parser._actions.remove(action)
            for option_string in action.option_strings:
                if option_string in parser._option_string_actions:
                    del parser._option_string_actions[option_string]


def parse_args(extra_args_provider=None, ignore_unknown_args=False):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Megatron-LM Arguments',
                                     allow_abbrev=False)

    # Standard arguments.
    parser = _add_network_size_args(parser)
    parser = _add_extra_network_size_args(parser)
    parser = _add_regularization_args(parser)
    parser = _add_training_args(parser)
    parser = _add_extra_training_args(parser)
    parser = _add_initialization_args(parser)
    parser = _add_learning_rate_args(parser)
    parser = _add_checkpointing_args(parser)
    parser = _add_mixed_precision_args(parser)
    parser = _add_distributed_args(parser)
    parser = _add_extra_distributed_args(parser)
    parser = _add_validation_args(parser)
    parser = _add_data_args(parser)
    parser = _add_tokenizer_args(parser)
    parser = _add_extra_tokenizer_args(parser)
    parser = _add_autoresume_args(parser)
    parser = _add_biencoder_args(parser)
    parser = _add_vision_args(parser)
    parser = _add_moe_args(parser)
    parser = _add_mla_args(parser)
    parser = _add_mtp_args(parser)
    parser = _add_logging_args(parser)
    parser = _add_straggler_detector_args(parser)
    parser = _add_inference_args(parser)
    parser = _add_transformer_engine_args(parser)
    parser = _add_retro_args(parser)
    parser = _add_experimental_args(parser)
    parser = _add_one_logger_args(parser)
    parser = _add_ft_package_args(parser)
    parser = _add_config_logger_args(parser)
    parser = _add_rerun_machine_args(parser)
    parser = _add_flux_args(parser)

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

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
    #args.rank = int(os.getenv('RANK', '0'))
    #args.world_size = int(os.getenv("WORLD_SIZE", '1'))

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
    group = parser.add_argument_group(title='extra training args')
    group.add_argument('--use-hip-profiler', action='store_true',
                       help='Use HIP PROFILER',
                       dest='use_hip_profiler')
    group.add_argument('--profile-dir', type=str, default="./",
                       help='profile dir to save.')

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
    return parser


def _add_mtp_args(parser):
    group = parser.add_argument_group(title='multi token prediction')
    group.add_argument('--mtp-num-layers', type=int, default=None,
                       help='Number of Multi-Token Prediction (MTP) Layers.'
                       'MTP extends the prediction scope to multiple future tokens at each position.'
                       'This MTP implementation sequentially predict additional tokens '
                       'by using D sequential modules to predict D additional tokens.')
    group.add_argument('--mtp-loss-scaling-factor', type=float, default=0.3,
                       help='Scaling factor of Multi-Token Prediction (MTP) loss. '
                       'We compute the average of the MTP losses across all depths, '
                       'and multiply it the scaling factor to obtain the overall MTP loss, '
                       'which serves as an additional training objective.')
    return parser


def _add_flux_args(parser):
    group = parser.add_argument_group(title='flux args')
    group.add_argument('--flux-transpose-weight', action='store_true', default=False,
                       help='Whether to transpose weight when using flux kernel')
    return parser

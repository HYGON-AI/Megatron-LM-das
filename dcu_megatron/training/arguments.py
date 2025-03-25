import argparse

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


def parse_args(extra_args_provider=None, ignore_unknown_args=False):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Megatron-LM Arguments',
                                     allow_abbrev=False)

    # Standard arguments.
    parser = _add_network_size_args(parser)
    parser = _add_regularization_args(parser)
    parser = _add_training_args(parser)
    parser = _add_initialization_args(parser)
    parser = _add_learning_rate_args(parser)
    parser = _add_checkpointing_args(parser)
    parser = _add_mixed_precision_args(parser)
    parser = _add_distributed_args(parser)
    parser = _add_validation_args(parser)
    parser = _add_data_args(parser)
    parser = _add_tokenizer_args(parser)
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


def _add_tokenizer_args(parser):
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--vocab-size', type=int, default=None,
                       help='Size of vocab before EOD or padding.')
    group.add_argument('--extra-vocab-size', type=int, default=0,
                       help="--extra-vocab-size")
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file.')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file.')
    group.add_argument('--vocab-extra-ids', type=int, default=0,
                       help='Number of additional vocabulary tokens. '
                            'They are used for span masking in the T5 model')
    group.add_argument('--tokenizer-type', type=str,
                       default=None,
                       choices=['BertWordPieceLowerCase',
                                'BertWordPieceCase',
                                'GPT2BPETokenizer',
                                'SentencePieceTokenizer',
                                'GPTSentencePieceTokenizer',
                                'HuggingFaceTokenizer',
                                'Llama2Tokenizer',
                                'TikTokenizer',
                                'MultimodalTokenizer',
                                'NullTokenizer',
                                'DeepSeekV2Tokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--tokenizer-model', type=str, default=None,
                       help='Sentencepiece tokenizer model.')
    group.add_argument('--tiktoken-pattern', type=str, default=None,
                       help='Which tiktoken pattern to use. Options: [v1, v2]')
    group.add_argument('--tiktoken-num-special-tokens', type=int, default=1000,
                       help='Number of special tokens in tiktoken tokenizer')
    group.add_argument('--tiktoken-special-tokens', type=str, nargs='+', default=None,
                       help='List of tiktoken special tokens, needs to have ["<unk>", "<s>", "</s>"]')
    return parser


def _add_mtp_args(parser):
    group = parser.add_argument_group(title='multi token prediction')
    group.add_argument('--num-nextn-predict-layers', type=int, default=0, help='Multi-Token prediction layer num')
    group.add_argument('--mtp-loss-scale', type=float, default=0.3, help='Multi-Token prediction loss scale')
    group.add_argument('--recompute-mtp-norm', action='store_true', default=False,
                       help='Multi-Token prediction recompute norm')
    group.add_argument('--recompute-mtp-layer', action='store_true', default=False,
                       help='Multi-Token prediction recompute layer')
    group.add_argument('--share-mtp-embedding-and-output-weight', action='store_true', default=False,
                       help='Main model share embedding and output weight with mtp layer.')
    return parser
import os
import argparse

from megatron.training.arguments import (
    _add_network_size_args,
    _add_regularization_args,
    _add_training_args,
    _add_initialization_args,
    _add_learning_rate_args,
    _add_checkpointing_args,
    _add_mixed_precision_args,
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


def _add_distributed_args(parser):
    group = parser.add_argument_group(title='distributed')

    group.add_argument('--tensor-model-parallel-size', type=int, default=1,
                       help='Degree of tensor model parallelism.')
    group.add_argument('--encoder-tensor-model-parallel-size', type=int, default=0,
                       help='Degree of tensor model parallelism for the encoder.')
    group.add_argument('--pipeline-model-parallel-size', type=int, default=1,
                       help='Degree of pipeline model parallelism.')
    group.add_argument('--encoder-pipeline-model-parallel-size', type=int, default=0,
                       help=('Degree of pipeline model parallelism in the encoder. This is '
                             'independent of the amount of pipeline in the decoder.'))
    group.add_argument('--pipeline-model-parallel-split-rank',
                       type=int, default=None,
                       help=('Rank where encoder and decoder should be split. '
                             'Deprecated; use --encoder-pipeline-model-parallel-size instead.'))
    group.add_argument('--decoder-first-pipeline-num-layers',
                       type=int, default=None,
                       help=('The number of transformer layers on the first pipeline stage of the decoder. '
                       'Default None is even split of transformer layers across all pipeline stages'))
    group.add_argument('--decoder-last-pipeline-num-layers',
                       type=int, default=None,
                       help=('The number of transformer layers on the last pipeline stage of the decoder. '
                       'Default None is even split of transformer layers across all pipeline stages'))
    group.add_argument('--model-parallel-size', type=int, default=None,
                       help='Old model parallel argument, do not use. Use '
                       '--tensor-model-parallel-size instead.')
    group.add_argument('--num-layers-per-virtual-pipeline-stage', type=int, default=None,
                       help='Number of layers per virtual pipeline stage')
    group.add_argument('--num-virtual-stages-per-pipeline-rank', type=int, default=None,
                       help='Number of virtual pipeline stages per pipeline parallelism rank')
    group.add_argument('--microbatch-group-size-per-virtual-pipeline-stage', type=int, default=None,
                       help='Number of contiguous microbatches per virtual pipeline stage',
                       dest='microbatch_group_size_per_vp_stage')
    group.add_argument('--no-overlap-p2p-communication', action='store_false',
                       help='overlap pipeline parallel communication with forward and backward chunks in 1F1B',
                       dest='overlap_p2p_comm')
    group.add_argument('--overlap-p2p-communication-warmup-flush', action='store_true',
                       default=False, help='if set, overlap pipeline parallel communication in warmup and flush',
                       dest='overlap_p2p_comm_warmup_flush')
    group.add_argument('--distributed-backend', default='nccl',
                       choices=['nccl', 'gloo'],
                       help='Which backend to use for distributed training.')
    group.add_argument('--distributed-timeout-minutes', type=int, default=10,
                       help='Timeout minutes for torch.distributed.')
    group.add_argument('--overlap-grad-reduce', action='store_true',
                       default=False, help='If set, overlap DDP grad reduce.')
    group.add_argument('--defer-embedding-wgrad-compute', action='store_true',
                       default=False, help='If set, defers the vocabulary projection linear layer weight'
                       'gradient compute to pipeline flush.', dest='defer_embedding_wgrad_compute')
    group.add_argument('--wgrad-deferral-limit', type=int, default=0, help='Number of micro-batches for which'
                       'weight gradient computation of vocabulary projection is deferred, defaults to 0 which'
                       'means all the micro-batches are deferred. Invalid if `defer-embedding-wgrad-compute`'
                       'is not set')
    group.add_argument('--no-align-grad-reduce', action='store_false',
                       help='If not set, all PP stages will launch gradient reduces simultaneously. '
                       'Otherwise, each PP stage will independently launch as needed.',
                       dest='align_grad_reduce')
    group.add_argument('--ddp-bucket-size', type=int, default=None,
                       help='Bucket size for data-parallel communication')
    group.add_argument('--ddp-average-in-collective', action='store_true',
                       default=False, help='If set, average directly in data-parallel communication collective.')
    group.add_argument('--overlap-param-gather', action='store_true',
                       default=False, help='If set, overlap param all-gather in distributed optimizer.')
    group.add_argument('--overlap-param-gather-with-optimizer-step', action='store_true',
                       default=False, help='If set, overlap param all-gather of first bucket with optimizer step.')
    group.add_argument('--no-align-param-gather', action='store_false',
                       help='If not set, all PP stages will launch param all-gathers simultaneously. '
                       'Otherwise, each PP stage will independently launch as needed.',
                       dest='align_param_gather')
    group.add_argument('--no-scatter-gather-tensors-in-pipeline', action='store_false',
                       help='If not set, use scatter/gather to optimize communication of tensors in pipeline.',
                       dest='scatter_gather_tensors_in_pipeline')
    group.add_argument('--use-ring-exchange-p2p', action='store_true',
                       default=False, help='If set, use custom-built ring exchange '
                       'for p2p communications. Note that this option will require '
                       'a custom built image that support ring-exchange p2p.')
    group.add_argument('--local-rank', type=int, default=int(os.getenv('LOCAL_RANK', '0')),
                       help='local rank passed from distributed launcher.')
    group.add_argument('--lazy-mpu-init', type=bool, required=False,
                       help='If set to True, initialize_megatron() '
                       'skips DDP initialization and returns function to '
                       'complete it instead.Also turns on '
                       '--use-cpu-initialization flag. This is for '
                       'external DDP manager.' )
    group.add_argument('--account-for-embedding-in-pipeline-split', action='store_true',
                       default=False, help='If set, *input* embedding layer will be treated as a standard transformer'
                       'layer in the context of partition and placement for pipeline parallelism.')
    group.add_argument('--account-for-loss-in-pipeline-split', action='store_true',
                       default=False, help='If set, loss layer will be treated as a standard transformer'
                       'layer in the context of partition and placement for pipeline parallelism.')
    group.add_argument('--use-distributed-optimizer', action='store_true',
                       help='Use distributed optimizer.')
    group.add_argument('--num-distributed-optimizer-instances', type=int, default=1,
                       help='Number of Distributed Optimizer copies across Data Parallel domain.')
    group.add_argument('--use-torch-fsdp2', action='store_true',
                       help="Use the torch FSDP2 implementation. FSDP2 is not currently working with Pipeline Parallel."
                       "It is still not in a stable release stage, and may therefore contain bugs or other potential issues.")
    group.add_argument('--context-parallel-size', type=int, default=1,
                       help='Degree of context parallelism.')
    group.add_argument('--cp-comm-type', nargs='+', type=str, default=["p2p"],
                       help='Inter-gpu communication type for context parallelism: '
                       'p2p, a2a, allgather or a2a+p2p. If a single string is provided, '
                       'all layers will share the same communication type. Users can also '
                       'specify separated types for each layer like '
                       '--cp-comm-type p2p p2p a2a a2a a2a+p2p a2a+p2p')
    group.add_argument('--hierarchical-context-parallel-sizes', nargs='+', type=int, default=None,
                       help='Degrees of the hierarchical context parallelism. Users should '
                       'provide a list to specify the sizes for different levels. '
                       '--hierarchical-context-parallel-sizes 2 4 indicates every two adjacent gpus '
                       'forms the first level of cp groups and the cp ranks with the same odevity '
                       'forms the second level of cp groups.')
    group.add_argument('--nccl-communicator-config-path', type=str, default=None,
                       help='Path to the yaml file with NCCL communicator '
                       'configurations. The number of min/max thread groups and thread '
                       'group cluster size of each communicator can be configured by '
                       'setting `min_ctas`, `max_ctas`, and `cga_cluster_size`.')
    group.add_argument('--use-tp-pp-dp-mapping', action='store_true', default=False,
                        help='If set, distributed ranks initialize order is changed '
                        'from tp-cp-ep-dp-pp to tp-cp-ep-pp-dp.')
    group.add_argument('--replication', action='store_true', default=False,
                       help="If set, replication of local checkpoints is enabled. "
                       "Needs to be enabled on all ranks.")
    group.add_argument('--replication-jump', default=None, type=int,
                       help="Specifies `J`, the spacing between ranks storing replicas of a given rank's data. "
                       "Replicas for rank `n` may be on ranks `n+J`, `n+2J`, ..., or `n-J`, `n-2J`, etc. "
                       "This flag has an effect only if --replication is used. "
                       "and must be consistent across all ranks.")
    group.add_argument('--replication-factor', default=2, type=int,
                       help="Number of machines storing the replica of a given rank's data.")
    group.add_argument('--rank', default=-1, type=int,
                       help='node rank for distributed training')
    group.add_argument('--world-size', type=int, default=8,
                       help='number of nodes for distributed training')
    group.add_argument('--dist-url',
                       help='Which master node url for distributed training.')
    return parser


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
import os
import argparse

from megatron.training.arguments import (
    _add_network_size_args,
    _add_regularization_args,
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
                                'Llama3Tokenizer',
                                'QwenTokenizer',
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


def _add_training_args(parser):
    group = parser.add_argument_group(title='training')

    group.add_argument('--micro-batch-size', type=int, default=None,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size times number of micro batches.')
    group.add_argument('--batch-size', type=int, default=None,
                       help='Old batch size parameter, do not use. '
                       'Use --micro-batch-size instead')
    group.add_argument('--global-batch-size', type=int, default=None,
                       help='Training batch size. If set, it should be a '
                       'multiple of micro-batch-size times data-parallel-size. '
                       'If this value is None, then '
                       'use micro-batch-size * data-parallel-size as the '
                       'global batch size. This choice will result in 1 for '
                       'number of micro-batches.')
    group.add_argument('--rampup-batch-size', nargs='*', default=None,
                       help='Batch size ramp up with the following values:'
                       '  --rampup-batch-size <start batch size> '
                       '                      <batch size incerement> '
                       '                      <ramp-up samples> '
                       'For example:'
                       '   --rampup-batch-size 16 8 300000 \\ '
                       '   --global-batch-size 1024'
                       'will start with global batch size 16 and over '
                       ' (1024 - 16) / 8 = 126 intervals will increase'
                       'the batch size linearly to 1024. In each interval'
                       'we will use approximately 300000 / 126 = 2380 samples.')
    group.add_argument('--decrease-batch-size-if-needed', action='store_true', default=False,
                       help='If set, decrease batch size if microbatch_size * dp_size'
                       'does not divide batch_size. Useful for KSO (Keep Soldiering On)'
                       'to continue making progress if number of healthy GPUs (and'
                       'corresponding dp_size) does not support current batch_size.'
                       'Old batch_size will be restored if training is re-started with'
                       'dp_size that divides batch_size // microbatch_size.')
    group.add_argument('--recompute-activations', action='store_true',
                       help='recompute activation to allow for training '
                       'with larger models, sequences, and batch sizes.')
    group.add_argument('--recompute-granularity', type=str, default=None,
                       choices=['full', 'selective'],
                       help='Checkpoint activations to allow for training '
                       'with larger models, sequences, and batch sizes. '
                       'It is supported at two granularities 1) full: '
                       'whole transformer layer is recomputed, '
                       '2) selective: core attention part of the transformer '
                       'layer is recomputed.')
    group.add_argument('--no-check-for-nan-in-loss-and-grad', action='store_false',
                       help='Check for NaNs in loss and grad',
                       dest='check_for_nan_in_loss_and_grad')
    group.add_argument('--check-for-spiky-loss', action='store_true',
                       help='Check for spiky loss',
                       dest='check_for_spiky_loss')
    group.add_argument('--distribute-saved-activations',
                       action='store_true',
                       help='If set, distribute recomputed activations '
                       'across model parallel group.')
    group.add_argument('--recompute-method', type=str, default=None,
                       choices=['uniform', 'block'],
                       help='1) uniform: uniformly divide the total number of '
                       'Transformer layers and recompute the input activation of '
                       'each divided chunk at specified granularity, '
                       '2) recompute the input activations of only a set number of '
                       'individual Transformer layers per pipeline stage and do the '
                       'rest without any recomputing at specified granularity'
                       'default) do not apply activations recompute to any layers')
    group.add_argument('--recompute-num-layers', type=int, default=None,
                       help='1) uniform: the number of Transformer layers in each '
                       'uniformly divided recompute unit, '
                       '2) block: the number of individual Transformer layers '
                       'to recompute within each pipeline stage.')
    group.add_argument('--no-clone-scatter-output-in-embedding', action='store_false',
                       help='If not set, clone the output of the scatter in embedding layer to GC original tensor.',
                       dest='clone_scatter_output_in_embedding')
    group.add_argument('--profile', action='store_true',
                       help='Enable nsys profiling. When using this option, nsys '
                       'options should be specified in commandline. An example '
                       'nsys commandline is `nsys profile -s none -t nvtx,cuda '
                       '-o <path/to/output_file> --force-overwrite true '
                       '--capture-range=cudaProfilerApi '
                       '--capture-range-end=stop`.')
    group.add_argument('--profile-step-start', type=int, default=10,
                       help='Global step to start profiling.')
    group.add_argument('--profile-step-end', type=int, default=12,
                       help='Global step to stop profiling.')
    group.add_argument('--use-pytorch-profiler', action='store_true',
                       help='Use the built-in pytorch profiler. '
                       'Useful if you wish to view profiles in tensorboard.',
                       dest='use_pytorch_profiler')
    group.add_argument('--profile-ranks', nargs='+', type=int, default=[0],
                       help='Global ranks to profile.')
    group.add_argument('--record-memory-history', action="store_true", default=False,
                       help='Record memory history in last rank.')
    group.add_argument('--memory-snapshot-path', type=str, default="snapshot.pickle",
                       help='Specifies where to dump the memory history pickle.')
    group.add_argument('--tp-comm-overlap', action='store_true', help='Enables the '
                       ' overlap of Tensor parallel communication and GEMM kernels.')
    group.add_argument('--tp-comm-overlap-cfg', type=str, default=None,
                       help='Config file when tp_comm_overlap is enabled.')
    group.add_argument('--disable-tp-comm-overlap-ag', action='store_false',
                       help=('Disables the All-Gather overlap with GEMM by '
                             'pipelining the GEMM and All-Gather.'),
                       dest='tp_comm_overlap_ag')
    group.add_argument('--disable-tp-comm-overlap-rs', action='store_false',
                       help=('Disables the Reduce-Scatter overlap with GEMM by '
                             'pipelining the GEMM and Reduce-Scatter.'),
                       dest='tp_comm_overlap_rs')
    group.add_argument('--tp-comm-overlap-rs-dgrad', action='store_true',
                       help = 'Enables the Reduce-Scatter overlap with dgrad GEMM.',
                       dest='tp_comm_overlap_rs_dgrad')
    group.add_argument('--disable-tp-comm-bulk-dgrad', action='store_false',
                       help='Disables the All-Gather overlap with bprop activation gradient GEMM.',
                       dest='tp_comm_bulk_dgrad')
    group.add_argument('--disable-tp-comm-bulk-wgrad', action='store_false',
                       help='Disables the Reduce-Scatter overlap with bprop weight gradient GEMM.',
                       dest='tp_comm_bulk_wgrad')
    group.add_argument('--tp-comm-bootstrap-backend', default='nccl', type=str,
                       choices=['nccl', 'mpi', 'gloo'],
                       help='Set the bootstrapping backend of Tensor parallel communications.')
    group.add_argument('--use-cpu-initialization', action='store_true',
                       default=None,
                       help='If set, initialize weights on the CPU. This eliminates init differences based on tensor parallelism.')
    group.add_argument('--empty-unused-memory-level', default=0, type=int,
                       choices=[0, 1, 2],
                       help='Call torch.cuda.empty_cache() each iteration '
                       '(training and eval), to reduce fragmentation.'
                       '0=off, 1=moderate, 2=aggressive.')
    group.add_argument('--deterministic-mode', action='store_true',
                       help='Choose code that has deterministic execution. This usually '
                       'means slower execution, but is good for debugging and testing.')
    group.add_argument('--check-weight-hash-across-dp-replicas-interval', type=int, default=None,
                       help='Interval to check weight hashes are same across DP replicas. If not specified, weight hashes not checked.')
    group.add_argument('--calculate-per-token-loss', action='store_true',
                       help=('Scale cross entropy loss by the number of non-padded tokens in the '
                             'global batch, versus the default behavior of assuming all tokens are non-padded.'))
    group.add_argument('--train-sync-interval', type=int, default=None,
                       help='Training CPU-GPU synchronization interval, to ensure that CPU is not running too far ahead of GPU.')

    # deprecated
    group.add_argument('--checkpoint-activations', action='store_true',
                       help='Checkpoint activation to allow for training '
                       'with larger models, sequences, and batch sizes.')
    group.add_argument('--train-iters', type=int, default=None,
                       help='Total number of iterations to train over all '
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
    group.add_argument('--train-samples', type=int, default=None,
                       help='Total number of samples to train over all '
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Report loss and timing interval.')
    group.add_argument('--exit-interval', type=int, default=None,
                       help='Exit the program after the iteration is divisible '
                       'by this value.')
    group.add_argument('--exit-duration-in-mins', type=int, default=None,
                       help='Exit the program after this many minutes.')
    group.add_argument('--exit-signal-handler', action='store_true',
                       help='Dynamically save the checkpoint and shutdown the '
                       'training if SIGTERM is received')
    group.add_argument('--tensorboard-dir', type=str, default=None,
                       help='Write TensorBoard logs to this directory.')
    group.add_argument('--no-masked-softmax-fusion',
                       action='store_false',
                       help='Disable fusion of query_key_value scaling, '
                       'masking, and softmax.',
                       dest='masked_softmax_fusion')
    group.add_argument('--no-bias-gelu-fusion', action='store_false',
                       help='Disable bias and gelu fusion.',
                       dest='bias_gelu_fusion')
    group.add_argument('--no-bias-swiglu-fusion', action='store_false',
                       help='Disable bias and swiglu fusion, the fusion is '
                       'available only when using megatron-core.',
                       dest='bias_swiglu_fusion')
    group.add_argument('--no-bias-dropout-fusion', action='store_false',
                       help='Disable bias and dropout fusion.',
                       dest='bias_dropout_fusion')
    group.add_argument('--no-rope-fusion', action='store_false',
                       help='Disable rope fusion, the fusion is available '
                       'only when using megatron-core.',
                       dest='apply_rope_fusion')
    group.add_argument('--cross-entropy-loss-fusion', action='store_true',
                       help='Enabled fusion of cross entropy loss calculation.',
                       dest='cross_entropy_loss_fusion')
    group.add_argument('--use-flash-attn', action='store_true',
                       help='use FlashAttention implementation of attention. '
                       'https://arxiv.org/abs/2205.14135')
    group.add_argument('--disable-bias-linear', action='store_false',
                       help='Disable bias in the linear layers',
                       dest='add_bias_linear')
    group.add_argument('--add-qkv-bias', action='store_true',
                       help='Enable bias only in the QKV linear layers',
                       dest='add_qkv_bias')
    group.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd'],
                       help='Optimizer function')
    group.add_argument('--dataloader-type', type=str, default=None,
                       choices=['single', 'cyclic', 'external'],
                       help='Single pass vs multiple pass data loader')
    group.add_argument('--no-async-tensor-model-parallel-allreduce',
                       action='store_false',
                       help='DEPRECATED. This flag is ignored.',
                       dest='async_tensor_model_parallel_allreduce')
    group.add_argument('--no-persist-layer-norm', action='store_true',
                       help='Disable using persistent fused layer norm kernel. '
                       'This kernel supports only a set of hidden sizes. Please '
                       'check persist_ln_hidden_sizes if your hidden '
                       'size is supported.')
    group.add_argument('--sequence-parallel', action='store_true',
                       help='Enable sequence parallel optimization.')
    group.add_argument('--no-gradient-accumulation-fusion',
                       action='store_false',
                       help='Disable fusing gradient accumulation to weight '
                       'gradient computation of linear layers',
                       dest='gradient_accumulation_fusion')
    group.add_argument('--use-mcore-models', action='store_true',
                       dest='deprecated_use_mcore_models',
                       help='DEPRECATED. Use the implementation from megatron core.'
                       'Now ignored and mcore models are the default, use '
                       '--use-legacy-models to not use core models.')
    group.add_argument('--use-legacy-models', action='store_true',
                       help='Use the legacy Megatron models, not Megatron-Core models.')
    group.add_argument('--manual-gc', action='store_true',
                       help='Disable the threshold-based default garbage '
                       'collector and trigger the garbage collection manually. '
                       'Manual garbage collection helps to align the timing of '
                       'the collection across ranks which mitigates the impact '
                       'of CPU-associated jitters. When the manual gc is enabled, '
                       'garbage collection is performed only at the start and the '
                       'end of the validation routine by default.')
    group.add_argument('--manual-gc-interval', type=int, default=0,
                       help='Training step interval to trigger manual garbage '
                       'collection. When the value is set to 0, garbage '
                       'collection is not triggered between training steps.')
    group.add_argument('--no-manual-gc-eval', action='store_false',
                       help='When using manual garbage collection, disable '
                       'garbage collection at the start and the end of each '
                       'evaluation run.', dest='manual_gc_eval')
    group.add_argument('--disable-tp-comm-split-ag', action='store_false',
                       help='Disables the All-Gather overlap with fprop GEMM.',
                       dest='tp_comm_split_ag')
    group.add_argument('--disable-tp-comm-split-rs', action='store_false',
                       help='Disables the Reduce-Scatter overlap with fprop GEMM.',
                       dest='tp_comm_split_rs')
    group.add_argument('--use-hip-profiler', action='store_true',
                       help='Use HIP PROFILER',
                       dest='use_hip_profiler')
    group.add_argument('--profile-dir', type=str, default="./",
                       help='profile dir to save.')
    
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
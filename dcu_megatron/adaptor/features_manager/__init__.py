from .pipeline_parallel.pipeline_feature import PipelineFeature
from .tensor_parallel.parallel_linear_feature import ParallelLinearFeature
from .optimizer.optimizer_feature import OptimizerFeature
from .communication.gradient_compress_feature import GradientCompressFeature
from .memory.swap_attention_feature import SwapAttentionFeature
from .memory.cpu_offload_feature import CPUOffloadFeature

ADAPTOR_FEATURES = [
    PipelineFeature(),
    OptimizerFeature(),
    ParallelLinearFeature(),
    GradientCompressFeature(),
    SwapAttentionFeature(),
    CPUOffloadFeature(),
]

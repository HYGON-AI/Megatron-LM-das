from .pipeline_parallel.pipeline_feature import PipelineFeature
from .pipeline_parallel.ripipe_feature import RiPipeFeature
from .tensor_parallel.parallel_linear_feature import ParallelLinearFeature
from .optimizer.optimizer_feature import OptimizerFeature
from .communication.gradient_compress_feature import GradientCompressFeature
from .communication.quantize_comm_feature import QuantizeCommFeature
from .memory.swap_attention_feature import SwapAttentionFeature
from .recompute.activation_function import RecomputeActivationFeature

ADAPTOR_FEATURES = [
    PipelineFeature(),
    RiPipeFeature(),
    OptimizerFeature(),
    ParallelLinearFeature(),
    GradientCompressFeature(),
    QuantizeCommFeature(),
    SwapAttentionFeature(),
    RecomputeActivationFeature(),
]

from .basic.megatron_basic import MegatronBasicFeature
from .basic.minimum_basic import MinimumBasicFeature
from .communication.gradient_compress_feature import GradientCompressFeature
from .communication.quantize_comm_feature import QuantizeCommFeature
from .fusions.linear_cross_entropy_feature import LinearCrossEntropyFeature
from .memory.swap_attention_feature import SwapAttentionFeature
from .moe.sync_free_moe_feature import SyncFreeMoeFeature
from .optimizer.optimizer_feature import OptimizerFeature
from .pipeline_parallel.pipeline_feature import PipelineFeature
from .pipeline_parallel.ripipe_feature import RiPipeFeature
from .recompute.activation_function import RecomputeActivationFeature
from .tensor_parallel.parallel_linear_feature import ParallelLinearFeature
from .transformer.hyper_connection_feature import HyperConnectionFeature
from .moe.sync_free_moe_feature import SyncFreeMoeFeature
from .moe.ultraep_feature import UltraEPFeature
from .transformer.dsa_feature import DSAFeature
from .moe.mega_moe_feature import MegaMoeFeature


ADAPTOR_FEATURES = [
    PipelineFeature(),
    RiPipeFeature(),
    OptimizerFeature(),
    ParallelLinearFeature(),
    GradientCompressFeature(),
    QuantizeCommFeature(),
    SwapAttentionFeature(),
    RecomputeActivationFeature(),
    MegatronBasicFeature(),
    LinearCrossEntropyFeature(),
    MinimumBasicFeature(),
    HyperConnectionFeature(),
    SyncFreeMoeFeature(),
    UltraEPFeature(),
    DSAFeature(),
    MegaMoeFeature(),
]

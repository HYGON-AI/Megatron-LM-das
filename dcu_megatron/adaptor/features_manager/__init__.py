from .pipeline_parallel.pipeline_feature import PipelineFeature
from .tensor_parallel.parallel_linear_feature import ParallelLinearFeature
from .optimizer.optimizer_feature import OptimizerFeature

ADAPTOR_FEATURES = [
    PipelineFeature(),
    OptimizerFeature(),
    ParallelLinearFeature(),
]

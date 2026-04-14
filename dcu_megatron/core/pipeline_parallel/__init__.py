from dcu_megatron.adaptor.megatron_adaptor import get_adaptor_args


if get_adaptor_args().schedule_method == "dualpipev":
    from .fine_grained_activation_offload_dualpipev import (
        PipelineOffloadManager,
        fine_grained_offloading_group_commit,
        fine_grained_offloading_group_start,
        get_fine_grained_offloading_context,
        fine_grained_offloading_set_last_layer,
    )
else:
    from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
        PipelineOffloadManager,
        fine_grained_offloading_group_commit,
        fine_grained_offloading_group_start,
    )

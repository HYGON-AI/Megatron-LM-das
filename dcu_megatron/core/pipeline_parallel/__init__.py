import os


if int(os.environ.get("USE_DUALPIPEV_SCHEDULE", 0)):
    from .cpu_offload_dualpipev import (
        PipelineOffloadManager,
        fine_grained_offloading_group_commit,
        fine_grained_offloading_group_start,
        get_fine_grained_offloading_context,
        fine_grained_offloading_set_last_layer,
    )
else:
    from .fine_grained_activation_offload import (
        PipelineOffloadManager,
        fine_grained_offloading_group_commit,
        fine_grained_offloading_group_start,
        get_fine_grained_offloading_context,
        fine_grained_offloading_set_last_layer,
    )

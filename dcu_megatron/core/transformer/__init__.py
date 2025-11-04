import os


if int(os.environ.get("USE_DUALPIPEV_SCHEDULE", 0)):
    from .cpu_offload_dualpipev import (
        PipelineOffloadManager,
        group_prefetch_offload_start,
        group_prefetch_offload_commit,
    )
else:
    from .cpu_offload import (
        PipelineOffloadManager,
        group_prefetch_offload_start,
        group_prefetch_offload_commit,
    )

def finish_grad_sync(self):
    """
    Finishes grad sync (all-reduce or reduce-scatter) communication operations
    for all model gradients.

    When overlap_grad_reduce is set to True, waits for asynchronous communication
    calls to complete. When overlap_grad_reduce is set to False, calls synchronous
    communication ops.
    """
    for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
        bucket_group.update()
        bucket_group.finish_grad_sync()
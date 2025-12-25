from functools import wraps

from megatron.training import get_args
from megatron.core.utils import get_model_config

# from .dualpipev.dualpipev_schedules import forward_backward_pipelining_with_cutinhalf
from ..transformer import PipelineOffloadManager


def get_forward_backward_func_wrapper(fn):
    @wraps(fn)
    def wrapper():
        """Retrieves the appropriate forward_backward function given the
        configuration of parallel_state.

        Returns a function that will perform all of the forward and
        backward passes of the model given the pipeline model parallel
        world size and virtual pipeline model parallel world size in the
        global parallel_state.

        """

        args = get_args()
        if args.schedule_method == "vanilla":
            return fn()
        elif args.schedule_method == "dualpipev":
            return forward_backward_pipelining_with_cutinhalf
        else:
            raise ValueError(f"schedule_method {args.schedule_method} is not supported")

    return wrapper


def forward_backward_pipelining_with_interleaving_wrapper(fn):
    @wraps(fn)
    def wrapper(
        *,
        forward_step_func,
        data_iterator,
        model,
        num_microbatches: int,
        seq_length: int,
        micro_batch_size: int,
        decoder_seq_length=None,
        forward_only=False,
        collect_non_loss_data=False,
        first_val_step=None,
        adjust_tensor_shapes_fn=None,  # unused
        p2p_communicator=None,
        pg_collection=None,
    ):

        config = get_model_config(model[0])
        if not forward_only and config.fine_grained_activation_offloading:
            PipelineOffloadManager.get_instance().reset()

        return fn(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_microbatches,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_lengthe,
            forward_only=forward_only,
            collect_non_loss_data=collect_non_loss_data,
            first_val_step=first_val_step,
            adjust_tensor_shapes_fn=adjust_tensor_shapes_fn,  # unused
            p2p_communicator=p2p_communicator,
            pg_collection=pg_collection,
        )

    return wrapper

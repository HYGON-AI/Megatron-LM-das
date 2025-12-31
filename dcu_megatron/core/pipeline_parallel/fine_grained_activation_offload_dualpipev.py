import copy
from collections import deque
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
import transformer_engine as te
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.tensor.quantized_tensor import QuantizedTensorBase

from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import get_num_microbatches


def get_backward_chunk_order(num_microbatches):
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()

    order = deque() # 0: first chunk; 1: second chunk
    for _ in range(pp_size - pp_rank - 1):   # 1b1w1f
        order.append(1)
    num_second_chunks = pp_size - pp_rank - 1

    num_remain_backward = 2 * num_microbatches - (pp_size - pp_rank - 1)
    next_backward_chunk = 1
    for _ in range(num_remain_backward):
        order.append(next_backward_chunk)
        num_second_chunks += next_backward_chunk

        next_backward_chunk = 1 - next_backward_chunk
        if (
            next_backward_chunk == 1
            and num_second_chunks == num_microbatches
        ):
            next_backward_chunk = 0

    return order


class PipelineOffloadManager:
    OFFLOAD_MGR = None

    @classmethod
    def get_instance(cls):
        if cls.OFFLOAD_MGR is None:
            cls.OFFLOAD_MGR = PipelineOffloadManager()
        return cls.OFFLOAD_MGR

    def __init__(self):
        self._queue = [deque(), deque()]

        # allocate streams and events for synchronization
        self._d2h_stream = torch.cuda.Stream()
        self._h2d_stream = torch.cuda.Stream()
        self.reset()

    @property
    def d2h_stream(self):
        return self._d2h_stream

    @property
    def h2d_stream(self):
        return self._h2d_stream

    def reset(self):
        # set_ideal_affinity_for_current_gpu()
        self._inside_context = False
        self._cur_forward_chunk = None
        self._cur_backward_chunk = None
        self._first_second_chunk = True
        self._backward_chunk_order = get_backward_chunk_order(get_num_microbatches())

    def push(self, handler, is_first_chunk=True):
        index = 0 if is_first_chunk else 1
        self._queue[index].append(handler)

    def pop(self):
        assert self.size()
        while self._backward_chunk_order:
            backward_chunk = self._backward_chunk_order.popleft()
            self._cur_backward_chunk = self._queue[backward_chunk].popleft()
            if not chunk_handler.is_empty_chunk():
                break

    def front(self):
        from dcu_megatron.training.utils import print_rank_message

        if not self.size():
            return None

        backward_chunk_index = [0, 0]
        for backward_chunk in self._backward_chunk_order:
            if not len(self._queue[backward_chunk]):
                return None

            chunk_handler = self._queue[backward_chunk][backward_chunk_index[backward_chunk]]
            if not self._cur_backward_chunk.is_empty_chunk():
                return chunk_handler

            backward_chunk_index[backward_chunk] += 1

        return None

    def size(self):
        return len(self._backward_chunk_order)

    def init_model_chunk_offload_handler(
            self,
            is_first_chunk,
            min_offloaded_tensor_size=1024 * 1024
        ):
        first_second_chunk = self._first_second_chunk and (not is_first_chunk)
        # If the model chunk contains only the dense layers, initialize a null chunk handler.
        cur_chunk = ChunkOffloadHandler(first_second_chunk, min_offloaded_tensor_size=min_offloaded_tensor_size)
        # save for latter push
        if first_second_chunk:
            self._first_second_chunk = False
        self.push(cur_chunk, is_first_chunk=is_first_chunk)
        self._cur_forward_chunk = cur_chunk

    def set_last_layer(self, is_last_layer):
        self._cur_forward_chunk.is_last_layer = is_last_layer

    def cur_forward_chunk(self):
        return self._cur_forward_chunk

    def cur_backward_chunk(self):
        return self._cur_backward_chunk

    def __enter__(self):
        from megatron.core.extensions.transformer_engine import cpu_offload

        if cpu_offload is not None:
            cpu_offload.CPUOffloadEnabled = True
        else:
            raise RuntimeError("TE CPU offload is not available")
        self.inside_context = True

        torch._C._autograd._push_saved_tensors_default_hooks(
            self.on_save_for_backward, self.on_get_saved_tensor
        )

    def __exit__(self, *args: Any):
        from megatron.core.extensions.transformer_engine import cpu_offload

        if cpu_offload is not None:
            cpu_offload.CPUOffloadEnabled = False
        else:
            raise RuntimeError("TE CPU offload is not available")
        self.inside_context = False

        torch._C._autograd._pop_saved_tensors_default_hooks()

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        assert self.inside_context, "Must be inside offload context"
        return self.cur_forward_chunk().tensor_push(tensor)

    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        """get hook"""
        return self.cur_backward_chunk().tensor_pop(saved_state)


class ChunkOffloadHandler():
    @staticmethod
    def offload(src_tensor, pin_memory=True):
        """Offload."""
        if not src_tensor.is_contiguous():
            src_tensor = src_tensor.contiguous()

        cpu_backup = torch.empty(
            src_tensor.size(),
            dtype=src_tensor.dtype,
            layout=src_tensor.layout,
            device="cpu",
            pin_memory=pin_memory,
        )

        cpu_backup.copy_(src_tensor, non_blocking=pin_memory)
        state = (src_tensor.device, cpu_backup)
        return state

    @staticmethod
    def reload(state, non_blocking=None):
        """Reload."""
        dev, cpu_backup = state
        if non_blocking is None:
            non_blocking = cpu_backup.is_pinned()

        return cpu_backup.to(dev, non_blocking=non_blocking)

    def __init__(self, is_first_second_chunk, min_offloaded_tensor_size):
        # Data Structure to maintain reference to activation tensors
        self._tensor_tag_to_state = {}
        # Data structure to hold the FP8/MXFP8 tensor objects
        self.fp8_tensor_object_map = {}
        self.float8_transpose_cache_valid = {}

        self._is_first_second_chunk = is_first_second_chunk

        self._offloaded_group_index = 0
        self._groups_to_offload = []
        self._groups_to_reload = []
        self._tensor_count_current_group = 0

        self.torch_tensor_count = 0
        self.d2h_stream = PipelineOffloadManager.get_instance().d2h_stream
        self.h2d_stream = PipelineOffloadManager.get_instance().h2d_stream
        self._offload_events = {}
        self._reload_events = {}
        self.min_offloaded_tensor_size = min_offloaded_tensor_size
        self.is_last_layer = False

        self.module_release_func_map = dict()

    def is_empty_chunk(self):
        """Check if this chunk has no tensors to manage."""
        return len(self._tensor_tag_to_state) == 0

    def is_first_last_layer(self):
        """Do not offload the last layer of the last pp stage."""
        return self._is_first_second_chunk and self.is_last_layer

    def tensor_push(self, tensor):
        def tensor_offloading_checker(device_tensor):
            if not self.should_bulk_offload():
                return False

            return self.tensor_need_offloading_checker(device_tensor)

        torch_stray_tensor = isinstance(
            tensor,
            (
                torch._subclasses.fake_tensor.FakeTensor,
                torch._subclasses.functional_tensor.FunctionalTensor,
            ),
        )

        if not torch_stray_tensor:
            # obtain a unique tensor tag
            tensor_tag = (self._offloaded_group_index, self._tensor_count_current_group)
            self._tensor_count_current_group += 1
            assert tensor_tag not in self._tensor_tag_to_state, "Duplicate tensor tag"

            is_quantized_tensor = isinstance(tensor, QuantizedTensorBase)
            if is_quantized_tensor and tensor_offloading_checker(tensor):
                tensor_list, _ = tensor.prepare_for_saving()

                self._tensor_tag_to_state[tensor_tag] = []
                for t in tensor_list:
                    if t is not None:
                        t.offloading_activation = True
                    self._tensor_tag_to_state[tensor_tag].append(t)

                tensor.clear()
                self.fp8_tensor_object_map[tensor_tag] = tensor
                if isinstance(tensor, Float8Tensor):
                    self.float8_transpose_cache_valid[tensor_tag] = getattr(tensor, "_transpose_invalid")
            else:
                self._tensor_tag_to_state[tensor_tag] = tensor
        else:
            tensor_tag = (-1, self.torch_tensor_count)
            self.torch_tensor_count += 1
            self._tensor_tag_to_state[tensor_tag] = tensor

        return tensor_tag

    def tensor_pop(self, tensor_tag):
        assert tensor_tag in self._tensor_tag_to_state, f"Tag {tensor_tag} not found"
        tensor = self._tensor_tag_to_state.pop(tensor_tag)
        assert not isinstance(tensor, tuple)
        return tensor

    def tensor_need_offloading_checker(self, tensor):
        """Check if the tensor needs to be offloaded."""
        if tensor is None:
            return False
        if tensor.numel() < self.min_offloaded_tensor_size::
            return False
        if hasattr(tensor, "offloading_activation") and not tensor.offloading_activation:
            return False
        return True

    def bulk_offload_group(self, group_to_offload):
        """offload a group of tensors recorded in tensor_push().
        """
        assert not self.is_first_last_layer()
        group_id_to_offload, name = group_to_offload
        torch.cuda.nvtx.range_push("activation offloading " + name)
        with torch.cuda.stream(self.d2h_stream):
            for tensor_tag, state in self._tensor_tag_to_state.items():
                group_id, _ = tensor_tag
                if group_id == group_id_to_offload:
                    assert not isinstance(state, tuple), "Tensor already offloaded"

                    is_quantized_tensor = isinstance(state, list)

                    if is_quantized_tensor:
                        tensor_list = state
                        self._tensor_tag_to_state[tensor_tag] = []
                    else:
                        tensor_list = [state]

                    for tensor_on_device in tensor_list:
                        to_offload_tensor = self.tensor_need_offloading_checker(tensor_on_device)
                        # if offload, return the reference to cpu copy
                        if to_offload_tensor:
                            state = self.offload(tensor_on_device)
                            # TODO: check if we really need it.
                            # Record the last offloading event for this group,
                            # which is used to avoid reloading before offloading.
                            event = torch.cuda.Event()
                            event.record(self.d2h_stream)
                            self._offload_events[name] = event
                            tensor_on_device.record_stream(self.d2h_stream)

                        if is_quantized_tensor:
                            self._tensor_tag_to_state[tensor_tag].append(state if to_offload_tensor else tensor_on_device)
                        else:
                            self._tensor_tag_to_state[tensor_tag] = state
        torch.cuda.nvtx.range_pop()

    def get_offload_event(self, name):
        return self._offload_events.get(name, None)

    def get_reload_event(self, name):
        return self._reload_events.get(name, None)

    def bulk_reload_group(self, group_to_reload):
        """Bulk reload group."""
        found_reload_group = False
        group_id_to_reload, name = group_to_reload
        torch.cuda.nvtx.range_push(name)
        with torch.cuda.stream(self.h2d_stream):
            # move back tensors
            for tensor_label, state in self._tensor_tag_to_state.items():
                group_id, _ = tensor_label
                if group_id == group_id_to_reload:
                    found_reload_group = True
                    event = self.get_offload_event(name)
                    if isinstance(state, tuple):
                        # make sure the tensor is already offloaded to cpu before reloading it.
                        torch.cuda.current_stream().wait_event(event)
                        recovered_tensor = self.reload(state)
                        event.record(self.h2d_stream)
                        self._reload_events[name] = event
                        self._tensor_tag_to_state[tensor_label] = recovered_tensor
                    elif isinstance(state, list):
                        tensor_list = []
                        for state_tuple in state:
                            if isinstance(state_tuple, tuple):
                                tensor_list.append(self.reload(state_tuple))
                            else:
                                tensor_list.append(state_tuple)

                        _ = self.fp8_tensor_object_map[tensor_label].restore_from_saved(
                            tensor_list
                        )
                        if isinstance(self.fp8_tensor_object_map[tensor_label], Float8Tensor):
                            self.fp8_tensor_object_map[tensor_label]._transpose_invalid = (
                                self.float8_transpose_cache_valid.pop(tensor_label)
                            )
                        self._tensor_tag_to_state[tensor_label] = self.fp8_tensor_object_map.pop(tensor_label)
        torch.cuda.nvtx.range_pop()
        return found_reload_group

    def pre_reload_last_layer(self):
        """Pre-reload the last layer of the next model chunk."""
        assert not self._is_first_second_chunk, "Should not pre-reload first chunk"

        if len(self._groups_to_reload) > 0:
            if self.bulk_reload_group(self._groups_to_reload[-1]):
                self._groups_to_reload.pop()

    def should_bulk_offload(self):
        """Check if the chunk should be offloaded."""
        # first backward chunk
        if self.is_first_last_layer():
            return False

        # if next backward chunk is this chunk
        next_backward_chunk = PipelineOffloadManager.get_instance().front()
        if next_backward_chunk is not None and next_backward_chunk is self:
            if self.is_last_layer:
                return False

        return True

    def bulk_offload(self, forced_released_tensors, delay_release_module=None):
        if self.should_bulk_offload():
            group_to_offload = self._groups_to_offload.pop()
            name = group_to_offload[1]
            self._groups_to_reload.append(group_to_offload)
            self.bulk_offload_group(group_to_offload)

            def release_func():
                if len(forced_released_tensors) > 0:
                    cur_stream = torch.cuda.current_stream()
                    for release_tensor in forced_released_tensors:
                        release_tensor.record_stream(cur_stream)
                        release_tensor.untyped_storage().resize_(0)
            if delay_release_module is None:
                release_func()
            else:
                self.module_release_func_map[delay_release_module] = release_func

    def on_group_commit_forward(self, forced_released_tensors, delay_release_module=None):
        """Offload a group of tensors."""

        # wait for the compute stream for offloading
        self.d2h_stream.wait_stream(torch.cuda.current_stream())
        self.bulk_offload(forced_released_tensors, delay_release_module)

    def bulk_reload(self):
        if len(self._groups_to_reload) > 0:
            # load next layer
            if self.bulk_reload_group(self._groups_to_reload[-1]):
                self._groups_to_reload.pop()
        else:
            # load the last layer of one backward chunk in advance
            next_backward_chunk = PipelineOffloadManager.get_instance().front()
            if next_backward_chunk is not None:
                next_backward_chunk.pre_reload_last_layer()

    def on_group_commit_backward(self, name):
        """Prepare for reloading the next group of tensors."""
        cur_backward_chunk = PipelineOffloadManager.get_instance().cur_backward_chunk()
        if not cur_backward_chunk is self:
            PipelineOffloadManager.get_instance().pop()
        cur_backward_chunk = PipelineOffloadManager.get_instance().cur_backward_chunk()
        assert cur_backward_chunk is self
        # make sure the reloading jobs for current computation are done.
        event = self.get_reload_event(name)
        if event is not None:
            torch.cuda.current_stream().wait_event(event)
        self._offloaded_group_index = self._offloaded_group_index - 1

    def on_group_start_forward(self, name):
        """Prepare for offloading the next group of tensors."""
        self._offloaded_group_index = self._offloaded_group_index + 1
        self._tensor_count_current_group = 0
        self._groups_to_offload.append((self._offloaded_group_index, name))

    def on_group_start_backward(self):
        """Reload the next group of tensors."""
        self.h2d_stream.wait_stream(torch.cuda.current_stream())
        self.bulk_reload()


class GroupCommitFunction(torch.autograd.Function):
    """this is a dummy op with output identical to input.
    However, it is necessary for marking a timepoint for offload handler to
    accomplish all synchronizations. Implementing it as a function is necessary
    because we need to actions in both forward and backward.
    """

    @staticmethod
    def forward(ctx, *args):
        # pylint: disable=missing-function-docstring
        delay_release_module = args[-1]
        forced_released_tensors = args[-2]
        name = args[-3]
        cpu_offload_handler = args[-4]
        tensor = args[:-4]
        cpu_offload_handler.on_group_commit_forward(forced_released_tensors, delay_release_module=delay_release_module)
        ctx.cpu_offload_handler = cpu_offload_handler
        ctx.name = name

        # return the identical tensor
        return tensor

    @staticmethod
    def backward(ctx, *grad_output):
        # pylint: disable=missing-function-docstring

        cpu_offload_handler = ctx.cpu_offload_handler
        cpu_offload_handler.on_group_commit_backward(ctx.name)
        return grad_output + (None, None, None, None)


def fine_grained_offloading_group_commit(*tensor, name, forced_released_tensors=[], delay_release_module=None):
    """Specify the tensors to be released after offloading.
    forced_released_tensors is a list of tensors to be released after offloading.
    The tensors will be untyped_storage().resize_(0) after offloading.
    Note: specify the tensors only when they are not automatically released by torch gc.
    """
    cur_forward_chunk = PipelineOffloadManager.get_instance().cur_forward_chunk()
    return FineGrainedOffloadingGroupCommitFunction.apply(
        *tensor, cur_forward_chunk, name, forced_released_tensors, delay_release_module
    )


class FineGrainedOffloadingGroupStartFunction(torch.autograd.Function):
    """this is a dummy op with output identical to input.
    However, it is necessary for marking a timepoint for offload handler to
    accomplish all synchronizations. Implementing it as a function is necessary
    because we need to actions in both forward and backward.
    """

    @staticmethod
    def forward(ctx, tensor, cpu_offload_handler, name):
        # pylint: disable=missing-function-docstring
        ctx.cpu_offload_handler = cpu_offload_handler

        cpu_offload_handler.on_group_start_forward(name)
        # return the identical tensor
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        # pylint: disable=missing-function-docstring
        cpu_offload_handler = ctx.cpu_offload_handler
        cpu_offload_handler.on_group_start_backward()
        return grad_output, None, None


def fine_grained_offloading_group_start(tensor, name=None):
    cur_forward_chunk = PipelineOffloadManager.get_instance().cur_forward_chunk()
    return FineGrainedOffloadingGroupStartFunction.apply(tensor, cur_forward_chunk, name)


def get_fine_grained_offloading_context(flag):
    return PipelineOffloadManager.get_instance() if flag else nullcontext()


def fine_grained_offloading_set_last_layer(is_last_layer):
    """Set the last layer flag."""
    PipelineOffloadManager.get_instance().set_last_layer(is_last_layer)


def fine_grained_offloading_init_chunk_handler(is_first_chunk, min_offloaded_tensor_size):
    """Initialize the chunk handler, called at the start of a microbatch forward pass."""
    PipelineOffloadManager.get_instance().init_model_chunk_offload_handler(
        is_first_chunk, min_offloaded_tensor_size
    )


def fine_grained_offloading_reset():
    """Reset the chunk handler, called at the start of a training iteration."""
    PipelineOffloadManager.get_instance().reset()

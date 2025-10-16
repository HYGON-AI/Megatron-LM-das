import copy
from collections import deque, defaultdict
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
import transformer_engine as te
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.cpu_offload import AsyncDoubleBufferGroupOffloadHandler
from transformer_engine.pytorch.tensor.quantized_tensor import QuantizedTensorBase

from megatron.core import parallel_state


_LAYER_INDEX = None

def set_layer_index(layer_index):
    global _LAYER_INDEX
    _LAYER_INDEX = layer_index


def get_layer_index():
    global _LAYER_INDEX
    assert _LAYER_INDEX is not None
    return _LAYER_INDEX

# cpu offload for pipeline
MIN_OFFLOADED_TENSOR_SIZE = 1024 * 1024

def set_ideal_affinity_for_current_gpu():
    import cuda.cuda
    import cuda.cudart
    import pynvml
    import uuid
    err, device_id = cuda.cudart.cudaGetDevice()
    assert err == cuda.cudart.cudaError_t.cudaSuccess
    err, device_uuid = cuda.cuda.cuDeviceGetUuid(device_id)
    assert err == cuda.cuda.CUresult.CUDA_SUCCESS
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByUUID("GPU-" + str(uuid.UUID(bytes=device_uuid.bytes)))
    pynvml.nvmlDeviceSetCpuAffinity(handle)


class PipelineOffloadManager:
    OFFLOAD_MGR = None

    @classmethod
    def get_instance(cls):
        if cls.OFFLOAD_MGR is None:
            cls.OFFLOAD_MGR = PipelineOffloadManager()
        return cls.OFFLOAD_MGR

    def __init__(self):
        self._queue = deque()
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is None:
            self._vpp = 1
        else:
            self._vpp = parallel_state.get_virtual_pipeline_model_parallel_world_size()

        # cache vpp - 1 stages
        self._stages = [[] for _ in range(self._vpp)]
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
        self._first_last_vpp_rank = True

    def flush(self):
        # put into the queue in the backward order
        if len(self._stages[0]) == len(self._stages[-1]):
            lens = [len(e) for e in self._stages]
            assert min(lens) == max(lens)
            self._stages[-1] = []
            for chunks in reversed(self._stages):
                for chunk in chunks:
                    self.push(chunk)
            for i in range(self._vpp):
                self._stages[i] = []

    def push(self, handler):
        self._queue.append(handler)

    def pop(self):
        assert self.size()
        while self._queue:
            self._cur_backward_chunk = self._queue.popleft()
            if not isinstance(self._cur_backward_chunk, NullChunkOffloadHandler):
                break

    def front(self):
        if not len(self._queue):
            return None
        for chunk_handler in self._queue:
            if not isinstance(chunk_handler, NullChunkOffloadHandler):
                return chunk_handler
        return None

    def size(self):
        return len(self._queue)

    def reset_chunk_handler(self, num_layer, vp_stage, offload=True, num_dense_layer=0, last_stage_is_loss=False):
        if vp_stage is None:
            cur_vpp_rank = 0
        else:
            cur_vpp_rank = vp_stage

        if last_stage_is_loss:
            from megatron.core import parallel_state
            vpp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()
            # skip the last stage
            if cur_vpp_rank == vpp_size - 1:
                return
            # reduce the vpp size
            if self._vpp == vpp_size:
                self._vpp -= 1
                self._stages = self._stages[:-1]

        first_last_vpp_rank = self._first_last_vpp_rank
        # rewind
        if cur_vpp_rank == self._vpp - 1:
            self.flush()
        first_last_vpp_rank = first_last_vpp_rank and (cur_vpp_rank == self._vpp - 1)
        # If the model chunk contains only the dense layers, initialize a null chunk handler.
        if num_layer <= num_dense_layer:
            cur_chunk = NullChunkOffloadHandler(num_layer, first_last_vpp_rank, offload)
        else:
            cur_chunk = ChunkOffloadHandler(num_layer, first_last_vpp_rank, offload)
        # save for latter push
        self._stages[cur_vpp_rank].append(cur_chunk)
        if cur_vpp_rank == self._vpp - 1:
            self._first_last_vpp_rank = False
            self.push(cur_chunk)
            self.flush()
        self._cur_forward_chunk = cur_chunk
        cur_chunk.vpp_rank = cur_vpp_rank

    def set_last_layer(self, is_last_layer):
        self._cur_forward_chunk.is_last_layer = is_last_layer

    def cur_forward_chunk(self):
        return self._cur_forward_chunk

    def cur_backward_chunk(self):
        return self._cur_backward_chunk

    def __enter__(self):
        self.OFFLOAD_MGR
        self.inside_context = True

        if not isinstance(self.cur_forward_chunk(), NullChunkOffloadHandler):
            torch._C._autograd._push_saved_tensors_default_hooks(
                self.on_save_for_backward, self.on_get_saved_tensor
            )

    def __exit__(self, *args: Any):
        self.inside_context = False
        if not isinstance(self.cur_forward_chunk(), NullChunkOffloadHandler):
            torch._C._autograd._pop_saved_tensors_default_hooks()

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        assert self.inside_context
        return self.cur_forward_chunk().tensor_push(tensor)

    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        """get hook"""
        return self.cur_backward_chunk().tensor_pop(saved_state)


class TECpuOffloadContextManager:
    """A reusable context manager for switch vpp stage"""

    def __init__(self, cpu_offloading):
        self.cpu_offloading = cpu_offloading

    def __enter__(self):
        self.origin_cpu_offloading = te.pytorch.cpu_offload.get_cpu_offloading()
        te.pytorch.cpu_offload.set_cpu_offloading(self.cpu_offloading)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        te.pytorch.cpu_offload.set_cpu_offloading(self.origin_cpu_offloading)


class ChunkOffloadHandler(AsyncDoubleBufferGroupOffloadHandler):
    @staticmethod
    def offload(src_tensor, pin_memory=True):
        """Offload."""
        if src_tensor is None:
            return None

        cpu_backup = torch.empty(
            src_tensor.size(),
            dtype=src_tensor.dtype,
            layout=src_tensor.layout,
            device="cpu",
            pin_memory=pin_memory,
        )

        if not src_tensor.is_contiguous():
            src_tensor = src_tensor.contiguous()

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

    def __init__(self, num_layer, is_first_last_vpp_chunk, offload=True):
        self._num_layers = num_layer
        # Data Structure to maintain reference to activation tensors
        self._tensor_tag_to_state = {}
        # Data structure to hold the FP8/MXFP8 tensor objects
        self.fp8_tensor_object_map = {}
        self.float8_transpose_cache_valid = {}
        # Tracking the number of layers offloaded
        # self._offloaded_group_count = 0
        self._is_first_last_vpp_chunk = is_first_last_vpp_chunk

        self._offloaded_group_index = 0
        self._groups_to_offload = []
        self._groups_to_reload = []
        self._layer_index = 0
        self._tensor_count_current_group = 0
        self.multi_input_offload_count = False
        # self.offload_count_per_layer = defaultdict(int)

        self.torch_tensor_count = 0
        self.d2h_stream = PipelineOffloadManager.get_instance().d2h_stream
        self.h2d_stream = PipelineOffloadManager.get_instance().h2d_stream
        self._offload_events = {}
        self._reload_events = {}
        self.do_offload = offload
        self.is_last_layer = False

        self.module_release_func_map = dict()

    def is_first_last_layer(self):
        """Do not offload the last layer of the last pp stage."""
        return self._is_first_last_vpp_chunk and self.is_last_layer

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
            assert tensor_tag not in self._tensor_tag_to_state

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
        assert tensor_tag in self._tensor_tag_to_state, f"{tensor_tag}, {self._tensor_tag_to_state.keys()}"
        tensor = self._tensor_tag_to_state.pop(tensor_tag)
        assert not isinstance(tensor, tuple)
        return tensor

    def tensor_need_offloading_checker(self, tensor):
        """Check if the tensor needs to be offloaded."""
        if tensor is None:
            return False
        if tensor.numel() < MIN_OFFLOADED_TENSOR_SIZE:
            return False
        if hasattr(tensor, "offloading_activation") and not tensor.offloading_activation:
            return False
        return True

    def bulk_offload_group(self, group_to_offload):
        """offload a group of tensors recorded in tensor_push().
        """
        if not self.do_offload:
            return

        assert not self.is_first_last_layer()
        group_id_to_offload, name = group_to_offload
        torch.cuda.nvtx.range_push(name)
        with torch.cuda.stream(self.d2h_stream):
            for tensor_tag, state in self._tensor_tag_to_state.items():
                group_id, _ = tensor_tag
                if group_id == group_id_to_offload:
                    assert not isinstance(state, tuple)

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
        if name in self._offload_events:
            return self._offload_events[name]
        else:
            return None

    def get_reload_event(self, name):
        if name in self._reload_events:
            return self._reload_events[name]
        else:
            return None

    def bulk_reload_group(self, group_to_reload):
        """Bulk reload group."""
        if not self.do_offload:
            return
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
        if not self.do_offload:
            return
        assert not self._is_first_last_vpp_chunk

        if len(self._groups_to_reload) > 0:
            if self.bulk_reload_group(self._groups_to_reload[-1]):
                self._groups_to_reload.pop()

    def should_bulk_offload(self):
        """Check if the chunk should be offloaded."""
        if not self.do_offload:
            return False
        # first backward chunk
        if self.is_first_last_layer():
            return False

        # if next backward chunk is this chunk (for last pp stage)
        next_backward_chunk = PipelineOffloadManager.get_instance().get_instance().front()
        if next_backward_chunk is not None and next_backward_chunk is self:
            if self.is_last_layer:
                return False

        return True

    def bulk_offload(self, release_tensors, delay_release_module=None):
        if self.should_bulk_offload():
            group_to_offload = self._groups_to_offload.pop()
            name = group_to_offload[1]
            self._groups_to_reload.append(group_to_offload)
            self.bulk_offload_group(group_to_offload)

            def release_func():
                if len(release_tensors) > 0:
                    cur_stream = torch.cuda.current_stream()
                    for release_tensor in release_tensors:
                        release_tensor.record_stream(cur_stream)
                        release_tensor.untyped_storage().resize_(0)
            if delay_release_module is None:
                release_func()
            else:
                self.module_release_func_map[delay_release_module] = release_func

    def on_group_commit_forward(self, release_tensors, delay_release_module=None):
        """Offload a group of tensors."""

        # wait for the compute stream for offloading
        self.d2h_stream.wait_stream(torch.cuda.current_stream())
        self.bulk_offload(release_tensors, delay_release_module)

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
        """Prepare for reloadingthe next group of tensors."""
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

    def register_offload_tensor(self, tensors):
        self.multi_input_offload_count = True
        if isinstance(tensors, list):
            for tensor in tensors:
                self._offload_tensor_ptrs.append(tensor.data_ptr())
        else:
            self._offload_tensor_ptrs.append(tensors.data_ptr())

    def is_registered_tensor(self, tensor_ptr: int) -> bool:
        if len(self._offload_tensor_ptrs) == 0:
            return False
        is_registered = tensor_ptr == self._offload_tensor_ptrs[0]
        if is_registered:
            self._offload_tensor_ptrs.popleft()
        return is_registered


class NullChunkOffloadHandler(ChunkOffloadHandler):
    pass


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
        release_tensors = args[-2]
        name = args[-3]
        cpu_offload_handler = args[-4]
        tensor = args[:-4]
        if not isinstance(cpu_offload_handler, NullChunkOffloadHandler):
            cpu_offload_handler.on_group_commit_forward(release_tensors, delay_release_module=delay_release_module)
        ctx.cpu_offload_handler = cpu_offload_handler
        ctx.name = name

        # return the identical tensor
        return tensor

    @staticmethod
    def backward(ctx, *grad_output):
        # pylint: disable=missing-function-docstring

        cpu_offload_handler = ctx.cpu_offload_handler
        if not isinstance(cpu_offload_handler, NullChunkOffloadHandler):
            cpu_offload_handler.on_group_commit_backward(ctx.name)
        return grad_output + (None, None, None, None)


def group_prefetch_offload_commit(*tensor, name, release_tensors=[], delay_release_module=None):
    """Specify the tensors to be released after offloading.
    release_tensors is a list of tensors to be released after offloading.
    The tensors will be untyped_storage().resize_(0) after offloading.
    Note: specify the tensors only when they are not automatically released by torch gc.
    """
    cur_forward_chunk = PipelineOffloadManager.get_instance().cur_forward_chunk()
    return GroupCommitFunction.apply(*tensor, cur_forward_chunk, name, release_tensors, delay_release_module)


class GroupStartFunction(torch.autograd.Function):
    """this is a dummy op with output identical to input.
    However, it is necessary for marking a timepoint for offload handler to
    accomplish all synchronizations. Implementing it as a function is necessary
    because we need to actions in both forward and backward.
    """

    @staticmethod
    def forward(ctx, tensor, cpu_offload_handler, name):
        # pylint: disable=missing-function-docstring
        ctx.cpu_offload_handler = cpu_offload_handler

        if not isinstance(cpu_offload_handler, NullChunkOffloadHandler):
            cpu_offload_handler.on_group_start_forward("activation offloading " + name)
        # return the identical tensor
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        # pylint: disable=missing-function-docstring
        cpu_offload_handler = ctx.cpu_offload_handler
        if not isinstance(cpu_offload_handler, NullChunkOffloadHandler):
            cpu_offload_handler.on_group_start_backward()
        return grad_output, None, None


def group_prefetch_offload_start(tensor, name=None):
    cur_forward_chunk = PipelineOffloadManager.get_instance().cur_forward_chunk()
    return GroupStartFunction.apply(tensor, cur_forward_chunk, name)


def get_offload_context(config):
    if config.fine_grained_activation_offloading:
        return PipelineOffloadManager.get_instance()
    else:
        return nullcontext()


def offload_checker_ctx(config, offload_checker_func):
    if config.fine_grained_activation_offloading:
        return (
            PipelineOffloadManager.get_instance()
            .cur_forward_chunk()
            .offload_checker_ctx(offload_checker_func)
        )
    return nullcontext()

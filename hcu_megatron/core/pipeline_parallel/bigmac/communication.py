# Copyright (c) 2026 Dots-Infra
# SPDX-License-Identifier: MIT
# Some of this code was adopted from https://github.com/Dots-Infra/BigMac.git

import copy
import os
import torch
import torch.distributed as dist
from megatron.core import DistributedDataParallel as DDP
from megatron.core.distributed.param_and_grad_buffer import BufferType
from collections import deque


def configure_vpp_p2p(communicator):
    # Give VPP's asynchronous calls a compatible, executor-local transport.
    if getattr(communicator.config, "use_ring_exchange_p2p", False):
        raise ValueError("BigMac VPP requires asynchronous P2P; use_ring_exchange_p2p is unsupported")
    # MCore's batched path requires wait_on_reqs=True, while the VPP executor
    # deliberately launches ahead of compute with overlap_p2p_comm=True.
    config = copy.copy(communicator.config)
    config.batch_p2p_comm = False
    config.overlap_p2p_comm = True
    communicator.config = config


def configure_qwen_p2p(communicator):
    mode = os.getenv("MEGATRON_BIGMAC_P2P_MODE", "unbatched").strip().lower()
    if mode not in ("unbatched", "native"):
        raise ValueError("MEGATRON_BIGMAC_P2P_MODE must be unbatched or native")
    config = communicator.config
    if mode == "unbatched" and not getattr(config, "use_ring_exchange_p2p", False):
        config = copy.copy(config)
        config.batch_p2p_comm = False
        communicator.config = config
    if getattr(config, "use_ring_exchange_p2p", False):
        return "ring"
    return "batched" if config.batch_p2p_comm else "unbatched"


def named_handles(handles, names):
    if handles is None:
        return {}
    if isinstance(handles, dict):
        return handles
    if isinstance(handles, (list, tuple)) and len(handles) == len(names):
        return dict(zip(names, handles))
    raise RuntimeError("Unexpected Megatron P2P handles; cannot safely release pipeline tensors")


class P2PHandleAdapter:
    def __init__(self, communicator):
        self.communicator = communicator

    def __getattr__(self, name):
        return getattr(self.communicator, name)

    def send_forward_recv_forward(self, output_tensor, recv_prev, tensor_shape, overlap_p2p_comm=False):
        result = self.communicator.send_forward_recv_forward(
            output_tensor, recv_prev=recv_prev, tensor_shape=tensor_shape,
            overlap_p2p_comm=overlap_p2p_comm,
        )
        if not overlap_p2p_comm:
            return result
        tensor, handles = result
        # _batched_p2p_ops order: send_prev, recv_prev, send_next, recv_next.
        names = (["recv_prev"] if recv_prev else []) + (["send_next"] if output_tensor is not None else [])
        return tensor, named_handles(handles, names)

    def send_backward_recv_backward(self, input_tensor_grad, recv_next, tensor_shape, overlap_p2p_comm=False):
        result = self.communicator.send_backward_recv_backward(
            input_tensor_grad, recv_next=recv_next, tensor_shape=tensor_shape,
            overlap_p2p_comm=overlap_p2p_comm,
        )
        if not overlap_p2p_comm:
            return result
        tensor, handles = result
        names = (["send_prev"] if input_tensor_grad is not None else []) + (["recv_next"] if recv_next else [])
        return tensor, named_handles(handles, names)


class _KeepAliveWork:
    # Own padding buffers until their asynchronous collective has completed.
    def __init__(self, work, *buffers):
        self.work = work
        self.buffers = buffers

    def wait(self):
        if self.work is not None:
            self.work.wait()
        self.buffers = ()


def _dynamic(meta):
    return meta.shape is not None and any(dim == -1 for dim in meta.shape)


def _padded_shape(shapes):
    # NCCL/Gloo need equal counts, including on a rank with no images.
    return tuple(max(1, *(shape[i] for shape in shapes)) for i in range(len(shapes[0])))


def _pad_tensor(data, shape):
    if tuple(data.shape) == tuple(shape):
        return data.contiguous()
    padded = data.new_zeros(shape)
    padded[tuple(slice(0, dim) for dim in data.shape)] = data
    return padded


def _unpad_tensor(data, shape):
    return data[tuple(slice(0, dim) for dim in shape)]


def _gather_tensor_shapes(local_shapes, pp_group, device):
    # Fixed-size integer header: one collective, no pickle/size exchange.
    #
    # Tensor ranks and field order come from the shared encoder contract; only
    # dimension values vary. One host read is needed to allocate the payloads.
    header = torch.tensor(
        [dim for shape in local_shapes.values() for dim in shape],
        dtype=torch.int64, device=device,
    )
    headers = [torch.empty_like(header) for _ in range(dist.get_world_size(pp_group))]
    dist.all_gather(headers, header, group=pp_group, async_op=False)
    gathered = []
    for values in torch.stack(headers).cpu().tolist():
        shapes, offset = {}, 0
        for name, shape in local_shapes.items():
            shapes[name] = tuple(values[offset:offset + len(shape)])
            offset += len(shape)
        gathered.append(shapes)
    return gathered


def release_tensor(tensor):
    if tensor is None:
        return
    tensor.data = torch.empty((1,), device=tensor.device, dtype=tensor.dtype)


def release_grad_buffer(model, buffer):
    if buffer.grad_data is None:
        return
    buffer.grad_data = None
    cur_bucket_id = -1
    for param in buffer.param_index_map.keys():
        data_start_index, data_end_index, bucket_id = buffer.param_index_map[param]
        if bucket_id != cur_bucket_id:
            assert bucket_id == (cur_bucket_id + 1)
            bucket = buffer.buckets[bucket_id]
            bucket.grad_data = None
            cur_bucket_id = bucket_id
        param.main_grad = None


def rebuild_grad_buffer(model, buffer):
    if buffer.grad_data is not None:
        return
    buffer.grad_data = torch.empty(
        buffer.numel,
        dtype=buffer.grad_dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    buffer.grad_data.zero_()
    cur_bucket_id = -1
    for param in buffer.param_index_map.keys():
        data_start_index, data_end_index, bucket_id = buffer.param_index_map[param]
        if bucket_id != cur_bucket_id:
            assert bucket_id == (cur_bucket_id + 1)
            bucket = buffer.buckets[bucket_id]
            bucket_start_index, bucket_end_index = buffer.bucket_indices[bucket_id]
            bucketed_grad_data = buffer._get(
                torch.Size([bucket_end_index - bucket_start_index]),
                bucket_start_index,
                buffer_type=BufferType.GRAD,
            )
            bucket.grad_data = bucketed_grad_data
            cur_bucket_id = bucket_id
        param.main_grad = buffer._get(
            param.data.shape, data_start_index, buffer_type=BufferType.GRAD
        )


def release_model_grad_buffer(model):
    if isinstance(model, list):
        for m in model:
            release_model_grad_buffer(m)
        return
    if not isinstance(model, DDP):
        return
    for buffer in model.buffers:
        release_grad_buffer(model, buffer)
    if hasattr(model, "expert_parallel_buffers"):
        for buffer in model.expert_parallel_buffers:
            release_grad_buffer(model, buffer)


def rebuild_model_grad_buffer(model):
    if isinstance(model, list):
        for m in model:
            rebuild_model_grad_buffer(m)
        return
    if not isinstance(model, DDP):
        return
    for buffer in model.buffers:
        rebuild_grad_buffer(model, buffer)
    if hasattr(model, "expert_parallel_buffers"):
        for buffer in model.expert_parallel_buffers:
            rebuild_grad_buffer(model, buffer)


def all_gather_along_pp(datas, pp_group, data_metas, comm_requires_grad_only=False):
    # general allgather along pp.
    # if comm_requires_grad_only is set, only communicate the data with
    # requires_grad=True; other entries are set to None.
    handles = []
    pp_size = dist.get_world_size(pp_group)
    pp_rank = dist.get_rank(pp_group)
    gathered_datas = [{} for _ in range(pp_size)]

    # Exchange all dynamic shapes together, before launching any tensor work.
    # A blocking metadata collective between tensor collectives would serialize
    # the otherwise asynchronous handoff, even for fixed-size image batches.
    local_shapes = {
        meta.name: tuple(data.shape)
        for data, meta in zip(datas, data_metas)
        if _dynamic(meta) and not meta.is_none and not meta.is_object
        and (not comm_requires_grad_only or meta.requires_grad)
    }
    gathered_shapes = [None] * pp_size
    if local_shapes:
        shape_device = next(data.device for data, meta in zip(datas, data_metas)
                            if meta.name in local_shapes)
        gathered_shapes = _gather_tensor_shapes(local_shapes, pp_group, shape_device)

    for data_idx, data in enumerate(datas):
        meta = data_metas[data_idx]
        if comm_requires_grad_only:
            if not meta.requires_grad:
                for i in range(pp_size):
                    gathered_datas[i][meta.name] = None
                continue
        if meta.is_none:
            assert data is None, f"data name: {meta.name}, data is not None"
            for i in range(pp_size):
                gathered_datas[i][meta.name] = None
            continue
        if meta.is_object:
            output_list = [None] * pp_size
            output_list[pp_rank] = data
            dist.all_gather_object(output_list, data, group=pp_group)
            for i in range(pp_size):
                gathered_datas[i][meta.name] = output_list[i]
            continue
        # Tensor entries stay asynchronous so callers can overlap PP comm with
        # later work and only wait right before the data is consumed.
        shapes = None
        if _dynamic(meta):
            shapes = [rank_shapes[meta.name] for rank_shapes in gathered_shapes]
            send_data = _pad_tensor(data, _padded_shape(shapes))
        else:
            send_data = data.contiguous()
        output_tensor_list = [torch.empty_like(send_data) for _ in range(pp_size)]
        handle = dist.all_gather(
            output_tensor_list,
            send_data,
            group=pp_group,
            async_op=True,
        )
        for i, output in enumerate(output_tensor_list):
            if shapes is not None:
                output = _unpad_tensor(output, shapes[i])
            gathered_datas[i][meta.name] = output.detach().requires_grad_(meta.requires_grad)
        if handle is not None:
            handles.append(_KeepAliveWork(handle, send_data, output_tensor_list))
    return gathered_datas, handles


def scatter_along_pp(datas, pp_group, src_pp_rank, data_metas, comm_requires_grad_only=False,
                     tensor_shapes=None):
    # scatter along pp,
    # if comm_requires_grad_only is set, only scatter the data with requires_grad is True, others will be set as None.
    pp_size = dist.get_world_size(pp_group)
    pp_rank = dist.get_rank(pp_group)
    assert 0 <= src_pp_rank < pp_size, (
        f"src_pp_rank must be in [0, {pp_size}), but got {src_pp_rank}"
    )
    scattered_datas = {}
    scatter_handles = []
    src_global_rank = dist.get_global_rank(pp_group, src_pp_rank)
    # Backward can reuse the shapes learned by every rank in this wave's
    # forward all-gather. Other callers exchange all missing shapes once.
    shapes_by_name = dict(tensor_shapes or {})
    missing = [
        (idx, meta) for idx, meta in enumerate(data_metas)
        if _dynamic(meta) and not meta.is_none and not meta.is_object
        and (not comm_requires_grad_only or meta.requires_grad)
        and meta.name not in shapes_by_name
    ]
    if missing:
        shape_payload = [{
            meta.name: [tuple(t.shape) for t in datas[idx]] for idx, meta in missing
        } if pp_rank == src_pp_rank else None]
        dist.broadcast_object_list(shape_payload, src=src_global_rank, group=pp_group)
        shapes_by_name.update(shape_payload[0])
    for data_idx, data in enumerate(datas):
        assert isinstance(data, list) and len(data) == pp_size, (
            f"data must be a list of length {pp_size}, but got {data}"
        )
        meta = data_metas[data_idx]
        if comm_requires_grad_only:
            if not meta.requires_grad:
                scattered_datas[meta.name] = None
                continue
        
        if meta.is_none:
            scattered_datas[meta.name] = None
            continue
        elif meta.is_object:
            output_list = [None]
            if pp_rank == src_pp_rank:
                objects = data
            else:
                objects = [None] * pp_size
            dist.scatter_object_list(
                output_list,
                objects,
                group=pp_group,
                src=src_global_rank,
            )
            scattered_datas[meta.name] = output_list[0]
        else:
            if pp_rank == src_pp_rank:
                scatter_list = data
            else:
                scatter_list = None
            # The receiver preallocates the exact meta-defined shape because PP
            # scatter must reconstruct one rank-local slice of the source wave.
            shape = meta.shape
            local_shape = None
            if _dynamic(meta):
                shapes = shapes_by_name[meta.name]
                local_shape = shapes[pp_rank]
                shape = _padded_shape(shapes)
                if scatter_list is not None:
                    if [tuple(t.shape) for t in scatter_list] != list(shapes):
                        raise ValueError(f"BigMac backward shape changed for {meta.name}")
                    scatter_list = [_pad_tensor(t, shape) for t in scatter_list]
            scattered_data = torch.empty(
                shape,
                device=torch.cuda.current_device() if torch.cuda.is_available() else "cpu",
                dtype=meta.dtype,
            )
            handle = dist.scatter(
                scattered_data,
                scatter_list,
                src=src_global_rank,
                group=pp_group,
                async_op=True,
            )
            if local_shape is not None:
                scattered_data = _unpad_tensor(scattered_data, local_shape)
            scattered_data = scattered_data.detach().requires_grad_(meta.requires_grad)
            scattered_datas[meta.name] = scattered_data
            if handle is not None:
                scatter_handles.append(_KeepAliveWork(handle, scatter_list, scattered_data))
    return scattered_datas, scatter_handles


def _grad_buckets(grads, bucket_bytes):
    bucket, size, key = [], 0, None
    for grad in grads:
        if grad.numel() == 0:
            continue
        grad_key = (grad.device, grad.dtype)
        nbytes = grad.numel() * grad.element_size()
        if bucket and (grad_key != key or size + nbytes > bucket_bytes):
            yield bucket
            bucket, size = [], 0
        bucket.append(grad)
        size += nbytes
        key = grad_key
    if bucket:
        yield bucket


def _flat_bucket(grads):
    # Use a view only when every element in the span belongs to this bucket.
    #
    # The caller supplies reverse parameter order, matching MCore's grad buffer.
    # Bucket membership/order must never depend on rank-local storage offsets:
    # another PP rank can pack the same bucket if its buffer has padding/gaps.
    first = grads[0]
    storage = first.untyped_storage().data_ptr()
    offset = first.storage_offset()
    for grad in grads:
        if (not grad.is_contiguous()
                or grad.untyped_storage().data_ptr() != storage
                or grad.storage_offset() != offset):
            return torch.cat([g.reshape(-1) for g in grads]), True
        offset += grad.numel()
    return first.as_strided((offset - first.storage_offset(),), (1,)), False


def all_reduce_grad_buckets(grads, group, bucket_bytes=64 * 1024 * 1024):
    # Sum, never average, and finish before native DP reduction/optimizer step.
    #
    # At most two buckets are outstanding. Contiguous DDP buffers need no packing
    # allocation; other layouts use bounded buckets (one oversized parameter is
    # kept whole). Work handles own their buffers until communication completes.
    if bucket_bytes <= 0:
        raise ValueError("BigMac gradient bucket size must be positive")
    pending = deque()

    def finish_one():
        work, flat, bucket, copy_back = pending.popleft()
        if work is not None:
            work.wait()
        if copy_back:
            offset = 0
            for grad in bucket:
                grad.copy_(flat.narrow(0, offset, grad.numel()).view_as(grad))
                offset += grad.numel()

    try:
        for bucket in _grad_buckets(grads, bucket_bytes):
            flat, copy_back = _flat_bucket(bucket)
            work = dist.all_reduce(flat, group=group, async_op=True)
            pending.append((work, flat, bucket, copy_back))
            if len(pending) == 2:
                finish_one()
    finally:
        while pending:
            finish_one()

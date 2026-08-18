# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Bytedance Inc. All rights reserved.
import os
from textwrap import wrap
import types
from functools import wraps
import torch
import torch.distributed

TRANSPOSE_BF16_BLOCK_SIZE = 4096 * 4096
TRANSPOSE_BF16_BIT = 32768


def _copy_model_params_to_main_params():
    pass


def load_parameter_state_from_dp_zero(*args, **kwargs):
    self = args[0]
    state_dict = args[1]

    update_legacy_format = kwargs['update_legacy_format']
    self.load_parameter_state_from_dp_zero_func(state_dict, update_legacy_format=update_legacy_format)
    self.first_sub_flag = False
    data_parallel_world_size = self.data_parallel_group_gloo.size()
    data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
    data_parallel_group_gloo = self.data_parallel_group_gloo
    data_parallel_global_ranks = torch.distributed.get_process_group_ranks(self.data_parallel_group_gloo)
    if data_parallel_world_size == 1 or \
            not hasattr(self, "shard_main_param_res_buffers"):
        return
    for i, shard_main_param_res_buffer in enumerate(self.shard_main_param_res_buffers):
        shard_res_numel = shard_main_param_res_buffer.numel()
        recv_tensor = torch.empty((shard_res_numel,), dtype=torch.float16, device="cpu")
        if data_parallel_rank == 0:
            send_tensors = [
                state_dict["shard_main_param_res"][i][
                dpr * shard_res_numel: (dpr + 1) * shard_res_numel] for dpr in range(data_parallel_world_size)
            ]
        else:
            send_tensors = None

        torch.distributed.scatter(
            recv_tensor,
            send_tensors,
            data_parallel_global_ranks[0],
            data_parallel_group_gloo,
        )
        recv_tensor_bf16_view = torch.tensor(recv_tensor.data.untyped_storage(), dtype=torch.bfloat16,
                                             device=recv_tensor.device)
        shard_main_param_res_buffer.copy_(recv_tensor_bf16_view)


def get_parameter_state_dp_zero(self):
    state = self.get_parameter_state_dp_zero_func()
    data_parallel_world_size = torch.distributed.get_world_size(self.data_parallel_group)
    data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
    data_parallel_group_gloo = self.data_parallel_group_gloo
    data_parallel_global_ranks = torch.distributed.get_process_group_ranks(self.data_parallel_group_gloo)
    if data_parallel_world_size == 1 or not hasattr(self, "shard_main_param_res_buffers"):
        return state

    # gather buffer res
    buffer_res_full_shard = []
    for shard_main_param_res_buffer in self.shard_main_param_res_buffers:
        if data_parallel_rank == 0:
            recv_tensors = [torch.empty((shard_main_param_res_buffer.numel(),), dtype=torch.float16, device="cpu") for _
                            in range(data_parallel_world_size)]
        else:
            recv_tensors = None

        send_tensor = torch.empty((shard_main_param_res_buffer.numel(),), dtype=torch.float16, device="cpu")
        send_tensor_bf16_view = torch.tensor(send_tensor.data.untyped_storage(), dtype=torch.bfloat16,
                                             device=send_tensor.device)
        send_tensor_bf16_view.copy_(shard_main_param_res_buffer.detach().cpu())
        torch.distributed.gather(
            send_tensor,
            recv_tensors,
            data_parallel_global_ranks[0],
            data_parallel_group_gloo,
        )
        if data_parallel_rank == 0:
            buffer_res_full_shard.append(torch.cat(recv_tensors))

    state['shard_main_param_res'] = buffer_res_full_shard
    return state


def fp16_tensor_convert_to_fp32_tensor_dis(self):
    """
    res(0000) + bf16(pppp) -> fp32(0p0p0p0p)

    Transform the bf16 data and residuals data in the continuous memory block
    into the fp32 tensor through view transposition.
    """
    data_parallel_world_size = torch.distributed.get_world_size(self.data_parallel_group)
    data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
    if data_parallel_world_size == 1:
        for shard_fp32_param_fp16_view in self.shard_fp32_param_fp16_view_group:
            shard_fp32_param_fp16_view.copy_(
                shard_fp32_param_fp16_view.view(2, -1).transpose(1, 0).reshape(-1).contiguous())
        for shard_res_and_buffer_model_param in self.shard_main_param_res_buffers:
            shard_main_param_int32_view_buffer = self.model_param_bucket_and_shard_main_param_int32_view_map[
                shard_res_and_buffer_model_param]
            if not self.first_sub_flag:
                shard_main_param_int32_view_buffer.sub_(TRANSPOSE_BF16_BIT)
    else:
        for buffer in self.buffers:
            for bucket in buffer.buckets:
                bucket_param_data = bucket.param_data
                param_data_dp_numel = bucket_param_data.numel() // data_parallel_world_size
                bucket_res = self.model_param_bucket_and_res_map[bucket_param_data]
                if data_parallel_rank == 0:
                    bucket_param_data[param_data_dp_numel:param_data_dp_numel * 2].copy_(
                        bucket_param_data[:param_data_dp_numel])
                bucket_res_position = max(0, data_parallel_rank - 1) * param_data_dp_numel
                shard_fp32_main_param_view = bucket_param_data[
                                             bucket_res_position: bucket_res_position + param_data_dp_numel * 2]
                shard_main_param_int32_view_bucket = self.model_param_bucket_and_shard_main_param_int32_view_map[
                    bucket_param_data]

                loops = param_data_dp_numel // TRANSPOSE_BF16_BLOCK_SIZE
                remain = param_data_dp_numel % TRANSPOSE_BF16_BLOCK_SIZE
                workspace = torch.zeros(
                    TRANSPOSE_BF16_BLOCK_SIZE * 2, dtype=torch.bfloat16, device=bucket_res.device)
                residual_space = bucket_res
                bf16_space_dp_rank = max(1, data_parallel_rank)
                bf16_space = bucket_param_data[
                             param_data_dp_numel * bf16_space_dp_rank:param_data_dp_numel * (bf16_space_dp_rank + 1)]

                for loop in range(loops):
                    copy_start = loop * TRANSPOSE_BF16_BLOCK_SIZE
                    copy_end = (loop + 1) * TRANSPOSE_BF16_BLOCK_SIZE
                    workspace_convert_view = workspace[:TRANSPOSE_BF16_BLOCK_SIZE * 2]
                    workspace[:TRANSPOSE_BF16_BLOCK_SIZE].copy_(residual_space[copy_start: copy_end])
                    workspace[TRANSPOSE_BF16_BLOCK_SIZE:TRANSPOSE_BF16_BLOCK_SIZE * 2].copy_(
                        bf16_space[copy_start: copy_end])
                    shard_fp32_main_param_view[copy_start * 2: copy_end * 2].copy_(
                        workspace_convert_view.view(2, -1).transpose(1, 0).reshape(-1).contiguous())

                if remain > 0:
                    workspace_convert_view = workspace[:remain * 2]
                    workspace[:remain].copy_(residual_space[-remain:])
                    workspace[remain:remain * 2].copy_(bf16_space[-remain:])
                    shard_fp32_main_param_view[-remain * 2:].copy_(
                        workspace_convert_view.view(2, -1).transpose(1, 0).reshape(-1).contiguous())

                if not self.first_sub_flag:
                    shard_main_param_int32_view_bucket[:param_data_dp_numel].sub_(TRANSPOSE_BF16_BIT)


def fp32_tensor_convert_to_fp16_tensor_dis(self):
    """
    fp32(0p0p0p0p) -> fp32(0'p0'p0'p0'p) -> res(0000) + bf16(pppp)

    Transform the fp32 tensor in the continuous memory block
    into the bf16 data and residual through view transposition.
    """
    data_parallel_world_size = torch.distributed.get_world_size(self.data_parallel_group)
    data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)

    if data_parallel_world_size == 1:
        for shard_res_and_buffer_model_param in self.shard_main_param_res_buffers:
            shard_main_param_int32_view_buffer = self.model_param_bucket_and_shard_main_param_int32_view_map[
                shard_res_and_buffer_model_param]
            shard_main_param_int32_view_buffer.add_(TRANSPOSE_BF16_BIT)
        self.first_sub_flag = False

        for shard_fp32_param_fp16_view in self.shard_fp32_param_fp16_view_group:
            shard_fp32_param_fp16_view.copy_(
                shard_fp32_param_fp16_view.view(-1, 2).transpose(1, 0).reshape(-1).contiguous())
    else:
        for buffer in self.buffers:
            for bucket in buffer.buckets:
                bucket_param_data = bucket.param_data
                param_data_dp_numel = bucket_param_data.numel() // data_parallel_world_size
                shard_main_param_int32_view_bucket = self.model_param_bucket_and_shard_main_param_int32_view_map[
                    bucket_param_data]
                shard_main_param_int32_view_bucket[:param_data_dp_numel].add_(TRANSPOSE_BF16_BIT)

        for buffer in self.buffers:
            for bucket in buffer.buckets:
                self.first_sub_flag = False
                bucket_param_data = bucket.param_data
                param_data_dp_numel = bucket_param_data.numel() // data_parallel_world_size
                bucket_res = self.model_param_bucket_and_res_map[bucket_param_data]
                bucket_res_position = max(0, data_parallel_rank - 1) * param_data_dp_numel
                shard_fp32_main_param_view = bucket_param_data[
                                             bucket_res_position: bucket_res_position + param_data_dp_numel * 2]

                loops = param_data_dp_numel // TRANSPOSE_BF16_BLOCK_SIZE
                remain = param_data_dp_numel % TRANSPOSE_BF16_BLOCK_SIZE
                workspace = torch.zeros(
                    TRANSPOSE_BF16_BLOCK_SIZE * 2, dtype=torch.bfloat16, device=bucket_res.device)
                bf16_space_dp_rank = max(0, data_parallel_rank - 1)
                residual_space = bucket_res
                bf16_space = bucket_param_data[
                             param_data_dp_numel * bf16_space_dp_rank:param_data_dp_numel * (bf16_space_dp_rank + 1)]

                for loop in range(loops):
                    workspace_convert_view = workspace[:TRANSPOSE_BF16_BLOCK_SIZE * 2]
                    workspace_convert_view.copy_(
                        shard_fp32_main_param_view[
                        loop * TRANSPOSE_BF16_BLOCK_SIZE * 2: (loop + 1) * TRANSPOSE_BF16_BLOCK_SIZE * 2])
                    temp = workspace_convert_view.view(-1, 2).transpose(1, 0).reshape(-1).contiguous()
                    residual_space[loop * TRANSPOSE_BF16_BLOCK_SIZE: (loop + 1) * TRANSPOSE_BF16_BLOCK_SIZE].copy_(
                        temp[:TRANSPOSE_BF16_BLOCK_SIZE])
                    bf16_space[loop * TRANSPOSE_BF16_BLOCK_SIZE: (loop + 1) * TRANSPOSE_BF16_BLOCK_SIZE].copy_(
                        temp[TRANSPOSE_BF16_BLOCK_SIZE: TRANSPOSE_BF16_BLOCK_SIZE * 2])

                if remain > 0:
                    workspace_convert_view = workspace[:remain * 2]
                    workspace_convert_view.copy_(shard_fp32_main_param_view[-remain * 2:])
                    temp = workspace_convert_view.view(-1, 2).transpose(1, 0).reshape(-1).contiguous()
                    residual_space[-remain:].copy_(temp[:remain])
                    bf16_space[-remain:].copy_(temp[remain: remain * 2])

                if data_parallel_rank != 0:
                    shard_fp32_main_param_view[param_data_dp_numel:param_data_dp_numel * 2].copy_(
                        shard_fp32_main_param_view[:param_data_dp_numel])

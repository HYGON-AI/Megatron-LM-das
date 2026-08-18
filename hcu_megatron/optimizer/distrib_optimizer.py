# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import torch
import torch.distributed


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
            recv_tensors = [torch.empty((shard_main_param_res_buffer.numel(),), dtype=torch.float16, device="cpu")
                                for _ in range(data_parallel_world_size)]
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

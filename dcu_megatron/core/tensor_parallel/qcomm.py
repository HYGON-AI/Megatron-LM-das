import torch
import triton
import triton.language as tl
import random
import unittest
import json
import os
import time


@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv_init_asym(
    K, Out, Out_scale_zero,
    stride_k_bs, stride_k_h, stride_k_d,
    stride_o_bs, stride_o_h, stride_o_d,
    stride_os_bs, stride_os_h, stride_os_d,
    head_num,head_dim,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_HEAD: tl.constexpr
):
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    dest_index = cur_index
    m1 = offs_h[:, None] < head_num
    m2 = offs_d[None,:] < head_dim
    mask = m1&m2
    src_data = tl.load(K + cur_index * stride_k_bs + offs_h[:, None] * stride_k_h + stride_k_d * offs_d[None, :],
                       mask=mask, other=0.0).to(tl.float32)

    src_data_max = tl.max(src_data, axis=1, keep_dims=True)
    src_data_min = tl.min(src_data, axis=1, keep_dims=True)
    data_scale = (src_data_max - src_data_min) / 255.0
    data_zero = (-1 * src_data_min / data_scale).to(tl.int32)
    q_src_data = (tl.clamp((src_data / data_scale).to(tl.int32).to(tl.float32) + data_zero.to(tl.float32), 0.0, 255.0).to(tl.int32) - 128).to(tl.int8)

    data_scale = data_scale.to(Out_scale_zero.dtype.element_ty)
    data_zero = data_zero.to(Out_scale_zero.dtype.element_ty)

    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]
    os_ptrs = Out_scale_zero + dest_index * stride_os_bs + stride_os_h * offs_h[:, None]
    oz_ptrs = Out_scale_zero + dest_index * stride_os_bs + stride_os_h * offs_h[:, None] + 1

    tl.store(o_ptrs, q_src_data, mask=mask)
    tl.store(os_ptrs, data_scale, mask=m1)
    tl.store(oz_ptrs, data_zero, mask=m1)


@torch.no_grad()
def destindex_copy_quantize_kv_init_asym(K, Out, Out_scale_zero):
    bs_seq = K.shape[0]
    head_num = K.shape[1]
    head_dim = K.shape[2]
    assert K.shape[1] == Out.shape[1] and K.shape[2] == Out.shape[2]
    BLOCK_HEAD = triton.next_power_of_2(head_num)
    BLOCK_DMODEL = triton.next_power_of_2(head_dim)
    grid = (bs_seq,)
    num_warps = 1

    _fwd_kernel_destindex_copy_quantize_kv_init_asym[grid](
        K, Out, Out_scale_zero,
        K.stride(0), K.stride(1), K.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        Out_scale_zero.stride(0), Out_scale_zero.stride(1), Out_scale_zero.stride(2),
        head_num,head_dim,
        BLOCK_DMODEL= BLOCK_DMODEL,
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=num_warps,
        num_stages=1,
    )
    return


@triton.jit
def _bwd_kernel_destindex_dequantize_kv(
    Quantized_Out, Out_scale_zero, Dequantized_Out,
    stride_qo_bs, stride_qo_h, stride_qo_d,
    stride_os_bs, stride_os_h, stride_os_d,
    stride_do_bs, stride_do_h, stride_do_d,
    head_num,head_dim,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_HEAD: tl.constexpr
):
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    scales_dtype = Out_scale_zero.dtype.element_ty

    dest_index = cur_index

    m1 = offs_h[:, None] < head_num
    m2 = offs_d[None,:] < head_dim
    mask = m1&m2


    # Load quantized data
    q_data = tl.load(
        Quantized_Out + dest_index * stride_qo_bs + offs_h[:, None] * stride_qo_h + stride_qo_d * offs_d[None, :],
        mask=mask,
        other=0
    )

    # Load scale and zero point
    data_scale = tl.load(
        Out_scale_zero + dest_index * stride_os_bs + stride_os_h * offs_h[:, None],
        mask=m1,
        other=1.0
    )
    data_zero = tl.load(
        Out_scale_zero + dest_index * stride_os_bs + stride_os_h * offs_h[:, None] + 1,
        mask=m1,
        other=0
    )

    # Dequantize
    dequantized_data = (q_data.to(tl.int32) + 128 - data_zero.to(tl.int32)).to(scales_dtype) * data_scale

    # Store dequantized data
    out_ptrs = Dequantized_Out + dest_index * stride_do_bs + stride_do_h * offs_h[:, None] + stride_do_d * offs_d[None, :]
    tl.store(out_ptrs, dequantized_data, mask=mask)


@torch.no_grad()
def destindex_dequantize_kv(Quantized_Out, Out_scale_zero, Dequantized_Out):
    bs_seq = Quantized_Out.shape[0]
    head_num = Quantized_Out.shape[1]
    head_dim = Quantized_Out.shape[2]
    assert Quantized_Out.shape[1] == Dequantized_Out.shape[1] and Quantized_Out.shape[2] == Dequantized_Out.shape[2]
    BLOCK_HEAD = triton.next_power_of_2(head_num)
    BLOCK_DMODEL = triton.next_power_of_2(head_dim)
    grid = (bs_seq,)
    num_warps = 1

    _bwd_kernel_destindex_dequantize_kv[grid](
        Quantized_Out, Out_scale_zero, Dequantized_Out,
        Quantized_Out.stride(0), Quantized_Out.stride(1), Quantized_Out.stride(2),
        Out_scale_zero.stride(0), Out_scale_zero.stride(1), Out_scale_zero.stride(2),
        Dequantized_Out.stride(0), Dequantized_Out.stride(1), Dequantized_Out.stride(2),
        head_num,head_dim,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=num_warps,
        num_stages=1,
    )

@torch.no_grad()
def fp16_to_int8s(fp16_tensor):
    fp16_bytes = fp16_tensor.contiguous().view(torch.int8)
    int8_high = fp16_bytes[::2]  # 高 8 位
    int8_low = fp16_bytes[1::2]  # 低 8 位
    return int8_high.unsqueeze(1), int8_low.unsqueeze(1)



@torch.no_grad()
def int8s_to_fp16(int8_high, int8_low):
    fp16_bytes = torch.stack([int8_high, int8_low], dim=-1).view(torch.int16)
    return fp16_bytes.view(torch.bfloat16)

def _alltoall(group, input, output_split_sizes, input_split_sizes):
    input = input.contiguous()
    if output_split_sizes is None:
        # Equal split (all2all)
        output = torch.empty_like(input)
    else:
        # Unequal split (all2all-v)
        output = input.new_empty(
            size=[sum(output_split_sizes)] + list(input.size()[1:]),
            dtype=input.dtype,
            device=torch.cuda.current_device(),
        )

    torch.distributed.all_to_all_single(
        output,
        input,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
    return output

def q_alltoall(output, input, output_split_sizes, input_split_sizes,group):
    t,s = input.shape[0],input.shape[1]
    input_buffer_int8 = torch.empty((t, 1, s), dtype=torch.int8, device="cuda")
    buffer_scales = torch.empty((t, 1, 2), dtype=torch.bfloat16, device="cuda")
    input_q = input.unsqueeze(1)

    destindex_copy_quantize_kv_init_asym(
            input_q,
            input_buffer_int8,
            buffer_scales,
            )

    input_buffer_int8 = input_buffer_int8.squeeze()
    buffer_scales = buffer_scales.squeeze()
    buffer_scales_h, buffer_scales_l = fp16_to_int8s(buffer_scales[:,0])
    buffer_shift_h, buffer_shift_l = fp16_to_int8s(buffer_scales[:,1])

    input_all = torch.cat([input_buffer_int8, buffer_scales_h, buffer_scales_l,buffer_shift_h, buffer_shift_l], dim=1)

    torch.distributed.all_to_all_single(
        output,
        input_all,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )


    scale = int8s_to_fp16(output[:,-4], output[:,-3])
    shift = int8s_to_fp16(output[:,-2], output[:,-1])
    scales = torch.cat([scale,shift],dim=1).unsqueeze(1)
    deq_out = torch.empty((output.shape[0], 1, output.shape[1]-4), dtype=torch.bfloat16, device="cuda")
    destindex_dequantize_kv(output[:,:-4].unsqueeze(1), scales, deq_out)

    return deq_out.squeeze()

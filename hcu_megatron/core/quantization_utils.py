# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl


@torch.no_grad()
def float_to_int8s(float_tensor):
    bytes_tensor = float_tensor.contiguous().view(torch.int8)
    return bytes_tensor


@torch.no_grad()
def int8s_to_float(scale_and_shift_int8, output_dtype=torch.bfloat16):
    if (
        output_dtype == torch.bfloat16
        or output_dtype == torch.float16
    ):
        assert scale_and_shift_int8.shape[-1] % 2 == 0
        float_tensor = scale_and_shift_int8.view(torch.int16)
    elif output_dtype == torch.float32:
        assert scale_and_shift_int8.shape[-1] % 4 == 0
        float_tensor = scale_and_shift_int8.view(torch.int32)
    else:
        raise ValueError(f"{output_dtype} is not supported")

    return float_tensor.view(output_dtype)


@triton.jit
def _fwd_kernel_destindex_copy_quantize_init_asym(
    K, Out, Out_scale_zero,
    stride_k_bs, stride_k_h, stride_k_d,
    stride_o_bs, stride_o_h, stride_o_d,
    stride_os_bs, stride_os_h, stride_os_d,
    head_num, head_dim,
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
    data_scale = tl.where(data_scale < 1e-8, 1e-8, data_scale)
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
def destindex_copy_quantize_int8(K, Out, Out_scale_zero):
    bs_seq = K.shape[0]
    head_num = K.shape[1]
    head_dim = K.shape[2]
    assert K.shape[1] == Out.shape[1] and K.shape[2] == Out.shape[2]
    BLOCK_HEAD = triton.next_power_of_2(head_num)
    BLOCK_DMODEL = triton.next_power_of_2(head_dim)
    grid = (bs_seq,)
    num_warps = 1

    _fwd_kernel_destindex_copy_quantize_init_asym[grid](
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
def _bwd_kernel_destindex_dequantize(
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
def destindex_dequantize_int8(Quantized_Out, Out_scale_zero, Dequantized_Out):
    bs_seq = Quantized_Out.shape[0]
    head_num = Quantized_Out.shape[1]
    head_dim = Quantized_Out.shape[2]
    assert Quantized_Out.shape[1] == Dequantized_Out.shape[1] and Quantized_Out.shape[2] == Dequantized_Out.shape[2]
    BLOCK_HEAD = triton.next_power_of_2(head_num)
    BLOCK_DMODEL = triton.next_power_of_2(head_dim)
    grid = (bs_seq,)
    num_warps = 1

    _bwd_kernel_destindex_dequantize[grid](
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


@triton.jit
def _fwd_kernel_destindex_copy_quantize_int4_init(
    K,
    Out,
    Out_scale_zero,
    stride_k_bs,
    stride_k_h,
    stride_k_g,
    stride_k_d,
    stride_o_bs,
    stride_o_h,
    stride_o_g,
    stride_o_d,
    stride_os_bs,
    stride_os_h,
    stride_os_g,
    group_size,
    BLOCK_GROUP_NUM: tl.constexpr,
    BLOCK_GROUP_DIM: tl.constexpr,
):
    cur_index = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_g = tl.arange(0, BLOCK_GROUP_NUM)
    offs_d = tl.arange(0, BLOCK_GROUP_DIM // 2)

    dest_index = cur_index

    offs_kv = cur_index * stride_k_bs + cur_head * stride_k_h + offs_g[:, None] * stride_k_g + offs_d[None, :] * 2

    src_data_0 = tl.load(
        K + offs_kv,
        mask=offs_g[:, None] < group_size,
        other=0.0,
    ).to(tl.float32)
    src_data_1 = tl.load(
        K + offs_kv + 1,
        mask=offs_g[:, None] < group_size,
        other=0.0,
    ).to(tl.float32)

    # 计算量化因子并量化数据
    min_data = tl.minimum(tl.min(src_data_0, axis=1), tl.min(src_data_1, axis=1))
    max_data = tl.maximum(tl.max(src_data_0, axis=1), tl.max(src_data_1, axis=1))

    scale = (max_data - min_data) / 15.0
    scale = tl.where(scale < 1e-8, 1e-8, scale)
    zero_point = -1 * min_data / scale

    # 非对称量化
    q_src_data_0 = ((src_data_0 / scale[:, None]) + zero_point[:, None]).to(tl.int8)
    q_src_data_0 = tl.clamp(q_src_data_0.to(tl.float32), 0.0, 15.0).to(tl.int8)

    q_src_data_1 = ((src_data_1 / scale[:, None]) + zero_point[:, None]).to(tl.int8)
    q_src_data_1 = tl.clamp(q_src_data_1.to(tl.float32), 0.0, 15.0).to(tl.int8)

    # 合并为 int4 数据
    low_4 = q_src_data_0 & 0xF
    high_4 = (q_src_data_1 & 0xF) << 4
    out_data = low_4 | high_4

    scale = scale.to(Out_scale_zero.dtype.element_ty)
    zero_point = zero_point.to(Out_scale_zero.dtype.element_ty)

    o_ptrs = Out + dest_index * stride_o_bs + cur_head * stride_o_h + offs_g[:, None] * stride_o_g + offs_d[None, :]
    os_ptrs = Out_scale_zero + dest_index * stride_os_bs + cur_head * stride_os_h + offs_g
    oz_ptrs = Out_scale_zero + dest_index * stride_os_bs + cur_head * stride_os_h + offs_g + BLOCK_GROUP_NUM
    tl.store(o_ptrs, out_data, mask=offs_g[:, None] < group_size)
    tl.store(os_ptrs, scale, mask=offs_g < group_size)
    tl.store(oz_ptrs, zero_point, mask=offs_g < group_size)


@torch.no_grad()
def destindex_copy_quantize_int4(K, Out, Out_scale, quant_group_dim):
    bs_seq = K.shape[0]
    head_num = K.shape[1]
    head_dim = K.shape[2]

    assert head_dim % quant_group_dim == 0, "error head dim, can not been supported to copy quant kv"
    grid = (bs_seq, head_num)
    num_warps = 1

    group_size = head_dim // quant_group_dim
    group_dim = quant_group_dim

    K = K.view((K.shape[0], K.shape[1], group_size, group_dim))
    Out = Out.view(
        Out.shape[0], Out.shape[1], group_size, group_dim // 2
    )  # OUt 是 int8 类型， 两个int4组一个int8，所以 group_dim // 2

    _fwd_kernel_destindex_copy_quantize_int4_init[grid](
        K,
        Out,
        Out_scale,
        K.stride(0),
        K.stride(1),
        K.stride(2),
        K.stride(3),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        Out.stride(3),
        Out_scale.stride(0),
        Out_scale.stride(1),
        Out_scale.stride(2),
        group_size,
        BLOCK_GROUP_NUM=triton.next_power_of_2(group_size),
        BLOCK_GROUP_DIM=group_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return


@triton.jit
def _bwd_kernel_destindex_dequantize_int4(
    Quantized,
    Scale,
    Out,
    stride_q_bs,
    stride_q_h,
    stride_q_g,
    stride_q_d,
    stride_s_bs,
    stride_s_h,
    stride_s_g,
    stride_o_bs,
    stride_o_h,
    stride_o_g,
    stride_o_d,
    group_size,
    BLOCK_GROUP_NUM: tl.constexpr,
    BLOCK_GROUP_DIM: tl.constexpr,
):
    cur_index = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_g = tl.arange(0, BLOCK_GROUP_NUM)
    offs_d = tl.arange(0, BLOCK_GROUP_DIM // 2)

    dest_index = cur_index
    scales_dtype = Scale.dtype.element_ty

    # 加载量化数据
    q_data = tl.load(
        Quantized + cur_index * stride_q_bs + cur_head * stride_q_h + offs_g[:, None] * stride_q_g + offs_d[None, :],
        mask=offs_g[:, None] < group_size,
        other=0.0,
    )

    # 分离 int8 的低 4 位（int4 数据 0）和高 4 位（int4 数据 1）
    low_4 = q_data & 0xF
    high_4 = (q_data >> 4) & 0xF

    src_data_0 = low_4.to(tl.int8)
    src_data_1 = high_4.to(tl.int8)

    # 加载反量化比例因子（scale）
    scale = tl.load(Scale + dest_index * stride_s_bs + cur_head * stride_s_h + offs_g, mask=offs_g < group_size)
    zero = tl.load(Scale + dest_index * stride_s_bs + cur_head * stride_s_h + offs_g + BLOCK_GROUP_NUM, mask=offs_g < group_size)

    # 反量化
    dequant_data_0 = (src_data_0.to(tl.float32) - zero.to(tl.float32)[:, None]).to(scales_dtype) * scale[:, None]
    dequant_data_1 = (src_data_1.to(tl.float32) - zero.to(tl.float32)[:, None]).to(scales_dtype) * scale[:, None]

    # 存储反量化的 float 数据
    o_ptrs_0 = Out + dest_index * stride_o_bs + cur_head * stride_o_h + offs_g[:, None] * stride_o_g + offs_d[None, :] * 2
    o_ptrs_1 = Out + dest_index * stride_o_bs + cur_head * stride_o_h + offs_g[:, None] * stride_o_g + offs_d[None, :] * 2 + 1

    tl.store(o_ptrs_0, dequant_data_0, mask=offs_g[:, None] < group_size)
    tl.store(o_ptrs_1, dequant_data_1, mask=offs_g[:, None] < group_size)


@torch.no_grad()
def destindex_dequantize_int4(Quantized, Scale, Out, quant_group_dim):
    bs = Out.shape[0]
    head_num = Quantized.shape[1]
    head_dim = Out.shape[2]

    assert head_dim % quant_group_dim == 0, "error head dim, can not been supported to copy dequant kv"
    grid = (bs, head_num)
    num_warps = 1

    group_size = head_dim // quant_group_dim
    group_dim = quant_group_dim

    Quantized = Quantized.view((Quantized.shape[0], Quantized.shape[1], group_size, group_dim // 2))
    Scale = Scale.view((Scale.shape[0], Scale.shape[1], group_size * 2))
    Out = Out.view(
        Out.shape[0], Out.shape[1], group_size, group_dim
    )  # Out 是 float16 类型，解压缩时需要两个 int4 恢复成 float16，所以 group_dim

    _bwd_kernel_destindex_dequantize_int4[grid](
        Quantized,
        Scale,
        Out,
        Quantized.stride(0),
        Quantized.stride(1),
        Quantized.stride(2),
        Quantized.stride(3),
        Scale.stride(0),
        Scale.stride(1),
        Scale.stride(2),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        Out.stride(3),
        group_size,
        BLOCK_GROUP_NUM=triton.next_power_of_2(group_size),
        BLOCK_GROUP_DIM=group_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return

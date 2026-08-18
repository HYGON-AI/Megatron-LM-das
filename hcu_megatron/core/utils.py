# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import torch

from hcu_megatron.training.arguments import get_adaptor_args


def get_batch_on_this_tp_rank(
    batch: dict[str, torch.Tensor],
    is_sft: bool,
    is_hybrid_cp: bool,
    create_attention_mask_in_dataloader: bool,
    broadcast_src_rank: int,
    broadcast_group: torch.distributed.ProcessGroup,
    cp_size: int,
    tp_rank: int,
    micro_batch_size: int,
    seq_length: int,
    mtp_on_this_rank: bool,
    pipeline_model_parallel_size: int = 1,
    is_pipeline_first_stage: bool = False,
    is_pipeline_last_stage: bool = False,
):
    """Broadcast batch tensors from TP rank 0 to all other ranks in the TP group.

    TP rank 0 holds the fully preprocessed batch (from the dataloader or from
    ``preprocess_sft_batch`` when SFT is enabled). This function broadcasts
    every required tensor to the remaining TP ranks so that all ranks hold
    identical data before the forward pass. The set of tensors broadcast depends
    on the pipeline stage and whether SFT / hybrid-CP modes are active.

    For SFT and hybrid-CP, variable-length metadata (``cu_seqlens``,
    ``cu_seqlens_padded``) is broadcast using a length-prefixed protocol: TP
    rank 0 first sends the numel, then the tensor itself, so receivers can
    allocate the correct buffer size.

    For hybrid-CP, the sequence length may differ per micro-batch (since it
    depends on `local_cp_size`), so the actual sequence length is broadcast
    before allocating receive buffers on non-zero TP ranks.

    Args:
        batch (dict[str, torch.Tensor]): The batch dict. On TP rank 0 this
            contains the actual data; on other ranks it is ignored (receive
            buffers are allocated internally).
        is_sft (bool): Whether this is an SFT (supervised fine-tuning) run
            using THD packed sequences.
        is_hybrid_cp (bool): Whether hybrid context parallelism is enabled.
        create_attention_mask_in_dataloader (bool): Whether the dataloader
            creates an explicit attention mask tensor.
        broadcast_src_rank (int): Global rank of the broadcast source (TP rank 0).
        broadcast_group (torch.distributed.ProcessGroup): The TP process group
            used for broadcasting.
        cp_size (int): Context-parallel world size.
        tp_rank (int): This rank's position within the TP group.
        micro_batch_size (int): Micro-batch size (number of samples).
        seq_length (int): Sequence length used for allocating receive buffers
            (ignored under hybrid-CP where it is broadcast dynamically).
        mtp_on_this_rank (bool): Whether Multi-Token Prediction layers are
            active on this rank (affects which tensors are needed).
        pipeline_model_parallel_size (int): Number of pipeline-parallel stages.
        is_pipeline_first_stage (bool): Whether this rank is on the first PP stage.
        is_pipeline_last_stage (bool): Whether this rank is on the last PP stage.

    Returns:
        dict[str, torch.Tensor]: The batch dict with all tensors populated on
        every TP rank. Keys include 'tokens', 'labels', 'loss_mask',
        'position_ids', 'attention_mask', 'cu_seqlens', 'cu_seqlens_padded',
        'max_seqlen', 'local_cp_size', and 'hybrid_cp_group'.
    """

    args = get_adaptor_args()

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(item, broadcast_src_rank, group=broadcast_group)

    if tp_rank == 0:

        def _broadcast_cu_seqlens(cu_seqlens):
            dev = torch.cuda.current_device()
            n = 0 if cu_seqlens is None else int(cu_seqlens.numel())
            n_tensor = torch.tensor(n, dtype=torch.int64, device=dev)
            _broadcast(n_tensor)

            if n > 0:
                assert isinstance(
                    cu_seqlens, torch.Tensor
                ), f"Expected cu_seqlens to be a torch.Tensor, got {type(cu_seqlens)}"
                assert (
                    cu_seqlens.dtype == torch.int32
                ), f"Expected cu_seqlens to be of type torch.int32, got {cu_seqlens.dtype}"
                _broadcast(cu_seqlens)

        if is_hybrid_cp:
            hybrid_cp_seq_length = torch.tensor(
                batch['tokens'].shape[1], dtype=torch.int32, device=torch.cuda.current_device()
            )
            _broadcast(hybrid_cp_seq_length)

        if args.enable_vocab_parallel or pipeline_model_parallel_size == 1 or mtp_on_this_rank:
            _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['position_ids'])
            if is_sft or is_hybrid_cp:
                _broadcast_cu_seqlens(batch['cu_seqlens'])
                _broadcast(batch['max_seqlen'])
                if cp_size > 1:
                    _broadcast_cu_seqlens(batch['cu_seqlens_padded'])
            if create_attention_mask_in_dataloader:
                _broadcast(batch['attention_mask'])
            if is_hybrid_cp:
                _broadcast(batch['local_cp_size'])

        elif is_pipeline_first_stage:
            _broadcast(batch['tokens'])
            _broadcast(batch['position_ids'])
            if is_sft:
                _broadcast_cu_seqlens(batch['cu_seqlens'])
                _broadcast(batch['max_seqlen'])
                if cp_size > 1:
                    _broadcast_cu_seqlens(batch['cu_seqlens_padded'])
            if create_attention_mask_in_dataloader:
                _broadcast(batch['attention_mask'])

            if args.schedule_method == "dualpipev":
                _broadcast(batch['loss_mask'])
                _broadcast(batch['labels'])
            else:
                batch["labels"] = None
                batch["loss_mask"] = None

        elif is_pipeline_last_stage:
            batch["tokens"] = None
            batch["position_ids"] = None

            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            if is_sft:
                _broadcast_cu_seqlens(batch['cu_seqlens'])
                _broadcast(batch['max_seqlen'])
                if cp_size > 1:
                    _broadcast_cu_seqlens(batch['cu_seqlens_padded'])
            if create_attention_mask_in_dataloader:
                _broadcast(batch['attention_mask'])

        elif is_sft:
            # NOTE(asolergi-nv): Broadcast required THD metadata for SFT to intermediate stages
            batch["tokens"] = None
            batch["labels"] = None
            batch["loss_mask"] = None
            batch["position_ids"] = None
            batch["attention_mask"] = None

            _broadcast_cu_seqlens(batch['cu_seqlens'])
            _broadcast(batch['max_seqlen'])
            if cp_size > 1:
                _broadcast_cu_seqlens(batch['cu_seqlens_padded'])

    else:
        if is_hybrid_cp:
            hybrid_cp_seq_length = torch.tensor(
                0, dtype=torch.int32, device=torch.cuda.current_device()
            )
            _broadcast(hybrid_cp_seq_length)
            shape = (micro_batch_size, hybrid_cp_seq_length.item())
        else:
            shape = (micro_batch_size, seq_length)

        tokens = torch.empty(shape, dtype=torch.int64, device=torch.cuda.current_device())
        labels = torch.empty(shape, dtype=torch.int64, device=torch.cuda.current_device())
        loss_mask = torch.empty(shape, dtype=torch.float32, device=torch.cuda.current_device())
        position_ids = torch.empty(shape, dtype=torch.int64, device=torch.cuda.current_device())
        cu_seqlens = None
        cu_seqlens_padded = None
        max_seqlen = None
        attention_mask = None
        local_cp_size = None

        if is_sft or is_hybrid_cp:
            max_seqlen = torch.empty(1, dtype=torch.int32, device=torch.cuda.current_device())
        if create_attention_mask_in_dataloader:
            attention_mask = torch.empty(
                (micro_batch_size, 1, seq_length, seq_length),
                dtype=torch.bool,
                device=torch.cuda.current_device(),
            )

        if is_hybrid_cp:
            local_cp_size = torch.empty(1, dtype=torch.int32, device=torch.cuda.current_device())

        def _broadcast_cu_seqlens():
            dev = torch.cuda.current_device()

            n = torch.empty((), dtype=torch.int64, device=dev)
            _broadcast(n)
            n = int(n.item())

            if n == 0:
                return None

            # cu_seqlens / cu_seqlens_padded carry the dataloader's batch dim
            # throughout (mbs=1 for packed sequences). Allocate (1, n) so the
            # shape on receiving ranks matches the (1, n) tensor TP rank 0 sent.
            cu_seqlens = torch.empty((1, n), dtype=torch.int32, device=dev)
            _broadcast(cu_seqlens)
            assert (
                cu_seqlens.dim() == 2 and cu_seqlens.shape[0] == 1
            ), f"Expected cu_seqlens shape (1, n), got {tuple(cu_seqlens.shape)}"
            assert (
                cu_seqlens.dtype == torch.int32
            ), f"Expected cu_seqlens to be of type torch.int32, got {cu_seqlens.dtype}"
            return cu_seqlens

        if args.enable_vocab_parallel or pipeline_model_parallel_size == 1 or mtp_on_this_rank:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(position_ids)
            if is_sft or is_hybrid_cp:
                cu_seqlens = _broadcast_cu_seqlens()
                _broadcast(max_seqlen)
                if cp_size > 1:
                    cu_seqlens_padded = _broadcast_cu_seqlens()
            if create_attention_mask_in_dataloader:
                _broadcast(attention_mask)
            if is_hybrid_cp:
                _broadcast(local_cp_size)

        elif is_pipeline_first_stage:
            _broadcast(tokens)
            _broadcast(position_ids)
            if is_sft:
                cu_seqlens = _broadcast_cu_seqlens()
                _broadcast(max_seqlen)
                if cp_size > 1:
                    cu_seqlens_padded = _broadcast_cu_seqlens()
            if create_attention_mask_in_dataloader:
                _broadcast(attention_mask)

            if args.schedule_method == "dualpipev":
                _broadcast(loss_mask)
                _broadcast(labels)
            else:
                labels=None
                loss_mask=None

        elif is_pipeline_last_stage:
            tokens = None
            position_ids = None

            _broadcast(labels)
            _broadcast(loss_mask)
            if is_sft:
                cu_seqlens = _broadcast_cu_seqlens()
                _broadcast(max_seqlen)
                if cp_size > 1:
                    cu_seqlens_padded = _broadcast_cu_seqlens()
            if create_attention_mask_in_dataloader:
                _broadcast(attention_mask)

        elif is_sft:
            # NOTE(asolergi-nv): Broadcast required THD metadata for SFT to intermediate stages
            tokens = None
            labels = None
            loss_mask = None
            position_ids = None

            cu_seqlens = _broadcast_cu_seqlens()
            _broadcast(max_seqlen)
            if cp_size > 1:
                cu_seqlens_padded = _broadcast_cu_seqlens()

        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'cu_seqlens': cu_seqlens,
            'cu_seqlens_padded': cu_seqlens_padded,
            'max_seqlen': max_seqlen,
            'local_cp_size': local_cp_size,
            'hybrid_cp_group': None,
        }

    return batch


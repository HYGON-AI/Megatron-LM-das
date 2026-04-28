from typing import Union

import torch

from megatron.training import get_args
from megatron.core import mpu


def print_rank_message(message, rank_id: Union[int, set, list]):
    """If distributed is initialized, print only on rank specified by rank_id."""
    if isinstance(rank_id, int):
        rank_id = {rank_id}
    rank_id = set(rank_id)
    if torch.distributed.is_initialized():
        current_rank = torch.distributed.get_rank()
        if not rank_id or current_rank in rank_id:
            print(f"[rank {current_rank}] {message}", flush=True)
    else:
        print(message, flush=True)


def get_batch_on_this_tp_rank(data_iterator, mtp_on_this_rank: bool = False):

    args = get_args()

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(
                item,
                mpu.get_tensor_model_parallel_src_rank(),
                group=mpu.get_tensor_model_parallel_group(),
            )

    if mpu.get_tensor_model_parallel_rank() == 0:

        assert data_iterator is not None
        data = next(data_iterator)
        batch = {
            'tokens': data["tokens"].cuda(non_blocking=True),
            'labels': data["labels"].cuda(non_blocking=True),
            'loss_mask': data["loss_mask"].cuda(non_blocking=True),
            'attention_mask': (
                None
                if "attention_mask" not in data
                else data["attention_mask"].cuda(non_blocking=True)
            ),
            'position_ids': data["position_ids"].cuda(non_blocking=True),
            'cu_seqlens': (
                None
                if "cu_seqlens" not in data
                else data["cu_seqlens"].cuda(non_blocking=True)
            ),
            'max_seqlen': (
                None
                if "max_seqlen" not in data
                else data["max_seqlen"].cuda(non_blocking=True)
            ),
            'local_cp_size': (
                None
                if "local_cp_size" not in data
                else data["local_cp_size"].cuda(non_blocking=True)
            ),
        }

        def _broadcast_cu_seqlens(cu_seqlens):
            dev = torch.cuda.current_device()
            n = 0 if cu_seqlens is None else int(cu_seqlens.numel())
            n_tensor = torch.tensor(n, dtype=torch.int64, device=dev)
            _broadcast(n_tensor)

            if n == 0:
                buf = torch.empty(0, dtype=torch.int32, device=dev)
            else:
                assert isinstance(cu_seqlens, torch.Tensor)
                assert cu_seqlens.dtype == torch.int32
                assert cu_seqlens.shape[0] == 1, "micro-batch-size must be 1 for packing"
                buf = cu_seqlens.to(device=dev, non_blocking=True).contiguous()
            _broadcast(buf)

        if args.hybrid_context_parallel:
            seq_len = torch.tensor(batch['tokens'].shape[0], dtype=torch.int32, device=torch.cuda.current_device())
            _broadcast(seq_len)

        if args.enable_vocab_parallel or args.pipeline_model_parallel_size == 1 or mtp_on_this_rank:
            _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])
            _broadcast_cu_seqlens(batch['cu_seqlens'])
            _broadcast(batch['max_seqlen'])
            _broadcast(batch['local_cp_size'])

        elif mpu.is_pipeline_first_stage():
            _broadcast(batch['tokens'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])
            _broadcast_cu_seqlens(batch['cu_seqlens'])
            _broadcast(batch['max_seqlen'])
            if args.schedule_method == "dualpipev":
                _broadcast(batch['loss_mask'])
                _broadcast(batch['labels'])

        elif mpu.is_pipeline_last_stage():
            # Multi-Token Prediction (MTP) layers need tokens and position_ids to calculate embedding.
            # Currently the Multi-Token Prediction (MTP) layers is fixed on the last stage, so we need
            # to broadcast tokens and position_ids to all of the tensor parallel ranks on the last stage.
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])

    else:
        if args.hybrid_context_parallel:
            seq_len = torch.tensor(0, dtype=torch.int32, device=torch.cuda.current_device())
            _broadcast(seq_len)
            shape = (seq_len.item())
        else:
            shape = (args.micro_batch_size, args.seq_length)

        tokens = torch.empty(
            shape,
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        labels = torch.empty(
            shape,
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        loss_mask = torch.empty(
            shape,
            dtype=torch.float32,
            device=torch.cuda.current_device(),
        )
        if args.create_attention_mask_in_dataloader:
            shape_attention_mask = (args.micro_batch_size, 1, args.seq_length, args.seq_length) if not args.hybrid_context_parallel else (1, 1, shape[0], shape[0])
            attention_mask = torch.empty(
                shape_attention_mask,
                dtype=torch.bool,
                device=torch.cuda.current_device(),
            )
        else:
            attention_mask = None
        position_ids = torch.empty(
            shape,
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        cu_seqlens = None
        if args.hybrid_context_parallel or args.sft
            max_seqlen = torch.empty(
                1,
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
        else:
            max_seqlen = None

        local_cp_size = torch.empty(
            1,
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        ) if args.hybrid_context_parallel else None

        def _broadcast_cu_seqlens():
            dev = torch.cuda.current_device()

            n = torch.empty((), dtype=torch.int64, device=dev)
            _broadcast(n)
            n = int(n.item())

            if n == 0:
                cu_seqlens = torch.empty(0, dtype=torch.int32, device=dev)
            else:
                cu_seqlens = torch.empty((args.micro_batch_size, n), dtype=torch.int32, device=dev)
            _broadcast(cu_seqlens)

            return cu_seqlens if n > 0 else None

        if args.enable_vocab_parallel or args.pipeline_model_parallel_size == 1 or mtp_on_this_rank:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(position_ids)
            cu_seqlens = _broadcast_cu_seqlens()
            _broadcast(max_seqlen)
            _broadcast(local_cp_size)

        elif mpu.is_pipeline_first_stage():
            _broadcast(tokens)
            _broadcast(attention_mask)
            _broadcast(position_ids)
            cu_seqlens = _broadcast_cu_seqlens()
            _broadcast(max_seqlen)

            if args.schedule_method == "dualpipev":
                _broadcast(loss_mask)
                _broadcast(labels)
            else:
                labels=None
                loss_mask=None

        elif mpu.is_pipeline_last_stage():
            # Multi-Token Prediction (MTP) layers need tokens and position_ids to calculate embedding.
            # Currently the Multi-Token Prediction (MTP) layers is fixed on the last stage, so we need
            # to broadcast tokens and position_ids to all of the tensor parallel ranks on the last stage.
            tokens = None
            position_ids = None
            cu_seqlens = None
            max_seqlen = None

            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)

        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'cu_seqlens': cu_seqlens,
            'max_seqlen': max_seqlen,
            'local_cp_size': local_cp_size,
        }

    return batch

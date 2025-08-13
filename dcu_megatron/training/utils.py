import torch

from megatron.training import get_args
from megatron.core import mpu


def print_rank_message(message, rank_id=0):
    """If distributed is initialized, print only on rank specified by rank_id."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == rank_id:
            print(f"[rank {rank_id}] {message}", flush=True)
    else:
        print(f"[rank {rank_id}] {message}", flush=True)


def get_batch_on_this_tp_rank(data_iterator):

    args = get_args()

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(
                item,
                mpu.get_tensor_model_parallel_src_rank(),
                group=mpu.get_tensor_model_parallel_group(),
            )

    if mpu.get_tensor_model_parallel_rank() == 0:

        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None

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
        }

        if args.pipeline_model_parallel_size == 1:
            _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])

        elif mpu.is_pipeline_first_stage():
            _broadcast(batch['tokens'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])
            if args.schedule_method == "dualpipev":
                _broadcast(batch['loss_mask'])
                _broadcast(batch['labels'])

        elif mpu.is_pipeline_last_stage():
            # Multi-Token Prediction (MTP) layers need tokens and position_ids to calculate embedding.
            # Currently the Multi-Token Prediction (MTP) layers is fixed on the last stage, so we need
            # to broadcast tokens and position_ids to all of the tensor parallel ranks on the last stage.
            if args.mtp_num_layers is not None:
                _broadcast(batch['tokens'])
                _broadcast(batch['position_ids'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])

    else:

        tokens = torch.empty(
            (args.micro_batch_size, args.seq_length),
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        labels = torch.empty(
            (args.micro_batch_size, args.seq_length),
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        loss_mask = torch.empty(
            (args.micro_batch_size, args.seq_length),
            dtype=torch.float32,
            device=torch.cuda.current_device(),
        )
        if args.create_attention_mask_in_dataloader:
            attention_mask = torch.empty(
                (args.micro_batch_size, 1, args.seq_length, args.seq_length),
                dtype=torch.bool,
                device=torch.cuda.current_device(),
            )
        else:
            attention_mask = None
        position_ids = torch.empty(
            (args.micro_batch_size, args.seq_length),
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )

        if args.pipeline_model_parallel_size == 1:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif mpu.is_pipeline_first_stage():
            _broadcast(tokens)
            _broadcast(attention_mask)
            _broadcast(position_ids)

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
            if args.mtp_num_layers is not None:
                _broadcast(tokens)
                _broadcast(position_ids)
            else:
                tokens = None
                position_ids = None

            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)

        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        }

    return batch

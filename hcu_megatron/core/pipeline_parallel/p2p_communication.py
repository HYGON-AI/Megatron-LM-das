# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from typing import List, Optional, Tuple, Union

import torch

from megatron.core.utils import nvtx_decorator
from megatron.core.pipeline_parallel.p2p_communication import is_single_shape, _batched_p2p_ops, _p2p_ops

from hcu_megatron.core.tensor_parallel.vocab_input_store import VocabInputStore
from hcu_megatron.training.arguments import get_adaptor_args

# Types
Shape = Union[List[int], torch.Size]


class P2PCommunicator:
    """P2P (Point-to-Point) Communicator for pipeline parallelism.

    This class handles communication between pipeline stages by managing
    tensor exchanges between consecutive stages in the pipeline.
    """
    def _communicate(
        self,
        *,
        tensor_send_next: Optional[torch.Tensor],
        tensor_send_prev: Optional[torch.Tensor],
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: Shape,
        wait_on_reqs: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Communicate tensors between stages. Used as helper method in other
        communication methods that are used in megatron/schedules.py.

        Args:
            tensor_send_next (torch.Tensor, optional):
                Tensor to send to next rank (no tensor sent if None)

            tensor_send_prev (torch.Tensor, optional):
                Tensor to send to prev rank (no tensor sent if None)

            recv_prev (boolean, required):
                whether tensor should be received from previous rank.

            recv_next (boolean, required):
                whether tensor should be received from next rank.

            tensor_shape (List[int] or torch.Size, required):
                shape of tensor to receive (this method assumes that all
                tensors sent and received in a single function call are
                the same shape).

            wait_on_reqs (boolean, optional, default=False):
                For non-batched p2p communication, wait on each request
                before returning.

        Returns:
            tuple containing

            - tensor_recv_prev: torch.Tensor if recv_prev is True, None otherwise.
            - tensor_recv_next: torch.Tensor if recv_next is True, None otherwise.

        """

        config = self.config
        tensor_recv_prev_func = None
        tensor_recv_next_func = None

        if config.variable_seq_lengths or config.mtp_standalone:
            recv_prev_shape, recv_next_shape = self._communicate_shapes(
                tensor_send_next, tensor_send_prev, recv_prev, recv_next
            )
        else:
            recv_prev_shape = tensor_shape
            recv_next_shape = tensor_shape

        def create_tensor_recv_prev():
            return torch.empty(
                recv_prev_shape,
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=config.pipeline_dtype,
            )

        def create_tensor_recv_next():
            return torch.empty(
                recv_next_shape,
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=config.pipeline_dtype,
            )

        if recv_prev:
            if config.pipeline_dtype is None:
                raise RuntimeError("pipeline_dtype must be provided if recv_prev is True")
            if tensor_shape is None:
                raise RuntimeError(
                    "tensor_shape must be specified if recv_prev is True. "
                    "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
                )
            tensor_recv_prev_func = create_tensor_recv_prev

        if recv_next:
            if config.pipeline_dtype is None:
                raise RuntimeError("dtype must be provided if recv_next is True")
            if tensor_shape is None:
                raise RuntimeError(
                    "tensor_shape must be specified if recv_next is True. "
                    "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
                )
            tensor_recv_next_func = create_tensor_recv_next

        # Send tensors in both the forward and backward directions as appropriate.
        if config.use_ring_exchange_p2p:

            def _ring_exchange_wrapper(**kwargs):
                torch.distributed.ring_exchange(**kwargs)
                return []

            p2p_func = _ring_exchange_wrapper
        elif config.batch_p2p_comm:
            # TODO dongcl
            # assert wait_on_reqs
            p2p_func = _batched_p2p_ops
        else:
            p2p_func = _p2p_ops

        pp_group = self.pp_group
        next_rank = self.next_rank
        prev_rank = self.prev_rank

        if config.use_ring_exchange_p2p or config.batch_p2p_comm:
            reqs = []
        else:
            reqs = {}

        tensor_recv_prev = None
        tensor_recv_next = None
        if tensor_recv_prev_func is not None:
            tensor_recv_prev = tensor_recv_prev_func()

        if tensor_recv_next_func is not None:
            tensor_recv_next = tensor_recv_next_func()

        p2p_reqs = p2p_func(
            tensor_send_prev=tensor_send_prev,
            tensor_recv_prev=tensor_recv_prev,
            tensor_send_next=tensor_send_next,
            tensor_recv_next=tensor_recv_next,
            group=pp_group,
            prev_pipeline_rank=prev_rank,
            next_pipeline_rank=next_rank,
        )
        if isinstance(p2p_reqs, list):
            reqs.extend(p2p_reqs)
        else:
            reqs.update(p2p_reqs)

        if wait_on_reqs and len(reqs) > 0:
            for req in reqs if isinstance(reqs, list) else reqs.values():
                req.wait()
            reqs = None

        if config.batch_p2p_comm and config.batch_p2p_sync:
            # To protect against race condition when using batch_isend_irecv().
            # User should assert that we have a modern enough PyTorch to not need this
            if not get_adaptor_args().enable_vocab_parallel:
                torch.cuda.synchronize()

        return tensor_recv_prev, tensor_recv_next, reqs

    @nvtx_decorator()
    def recv_forward(
        self, tensor_shapes, is_first_stage: bool
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Receive tensor from previous rank in pipeline (forward receive)."""
        unwrap_tensor_shapes = False
        if is_single_shape(tensor_shapes):
            unwrap_tensor_shapes = True
            tensor_shapes = [tensor_shapes]
        input_tensors = []
        config = self.config
        for tensor_shape in tensor_shapes:
            if is_first_stage:
                input_tensor = None
                if get_adaptor_args().enable_vocab_parallel:
                    input_tensor = VocabInputStore.forward_get(remove=False)
            else:
                if config.timers is not None:
                    config.timers('forward-recv', log_level=2).start()
                input_tensor, _, _ = self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=None,
                    recv_prev=True,
                    recv_next=False,
                    tensor_shape=tensor_shape,
                )
                if config.timers is not None:
                    config.timers('forward-recv').stop()
            input_tensors.append(input_tensor)
        if unwrap_tensor_shapes:
            return input_tensors[0]
        return input_tensors

    @nvtx_decorator()
    def send_backward_recv_forward(
        self, input_tensor_grads, tensor_shapes, is_first_stage: bool
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Batched send and recv with previous rank in pipeline."""
        config = self.config
        unwrap_input_tensor_grads = False
        if not isinstance(input_tensor_grads, list):
            unwrap_input_tensor_grads = True
            input_tensor_grads = [input_tensor_grads]
        if not isinstance(tensor_shapes, list):
            tensor_shapes = [tensor_shapes]
        input_tensors = []
        for input_tensor_grad, tensor_shape in zip(input_tensor_grads, tensor_shapes):
            if is_first_stage:
                input_tensor = None
                if get_adaptor_args().enable_vocab_parallel:
                    input_tensor = VocabInputStore.forward_get(remove=False)
            else:
                if config.timers is not None:
                    config.timers('backward-send-forward-recv', log_level=2).start()
                input_tensor, _, _ = self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=input_tensor_grad,
                    recv_prev=True,
                    recv_next=False,
                    tensor_shape=tensor_shape,
                )
                if config.timers is not None:
                    config.timers('backward-send-forward-recv').stop()
            input_tensors.append(input_tensor)
        if unwrap_input_tensor_grads:
            return input_tensors[0]
        return input_tensors

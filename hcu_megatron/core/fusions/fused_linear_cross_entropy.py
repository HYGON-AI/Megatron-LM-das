# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""PR2256-style Linear Cross Entropy implementation for HCU Megatron.

Backend validation and architecture-specific native loading remain in the
``linear_cross_entropy`` subpackage.
"""

import typing
from functools import lru_cache

import torch


class Platform:
    """
    Singleton binding the PR2256 call contract to the HCU backend.
    """

    _instance: typing.Optional["Platform"] = None

    def __new__(cls) -> "Platform":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        from .linear_cross_entropy import entry as gpu_entry

        self.forward_func: typing.Callable[..., typing.Any] = gpu_entry.forward
        self.backward_func: typing.Callable[..., typing.Any] = gpu_entry.backward

        self._initialized = True


@lru_cache(maxsize=1)
def _get_platform() -> Platform:
    """
    Helper function to lazy initialize the platform.
    """
    return Platform()


class LinearCrossEntropy(torch.autograd.Function):
    """
    This class implements a custom autograd function for linear and cross entropy,
    whose equivalent logic in PyTorch is:
        ```python
        def torch_entropy(hidden, weight, labels):
            logits = torch.matmul(hidden, weight)
            logprobs = torch.nn.functional.cross_entropy(logits, labels)
            return logprobs
        ```
    """

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        tp_group: typing.Optional[torch.distributed.ProcessGroup] = None,
        reduction: typing.Literal["none", "sum", "mean"] = "mean",
        ignore_index: int = -100,
        sequence_parallel: bool = False,
    ) -> torch.Tensor:
        """
        The forward pass of the Linear Cross Entropy.
        The HCU overlay retains the PR2256 call surface, but currently accepts
        only its documented DP contract. TP, SP, and other
        reductions fail closed in the backend before native launch.

        If tp_group is not None, the weight tensor to each TP rank should be
        (global_vocab_size // world_size, dim).
        Note that each of the ranks should get equal shards along the vocab_size dimension.

        Args:
            @param hidden: the input tensor with shape (num_tokens, dim)
            @param weight: the lm_head weight tensor with shape (local_vocab_size, dim)
            @param labels: the labels tensor with shape (num_tokens,)
            @param tp_group: the distributed process group for TP.
            @param reduction: Default to "mean", and can be one of "none", "sum", "mean".
            @param ignore_index: The index to ignore. Default to -100.
            @param sequence_parallel: Whether to use sequence parallel. Default to False.
        Returns:
            @return: logprobs with shape
                - either (num_tokens,) when reduction is "none"
                - or (1,) when reduction is "mean" or "sum"

        tp_group is None ----------------------------------> DP
                B
            A   C
        tp_group is not None & sequence_parallel is False -> TP
                B0  B1
            A   C0  C1
        tp_group is not None & sequence_parallel is True --> SP
                B0  B1
            A0  C0  XX
            A1  XX  C1

        When tp_group is not None, the weight tensor will be split along the vocab_size
        dimension, which means each rank will get equal shards along the global_vocab_size
        dimension. Specifically, the weight tensor to each rank will be (local_vocab_size, dim).
        And there is an assumption that each rank will get the same local_vocab_size.

        When sequence_parallel is True, the hidden tensor will be split along the
        sequence length dimension, which means each rank will get equal shards along
        the sequence length dimension. Specifically, the hidden tensor to each rank
        will be (local_num_tokens, dim). And there is an assumption that each rank
        will get the same local_num_tokens.

        In TP forward pass, the hidden tensor and label tensor shall be identical
        among all TP ranks, and it's user's responsibility to ensure the hidden tensor
        is identical among all TP ranks. Then this operation will produce identical
        logprobs among all TP ranks.

        In TP backward pass, the gradient of the logprobs shall be identical among all
        TP ranks, and it's user's responsibility to ensure the gradient of the logprobs
        is identical among all TP ranks. Then this operation will produce distinct gradients
        for the local weight tensor, and identical gradients for the hidden tensor.

        ```python
        # ------------ forward pass ------------ #
        hidden = tp_group.broadcast(hidden, src=0) # handled by framework
        labels = tp_group.broadcast(labels, src=0) # handled by framework
        logprobs = linear_cross_entropy(...)
        # each rank will get the same logprobs

        # ------------ backward pass ------------ #
        g_logprobs = tp_group.broadcast(g_logprobs, src=0) # handled by framework
        d_hidden, d_weight = torch.autograd.grad(...)
        # each rank will get the same d_hidden,
        # and distinct d_weight for local weight shard
        ```

        In SP forward pass, the hidden tensor shall be split along the sequence length dimension,
        and the label tensor shall be identical among all TP ranks.
        Then this operation will produce identical logprobs among all TP ranks.

        In SP backward pass, the gradient of the logprobs shall be identical among all TP ranks,
        Then this operation will produce distinct gradients for the local hidden tensor
        and local weight tensor.
        ```python
        # ------------ forward pass ------------ #
        hidden = global_hidden[tp_rank] # handled by framework
        labels = tp_group.broadcast(labels, src=0) # handled by framework
        logprobs = linear_cross_entropy(...)
        # each rank will get the same logprobs

        # ------------ backward pass ------------ #
        g_logprobs = tp_group.broadcast(g_logprobs, src=0) # handled by framework
        d_hidden, d_weight = torch.autograd.grad(...)
        # each rank will get distinct local d_hidden and d_weight
        ```
        """
        with torch.cuda.nvtx.range("LinearCrossEntropy-forward"):
            (
                logprobs,
                _maximum,
                _acc,
                _num_valid_tokens,
                tp_rank,
                tp_world_size,
                global_hidden,
            ) = _get_platform().forward_func(
                hidden, weight, labels, tp_group, reduction, ignore_index, sequence_parallel
            )
            ctx.save_for_backward(global_hidden, weight, labels, _maximum, _acc, _num_valid_tokens)
            ctx.tp_group = tp_group
            ctx.ignore_index = ignore_index
            ctx.reduction = reduction
            ctx.tp_rank = tp_rank
            ctx.tp_world_size = tp_world_size
            ctx.sequence_parallel = sequence_parallel

        return logprobs

    @staticmethod
    def backward(
        ctx, dlogprobs: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, None, None, None, None, None]:
        """
        The backward pass of the Linear Cross Entropy.
        Args:
            dlogprobs (torch.Tensor): The gradient of the cross entropy, with shape
                - either (num_tokens,) when reduction is "none"
                - or (1,) when reduction is "mean" or "sum"
        Returns:
            dhidden (torch.Tensor): The gradient of the hidden.
            dweight (torch.Tensor): The gradient of the weight.
        """
        with torch.cuda.nvtx.range("LinearCrossEntropy-backward"):
            (global_hidden, weight, labels, _maximum, _accu, _num_valid_tokens) = ctx.saved_tensors

            tp_group = ctx.tp_group
            ignore_index = ctx.ignore_index
            reduction = ctx.reduction
            tp_rank = ctx.tp_rank
            tp_world_size = ctx.tp_world_size
            sequence_parallel = ctx.sequence_parallel

            d_hidden, d_weight = _get_platform().backward_func(
                dlogprobs,
                global_hidden,
                weight,
                labels,
                _maximum,
                _accu,
                _num_valid_tokens,
                reduction,
                ignore_index,
                tp_group,
                tp_rank,
                tp_world_size,
                sequence_parallel,
            )

        return d_hidden, d_weight, None, None, None, None, None


def linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    tp_group: typing.Optional[torch.distributed.ProcessGroup] = None,
    reduction: typing.Literal["none", "sum", "mean"] = "mean",
    ignore_index: int = -100,
    sequence_parallel: bool = False,
) -> torch.Tensor:
    """
    helper function for linear cross entropy.
    """
    _impl = LinearCrossEntropy.apply
    return _impl(hidden, weight, labels, tp_group, reduction, ignore_index, sequence_parallel)


def linear_cross_entropy_for_training(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute HCU Linear CE while preserving Megatron's token-loss contract.

    Megatron's GPT training loop expects the model to return a contiguous
    ``[batch, sequence]`` tensor of token losses.  The current HCU native
    operator deliberately exposes only a scalar, mean-reduced DP loss.  This
    adapter masks labels before the native call, then expands and scales that
    scalar so the existing training loss function still computes the same
    summed loss and gradient without materializing vocabulary logits.

    ``loss_mask`` must be binary.  A non-binary weighted loss mask cannot be
    represented by the current mean-only native operator.
    """
    if labels.shape != loss_mask.shape:
        raise ValueError(
            "labels and loss_mask must have identical [batch, sequence] shapes, got "
            f"{tuple(labels.shape)} and {tuple(loss_mask.shape)}"
        )
    if labels.dim() != 2:
        raise ValueError(
            "HCU Linear Cross Entropy training expects labels and loss_mask with "
            f"[batch, sequence] shape, got {tuple(labels.shape)}"
        )
    if labels.numel() == 0:
        raise ValueError("HCU Linear Cross Entropy training requires at least one token")
    if hidden.dim() != labels.dim() + 1:
        raise ValueError(
            "hidden must have one more dimension than labels, got "
            f"hidden.dim={hidden.dim()} and labels.dim={labels.dim()}"
        )
    expected_hidden_shape = (labels.shape[1], labels.shape[0])
    if tuple(hidden.shape[:-1]) != expected_hidden_shape:
        raise ValueError(
            "hidden must use Megatron's [sequence, batch, hidden] layout matching labels; "
            f"got hidden {tuple(hidden.shape)} and labels {tuple(labels.shape)}"
        )

    # The standard pretraining data path supplies a binary mask.  Avoid a
    # device-to-host synchronization here while still rejecting weighted masks,
    # whose semantics cannot be represented by the current mean-only native op.
    torch._assert_async(
        torch.all((loss_mask == 0) | (loss_mask == 1)),
        "HCU Linear Cross Entropy requires a binary loss_mask",
    )
    valid_tokens = (loss_mask != 0) & (labels != ignore_index)
    native_labels = torch.where(
        valid_tokens, labels, torch.full_like(labels, ignore_index)
    ).transpose(0, 1).contiguous()
    num_valid_tokens = valid_tokens.sum()

    # Native mean CE returns NaN when every token is ignored.  Keep one benign
    # target for the launch, then multiply its contribution by zero below.  The
    # returned loss and all gradients are consequently zero, matching the
    # existing masked-token training contract.
    safe_native_labels = native_labels.view(-1).clone()
    safe_native_labels[0] = torch.where(
        num_valid_tokens == 0,
        torch.zeros((), dtype=safe_native_labels.dtype, device=safe_native_labels.device),
        safe_native_labels[0],
    )
    safe_native_labels = safe_native_labels.view_as(native_labels)

    native_loss = linear_cross_entropy(
        hidden.contiguous(),
        weight.contiguous(),
        safe_native_labels,
        tp_group=None,
        reduction="mean",
        ignore_index=ignore_index,
        sequence_parallel=False,
    )

    # ``pretrain_gpt.loss_func`` subsequently computes
    # ``sum(token_losses * loss_mask)``.  Scale the native mean so this equals
    # the conventional sum over valid tokens, including labels that were
    # already equal to ``ignore_index``.
    schedule_token_count = loss_mask.float().sum().clamp_min(1.0)
    schedule_loss = native_loss * num_valid_tokens.to(native_loss.dtype) / schedule_token_count
    return schedule_loss.expand_as(labels).contiguous()


__all__ = ["linear_cross_entropy", "linear_cross_entropy_for_training", "LinearCrossEntropy"]

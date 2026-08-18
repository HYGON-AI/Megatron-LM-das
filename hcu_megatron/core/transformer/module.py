# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import torch

from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.module import Float16Module as MegatronCoreFloat16Module
from megatron.core.transformer.module import float16_to_fp32, fp32_to_float16


class Float16Module():
    def __init__(self, config: TransformerConfig, module: torch.nn.Module, is_embedding_chunk=False, force_output_fp32=False):
        super(MegatronCoreFloat16Module, self).__init__(config)
        self.config = config
        self.fp16 = config.fp16
        self.bf16 = config.bf16
        self.vp_stage = getattr(module, 'vp_stage', None)

        if self.fp16:
            self.add_module('module', module.half())

            def float16_convertor(val):
                return val.half()

        elif self.bf16:
            self.add_module('module', module.bfloat16())

            def float16_convertor(val):
                return val.bfloat16()

        else:
            raise Exception('Either config.fp16 or config.bf16 should be True.')

        self.float16_convertor = float16_convertor

        self.is_embedding_chunk = is_embedding_chunk
        self.force_output_fp32 = force_output_fp32

    def forward(self, *inputs, fp32_output=True, **kwargs):
        """
        Execute the wrapped module in model precision and optionally upcast outputs to fp32.

        On the first pipeline stage, positional/keyword tensor inputs are converted to the
        module precision (fp16 or bf16) before invoking the wrapped module. The wrapped module
        is called with the provided inputs and keyword arguments. On the last pipeline stage
        only, outputs are upcast to fp32 if ``fp32_output`` is True; otherwise, outputs are
        returned in the model precision (fp16/bf16).

        Args:
            *inputs: Positional inputs forwarded to the wrapped module (converted to fp16/bf16 on
                the pipeline first stage).
            fp32_output (bool, keyword-only): If True (default), upcast outputs to fp32 on the
                pipeline last stage. Has no effect on non-last stages. Set to False to keep outputs
                in model precision when downstream consumers expect half precision or to avoid
                extra casts.
            **kwargs: Keyword arguments forwarded to the wrapped module.

        Returns:
            The wrapped module's outputs, potentially upcast to fp32 depending on pipeline stage
            and ``fp32_output``.
        """
        if self.is_embedding_chunk:
            inputs = fp32_to_float16(inputs, self.float16_convertor)
        outputs = self.module(*inputs, **kwargs)
        if (
            self.force_output_fp32
            and fp32_output is True
        ):
            outputs = float16_to_fp32(outputs)
        return outputs

    def backward_dw(self, *inputs, **kwargs):
        """
        Calls the wrapped module's backward_dw() method.
        """
        return self.module.backward_dw(*inputs, **kwargs)

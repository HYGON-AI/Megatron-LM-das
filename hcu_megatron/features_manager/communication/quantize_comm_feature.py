# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
from argparse import ArgumentParser

from ..feature import AbstractFeature


QUANT_BIT_DEFAULT_GROUP_SIZE_MAP = {
    4: 32,
    8: 128,
}
QUANT_BIT_GROUP_SIZE_CHOICES_MAP = {
    4: {16, 32},
    8: {64, 128},
}

class QuantizeCommFeature(AbstractFeature):

    def __init__(self):
        super().__init__('use-quantize-comm', 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)

        group.add_argument('--use-quantize-comm',
                           default=False,
                           action="store_true",
                           help='use quantized communication')
        group.add_argument('--quant-comm-bits', type=int,
                           default=8,
                           choices=[4, 8],
                           help='the number of bits to quantize to, supported numbers are (4, 8)')
        group.add_argument('--quant-group-size', type=int,
                           default=None,
                           help='the group size to use for quantization. If not specified, uses per-column quantization')
        group.add_argument('--quant-scale-dtype', type=str,
                           default="bf16",
                           choices=["bf16", "fp16", "fp32"],
                           help='the dtype of quantization scale')

    def validate_args(self, args):
        assert args.quant_comm_bits in {4, 8}, f"quant_comm_bits {args.quant_comm_bits} only accepts values from [4, 8]"
        if (
            args.quant_group_size is not None
            and args.quant_group_size not in QUANT_BIT_GROUP_SIZE_CHOICES_MAP[args.quant_comm_bits]
        ):
            raise ValueError(f"quant_group_size {args.quant_group_size} only accepts values from {QUANT_BIT_GROUP_SIZE_CHOICES_MAP[args.quant_comm_bits]}")

        return args

    def register_patches(self, patch_manager, args):
        from hcu_megatron.core.tensor_parallel.mappings import all_to_all

        if args.use_quantize_comm:
            patch_manager.register_patch('megatron.core.tensor_parallel.mappings.all_to_all',
                                         all_to_all)

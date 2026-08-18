# Copyright (c) 2026 Hygon Information Technology Co., Ltd."
# SPDX-License-Identifier: Apache-2.0

import enum

class DualpipeVChunkType(enum.Enum):
    """Chunk type
    first_block: first transformer block.
        The block contains only transformer layers if enable_vocab_parallel is true;
        otherwise, the block is composed of an embedding layer and several transformer layers.
    second_block: second transformer block.
        The block contains only transformer layers if enable_vocab_parallel is true;
        otherwise, the block comprises several transformer layers, followed by an output layer and a loss layer.
    output: output layer
    embedding: embedding layer
    loss: loss layer
    """

    first_block = 0
    second_block = 1
    output = 2
    embedding = 3
    loss = 4

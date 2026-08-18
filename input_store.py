# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# This code was adopted from https://github.com/sail-sg/VocabularyParallelism
from megatron.core import mpu
from megatron.training import get_args

from hcu_megatron.core import parallel_state
from hcu_megatron.core.transformer.enums import DualpipeVChunkType


class InputStore:
    """
    For storing and retrieving batch input that are partially unused.
    """

    cache = []

    @classmethod
    def save_batch(cls, microbatch_id, data):
        while len(cls.cache) <= microbatch_id:
            cls.cache.append(None)
        cls.cache[microbatch_id] = data

    @classmethod
    def get_batch(cls, microbatch_id):
        contents = cls.cache[microbatch_id]

        if get_args().schedule_method == "dualpipev":
            if (
                parallel_state.get_virtual_vocab_parallel_chunk() == DualpipeVChunkType.loss.value
            ):
                cls.cache[microbatch_id] = None
            elif (
                not mpu.is_pipeline_first_stage()
                and (parallel_state.get_virtual_vocab_parallel_chunk() == DualpipeVChunkType.output.value)
            ):
                cls.cache[microbatch_id] = None
            return contents

        if (
            parallel_state.get_virtual_vocab_parallel_chunk() == 3
        ):
            cls.cache[microbatch_id] = None
        elif (
            not mpu.is_pipeline_last_stage()
            and (parallel_state.get_virtual_vocab_parallel_chunk() == 1)
        ):
            cls.cache[microbatch_id] = None
        return contents

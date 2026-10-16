# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import inspect

from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.dot_product_attention import DotProductAttention as MCoreDotProductAttention
from megatron.core.transformer.enums import AttnBackend

from megatron.training.models.gpt import (
    default_layer_spec,
    mtp_block_spec,
)
from megatron.training.vocab_utils import calculate_padded_vocab_size

from hcu_megatron.core.parallel_state import get_parallel_group_ranks
from hcu_megatron.training.utils import print_rank_message


class GPTModelBuilder():

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> GPTModel:
        """Build a single ``MCoreGPTModel`` stage.

        Args:
            pg_collection: Process groups for distributed training
            pre_process: Include embedding layer
            post_process: Include output layer
            vp_stage: Virtual pipeline stage

        Returns:
            The constructed model
        """
        transformer_layer_spec = self._model_config.transformer_layer_spec
        if transformer_layer_spec is None:
            transformer_layer_spec = default_layer_spec(self._model_config, vp_stage)
        elif not isinstance(transformer_layer_spec, ModuleSpec) and callable(transformer_layer_spec):
            # Check if the transformer_layer_spec function accepts vp_stage parameter
            if "vp_stage" in inspect.signature(transformer_layer_spec).parameters:
                transformer_layer_spec = transformer_layer_spec(self._model_config, vp_stage=vp_stage)
            else:
                transformer_layer_spec = transformer_layer_spec(self._model_config)

        assert self._model_config.vocab_size is not None, "vocab_size must be configured before calling build_model()"
        if self._model_config.should_pad_vocab:
            padded_vocab_size = calculate_padded_vocab_size(
                self._model_config.vocab_size,
                self._model_config.make_vocab_size_divisible_by,
                self._model_config.transformer.tensor_model_parallel_size,
            )
        else:
            padded_vocab_size = self._model_config.vocab_size

        mtp_spec = mtp_block_spec(self._model_config, transformer_layer_spec, vp_stage=vp_stage)

        # override spec with local backend if configured
        if self._model_config.attention_backend == AttnBackend.local:
            if hasattr(transformer_layer_spec, "submodules"):
                transformer_layer_spec.submodules.self_attention.submodules.core_attention = MCoreDotProductAttention

        # Determine pre/post flags if not provided using vp + pp stage
        vp_size = self._model_config.virtual_pipeline_model_parallel_size
        if pre_process is None:
            pre_process = is_vp_first_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_first_stage(pg_collection.pp)
        if post_process is None:
            post_process = is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_last_stage(pg_collection.pp)

        model = GPTModel(
            config=self._model_config.transformer,
            transformer_layer_spec=transformer_layer_spec,
            mtp_block_spec=mtp_spec,
            vocab_size=padded_vocab_size,
            max_sequence_length=self._model_config.seq_length,
            fp16_lm_cross_entropy=self._model_config.fp16_lm_cross_entropy,
            parallel_output=self._model_config.parallel_output,
            share_embeddings_and_output_weights=self._model_config.share_embeddings_and_output_weights,
            position_embedding_type=self._model_config.position_embedding_type,
            rotary_percent=self._model_config.rotary_percent,
            rotary_base=self._model_config.rotary_base,
            rope_scaling=self._model_config.rope_scaling,
            rope_scaling_factor=self._model_config.rope_scaling_factor,
            seq_len_interpolation_factor=self._model_config.seq_len_interpolation_factor,
            scatter_embedding_sequence_parallel=self._model_config.scatter_embedding_sequence_parallel,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )

        print_rank_message(model, rank_id=get_parallel_group_ranks()["PIPELINE_MODEL_PARALLEL_GROUP"][0])

        return model

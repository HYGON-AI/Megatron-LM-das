# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import sys
import types
import argparse
import torch


class MegatronAdaptation:
    """
        A module manager supports adaptation registration, application and execution.
    """
    _patch_info_collection = {}
    _args = None

    @classmethod
    def execute(cls):
        """
        Execute adaptations.
        """
        for adaptation in [CoreAdaptation(), LegacyAdaptation()]:
            adaptation.execute()
        MegatronAdaptation.apply()
        # MegatronAdaptation.post_execute()

    @classmethod
    def register(cls, orig_func_name, new_func=None, force_patch=False, create_dummy=False):
        """
        Register adaptations into collection.
        """
        if orig_func_name not in cls._patch_info_collection:
            from .patch_utils import Patch
            cls._patch_info_collection[orig_func_name] = Patch(orig_func_name, new_func, create_dummy)
        else:
            cls._patch_info_collection.get(orig_func_name).set_patch_func(new_func, force_patch)

    @classmethod
    def apply(cls):
        """
        Apply adaptations.
        """
        for patch in cls._patch_info_collection.values():
            patch.apply_patch()

    @classmethod
    def post_execute(cls):
        """
        Execute after other adaptations.
        """
        from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
        from megatron.core.transformer.transformer_block import TransformerBlock


class MegatronAdaptationABC:
    """
    Abstract class for adaptation.
    """
    @abc.abstractmethod
    def execute(self):
        """
        Do Adaptation
        """


class CoreAdaptation(MegatronAdaptationABC):
    """
    Adaptations for models in Megatron-LM Core structure.
    """
    def execute(self):
        self.patch_core_distributed()
        self.patch_core_models()
        self.patch_core_transformers()
        self.patch_tensor_parallel()
        self.patch_training()
        self.patch_miscellaneous()

    def patch_core_distributed(self):
        # Mtp share embedding
        from ..core.distributed.finalize_model_grads import _allreduce_word_embedding_grads
        MegatronAdaptation.register('megatron.core.distributed.finalize_model_grads._allreduce_word_embedding_grads',
                                    _allreduce_word_embedding_grads)

    def patch_core_models(self):
        from ..core.models.common.embeddings.language_model_embedding import (
            language_model_embedding_forward,
            language_model_embedding_init_func
        )
        from ..core.models.gpt.gpt_model import (
            gpt_model_forward,
            gpt_model_init,
            shared_embedding_or_mtp_embedding_weight
        )

        # Embedding
        MegatronAdaptation.register(
            'megatron.core.models.common.embeddings.language_model_embedding.LanguageModelEmbedding.__init__',
            language_model_embedding_init_func)
        MegatronAdaptation.register(
            'megatron.core.models.common.embeddings.language_model_embedding.LanguageModelEmbedding.forward',
            language_model_embedding_forward)

        # GPT Model
        MegatronAdaptation.register('megatron.core.models.gpt.gpt_model.GPTModel.forward', gpt_model_forward)
        MegatronAdaptation.register('megatron.core.models.gpt.gpt_model.GPTModel.__init__', gpt_model_init)

        from megatron.core.models.gpt.gpt_model import GPTModel
        setattr(GPTModel, 'shared_embedding_or_mtp_embedding_weight', shared_embedding_or_mtp_embedding_weight)

    def patch_core_transformers(self):
        from ..core import transformer_block_init_wrapper, transformer_block_forward
        from ..core.transformer.transformer_config import TransformerConfig, MLATransformerConfig

        # Transformer block
        MegatronAdaptation.register('megatron.core.transformer.transformer_block.TransformerBlock.__init__',
                                    transformer_block_init_wrapper)
        MegatronAdaptation.register('megatron.core.transformer.transformer_block.TransformerBlock.forward',
                                    transformer_block_forward)

        # Transformer config
        MegatronAdaptation.register('megatron.core.transformer.transformer_config.TransformerConfig',
                                    TransformerConfig)
        # Transformer config
        MegatronAdaptation.register('megatron.core.transformer.transformer_config.MLATransformerConfig',
                                    MLATransformerConfig)

    def patch_tensor_parallel(self):
        from ..core import vocab_parallel_embedding_forward, vocab_parallel_embedding_init

        MegatronAdaptation.register('megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward',
                                    vocab_parallel_embedding_forward)
        MegatronAdaptation.register('megatron.core.tensor_parallel.layers.VocabParallelEmbedding.__init__',
                                    vocab_parallel_embedding_init)

    def patch_training(self):
        from ..training.tokenizer import build_tokenizer
        from ..training.initialize import initialize_megatron

        MegatronAdaptation.register('megatron.training.tokenizer.tokenizer.build_tokenizer',
                                    build_tokenizer)
        MegatronAdaptation.register('megatron.training.initialize.initialize_megatron',
                                    initialize_megatron)

    def patch_miscellaneous(self):
        from ..training.arguments import parse_args

        MegatronAdaptation.register('megatron.training.arguments.parse_args', parse_args)


class LegacyAdaptation(MegatronAdaptationABC):
    """
        Adaptations for models in legacy structure.
    """

    def execute(self):
        pass


MegatronAdaptation.execute()

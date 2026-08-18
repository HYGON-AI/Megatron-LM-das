# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import logging

import torch

from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_vp_first_stage,
)
from megatron.core.utils import unwrap_model


class LanguageModule():
    def setup_embeddings_and_output_layer(self) -> None:
        """Sets up embedding layer in first stage and output layer in last stage.

        This function initalizes word embeddings in the final stage when we are
        using pipeline parallelism and sharing word embeddings, and sets up param
        attributes on the embedding and output layers.

        Parameter attributes set:
        - `is_embedding_or_output_parameter`: True for embedding + output layer weights.
          Used by decoupled_lr, Muon optimizer, and other Megatron features.
        - `is_embedding_parameter`: True for MuP "embedding-class" parameters.
          Used by MuP for table-8 style optimizer grouping (base LR/eps for vector-like params).
        """

        # Mark embedding and output layer for decoupled_lr and other features.
        # This is the original Megatron attribute used by decoupled_lr, Muon, FSDP, etc.
        # Include MTP-stage embedding too: it is a duplicated copy of the pre_process
        # embedding (kept in sync via cross-stage all-reduce). Without this tag, the
        # LayerWise distributed optimizer routes it to its Muon-managed buffer and
        # `_emit_bucket(shared_embedding=True)` replicates the (vocab x hidden) tensor
        # across all dp_size shards, blowing up the chunk's buffer by ~8x.
        if (self.has_vocab_embedding or getattr(self, 'mtp_process', False)) and hasattr(self, 'embedding'):
            self.embedding.word_embeddings.weight.is_embedding_or_output_parameter = True
        if (
            self.post_process
            and hasattr(self, 'output_layer')
            and self.output_layer.weight is not None
        ):
            self.output_layer.weight.is_embedding_or_output_parameter = True

        # Mark embedding-class parameters for MuP optimizer grouping.
        # Under MuP table-8-style grouping, embeddings/output use base LR/eps while
        # hidden matrix-like params use width-scaled LR/eps.
        mtp_process = getattr(self, 'mtp_process', False)
        if self.config.use_mup and (self.pre_process or mtp_process) and hasattr(self, 'embedding'):
            for param in self.embedding.parameters():
                param.is_embedding_parameter = True
        if (
            self.config.use_mup
            and self.post_process
            and hasattr(self, 'output_layer')
            and self.output_layer.weight is not None
        ):
            self.output_layer.weight.is_embedding_parameter = True

        # If share_embeddings_and_output_weights is True, we need to maintain duplicated
        # embedding weights in post processing stage. If use Multi-Token Prediction (MTP),
        # we also need to maintain duplicated embedding weights in mtp process stage.
        # So we need to copy embedding weights from pre processing stage as initial parameters
        # in these cases.
        if (
            (not self.share_embeddings_and_output_weights and not getattr(self.config, 'mtp_num_layers', 0))
            or self.config.schedule_method == 'dualpipev'
        ):
            return

        if self.config.pipeline_model_parallel_size == 1:
            # Zero out wgrad if sharing embeddings between two layers on same
            # pipeline stage to make sure grad accumulation into main_grad is
            # correct and does not include garbage values (e.g., from torch.empty).
            self.shared_embedding_or_output_weight().zero_out_wgrad = True
            return

        if (
            is_vp_first_stage(self.vp_stage, self.vp_size)
            and is_pp_first_stage(self.pp_group)
            and self.has_vocab_embedding
            and not self.post_process
        ):
            self.shared_embedding_or_output_weight().shared_embedding = True

        if (
            (self.post_process and self.share_embeddings_and_output_weights)
            or getattr(self, 'mtp_process', False)
        ) and not self.has_vocab_embedding:
            # disabled the following assertion, intended behavior in vocab parallel (TODO dongcl)
            # assert not (
            #     is_vp_first_stage(self.vp_stage, self.vp_size) and is_pp_first_stage(self.pp_group)
            # )
            # set weights of the duplicated embedding to 0 here,
            # then copy weights from pre processing stage using all_reduce below.
            weight = self.shared_embedding_or_output_weight()
            weight.data.fill_(0)
            weight.shared = True
            weight.shared_embedding = True
            # Keep optimizer grouping consistent for tied embedding/output copies.
            if self.config.use_mup:
                weight.is_embedding_parameter = True

        # Parameters are shared between the word embeddings layers, and the
        # heads at the end of the model. In a pipelined setup with more than
        # one stage, the initial embedding layer and the head are on different
        # workers, so we do the following:
        # 1. Create a second copy of word_embeddings on the last stage, with
        #    initial parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that
        #    the two copies of word_embeddings start off with the same
        #    parameter values.
        # 3. In the training loop, before an all-reduce between the grads of
        #    the two word_embeddings layers to ensure that every applied weight
        #    update is the same on both stages.

        # Ensure that first and last stages have the same initial parameter
        # values.
        if torch.distributed.is_initialized():
            if self._is_in_embd_group() and not self.config.init_model_with_meta_device:
                weight = self.shared_embedding_or_output_weight()
                weight.data = weight.data.cuda()
                torch.distributed.all_reduce(weight.data, group=self.embd_group)

        elif not getattr(LanguageModule, "embedding_warning_printed", False):
            logging.getLogger(__name__).warning(
                "Distributed processes aren't initialized, so the output layer "
                "is not initialized with weights from the word embeddings. "
                "If you are just manipulating a model this is fine, but "
                "this needs to be handled manually. If you are training "
                "something is definitely wrong."
            )
            LanguageModule.embedding_warning_printed = True


_SHARED_EMBEDDING = None

def get_shared_embedding_from_dual_chunk():
    assert _SHARED_EMBEDDING is not None
    return _SHARED_EMBEDDING


def set_shared_embedding_from_dual_chunk(model1, model2, enable_vocab_parallel=False):
    global _SHARED_EMBEDDING
    if _SHARED_EMBEDDING is not None:
        return

    model1 = unwrap_model(model1)
    model2 = unwrap_model(model2)

    if enable_vocab_parallel:
        if model1.split_vocab_embedding:
            _SHARED_EMBEDDING = model1.embedding.word_embeddings.weight
        elif model2.split_vocab_embedding:
            _SHARED_EMBEDDING = model2.embedding.word_embeddings.weight
    else:
        if model1.pre_process:
            _SHARED_EMBEDDING = model1.embedding.word_embeddings.weight
        elif model2.pre_process:
            _SHARED_EMBEDDING = model2.embedding.word_embeddings.weight

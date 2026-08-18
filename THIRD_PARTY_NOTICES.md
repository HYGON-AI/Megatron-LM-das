# Third Party Notices

This document lists third-party projects whose source code is included in, adapted by,
or referenced by `hcu_megatron`. Hygon acknowledges the original
authors and preserves their licenses in accordance with the terms below.

## Megatron-LM

- Source: https://github.com/NVIDIA/Megatron-LM
- Location: `3rdparty/Megatron-LM` (git submodule)
- License: Apache License, Version 2.0 (files under the `megatron/core` subtree) and the
  original NVIDIA BSD-style license (see the project `LICENSE` file for the full text).
- Original copyright: Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
- Usage: `hcu_megatron` extends Megatron-LM by patching and subclassing many of its
  modules (`megatron.core.*`, `megatron.training.*`) — see `hcu_megatron/megatron_adaptor.py`
  and `hcu_megatron/patch_utils.py`. Numerous source files under `hcu_megatron/core/**`
  retain the upstream NVIDIA copyright header alongside the Hygon modification notice.
- Hygon modifications: HCU platform adaptations, new features, memory optimizations,
  pipeline-parallel schedules, and MoE / VLM enhancements.

## Megatron-Bridge

- Source: https://github.com/NVIDIA/Megatron-Bridge
- Location: `3rdparty/Megatron-Bridge` (git submodule)
- License: Apache License, Version 2.0
- Original copyright: Copyright (c) NVIDIA CORPORATION & AFFILIATES.
- Usage: bridge utilities used together with Megatron-LM on the HCU platform.

## Megatron-Energon

- Source: https://github.com/NVIDIA/Megatron-Energon
- Location: `3rdparty/Megatron-Energon` (git submodule)
- License: Apache License, Version 2.0 (see the upstream `LICENSE` file for the full text).
- Original copyright: Copyright (c) NVIDIA CORPORATION & AFFILIATES.
- Usage: dataset streaming and multi-modal data loading utilities used by the VLM
  training pipeline.

## MindSpeed

- Source: https://gitcode.com/Ascend/MindSpeed
- License: Apache License, Version 2.0
- Original copyright: Copyright (c) Huawei Technologies Co., Ltd.
- Usage: the adaptor / patching architecture and several feature-manager modules are
  adopted from MindSpeed and adapted for the HCU platform. Attributed files include:
  - `hcu_megatron/megatron_adaptor.py`
  - `hcu_megatron/patch_utils.py`
  - `hcu_megatron/features_manager/feature.py`
  - `hcu_megatron/core/tensor_parallel/checkpoint_manager.py`
- Hygon modifications: adaptation to `hcu_megatron` naming, feature registration, and
  HCU-specific optimizations.

## VocabularyParallelism

- Source: https://github.com/sail-sg/VocabularyParallelism
- License: Apache License, Version 2.0
- Original copyright: Copyright (c) Sea AI Lab (sail-sg).
- Usage: vocabulary-parallel input / output layers and the associated store utilities
  are adopted from VocabularyParallelism. Attributed files include:
  - `input_store.py`
  - `hcu_megatron/core/tensor_parallel/vocab_input.py`
  - `hcu_megatron/core/tensor_parallel/vocab_input_store.py`
  - `hcu_megatron/core/tensor_parallel/vocab_output.py`
  - `hcu_megatron/core/tensor_parallel/vocab_output_store.py`
- Hygon modifications: integration with `hcu_megatron.core.parallel_state` and the
  DualpipeV pipeline-parallel schedules.

## Seq1F1B

- Source: https://github.com/MayDomine/Seq1F1B
- License: Apache License, Version 2.0
- Usage: the sequence-parallel 1F1B pipeline schedule implementation is adopted from
  Seq1F1B. Attributed files include:
  - `hcu_megatron/core/pipeline_parallel/seq1f1b/schedules.py`
  - `hcu_megatron/core/pipeline_parallel/seq1f1b/split_solver.py`
  - `hcu_megatron/core/pipeline_parallel/seq1f1b/sp_utils.py`
- Hygon modifications: integration with Megatron-LM P2P communication primitives and
  the HCU platform.

## Primus (AMD-AGI)

- Source: https://github.com/AMD-AGI/Primus
- License: MIT License
- Original copyright: Copyright (c) Advanced Micro Devices, Inc. (AMD-AGI).
- Usage: portions of the MoE token dispatcher, router, and grouped-MLP expert
  implementations are adopted from Primus. Attributed files include:
  - `hcu_megatron/core/transformer/moe/experts.py`
  - `hcu_megatron/core/transformer/moe/router.py`
  - `hcu_megatron/core/transformer/moe/token_dispatcher.py`
- Hygon modifications: adaptation to Megatron-LM interfaces and HCU-specific tuning.

## NVIDIA TransformerEngine

- Source: https://github.com/NVIDIA/TransformerEngine
- License: Apache License, Version 2.0
- Original copyright: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
- Usage: Triton permutation kernels under
  `hcu_megatron/primus/backends/transformer_engine/pytorch/common/triton/permutation.py`
  retain the upstream TransformerEngine copyright header. The argsort helper kernels
  in that file are additionally adapted from the discussion at
  https://github.com/triton-lang/triton/issues/3698.

---

For per-file attribution, please refer to the copyright headers preserved at the top of
each source file. If you find that a third-party project used by `hcu_megatron` is
missing from this list, please open an issue so we can update the notice.

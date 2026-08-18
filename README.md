# <div align="center"><strong>hcu-megatron</strong></div>

## Introduction

`hcu-megatron` extends [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
by patching or replacing selected Megatron functions and classes to introduce new
features and deliver better performance on the Hygon HCU platform.

## Requirements

> Version dependencies: `dtk >= 25.04`, `transformer-engine >= 2.4.0`, `torch >= 2.6.0`

## Installation

1. Online (`git clone`):
   ```bash
   git clone -b <latest-branch> --recurse-submodules https://github.com/HYGON-AI/Megatron-LM-das.git
   ```
2. Offline:
   ```
   1. Download the repository release package.
   2. Click `Megatron-LM@<version>` to download the matching Megatron-LM release package.
   3. Extract the Megatron-LM package into `hcu_megatron/Megatron-LM`.
   ```

## Usage

Supports LLM pre-training, LLM SFT, and VLM SFT.

See [docs/getting-started.md](./docs/getting-started.md) for detailed usage.

## Upstream Attribution

This repository is based on the following fixed upstream baseline:

- **Upstream project**: Megatron-LM
- **Upstream repository**: https://github.com/NVIDIA/Megatron-LM.git
- **Upstream branch**: `core_r0.18.0`
- **Upstream tag**: `core_v0.18.2`
- **Upstream commit**: `571370c829ca768fe37244f4e2e7f28d8accc4ab`
- **Upstream license**: BSD-3-Clause

HCU adaptations, modifications, and original contributions by Hygon Information
Technology Co., Ltd. are licensed under the BSD-3-Clause License.

**Modified by Hygon Information Technology Co., Ltd.** Original copyright notices
and license terms from the upstream Megatron-LM project are retained.

See [LICENSE](./LICENSE) and [NOTICE](./NOTICE) for details.

## Third-Party Notices

This repository contains or references third-party source code and dependencies in
addition to the upstream Megatron-LM baseline (for example, Megatron-Bridge,
Megatron-Energon, MindSpeed, VocabularyParallelism, Seq1F1B, Primus, and NVIDIA
TransformerEngine). For a complete list of third-party components, their licenses,
attribution, and applicable notices, please refer to the formal third-party notice
file: [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).

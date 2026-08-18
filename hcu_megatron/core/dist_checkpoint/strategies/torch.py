# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import io
from pathlib import Path
from typing import Dict, List, Union, cast

import torch
from torch.distributed import checkpoint
from torch.distributed._shard.sharded_tensor import ShardedTensor as TorchShardedTensor
from torch.distributed.checkpoint import (
    BytesStorageMetadata,
    Metadata,
    TensorStorageMetadata,
)
from hyckpt_torch import FileSystemReader
from torch.distributed.checkpoint.metadata import Metadata

from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import (
    ShardedObject,
    ShardedStateDict,
    ShardedTensor,
    StateDict,
)

from megatron.core.dist_checkpointing.strategies.cached_metadata_filesystem_reader import CachedMetadataFileSystemReader
from megatron.core.dist_checkpointing.strategies.torch import (
    _replace_state_dict_keys_with_sharded_keys,
    mcore_to_pyt_state_dict,
    MCoreLoadPlanner,
    _replace_sharded_keys_with_state_dict_keys,
    _restore_dict_types,
    _unwrap_pyt_sharded_tensor
)


class TorchDistLoadShardedStrategy():
    """Basic load strategy for the PyT Distributed format."""

    def load(
        self,
        sharded_state_dict: ShardedStateDict,
        checkpoint_dir: Path,
        async_strategy: str = "mcore",
    ) -> StateDict:
        """Translates MCore ShardedTensors to PyT ShardedTensors & loads from PyT Distributed fmt.

        Args:
            sharded_state_dict (ShardedStateDict): sharded state dict with mapping
                information to instruct loading
            checkpoint_dir (Path): checkpoint directory

        Returns: loaded state dict
        """
        flexible_shape_sharded_tensors = [
            sh_ten
            for sh_ten in nested_values(sharded_state_dict)
            if isinstance(sh_ten, ShardedTensor) and not sh_ten.allow_shape_mismatch
        ]
        allow_shape_mismatch_sharded_tensors = {
            sh_ten.key: sh_ten
            for sh_ten in nested_values(sharded_state_dict)
            if isinstance(sh_ten, ShardedTensor) and sh_ten.allow_shape_mismatch
        }

        orig_sharded_state_dict = sharded_state_dict
        # MCore state dict to PyT Distributed compatible
        (sharded_state_dict, flat_mapping, rename_mapping) = (
            _replace_state_dict_keys_with_sharded_keys(sharded_state_dict)
        )
        pyt_state_dict = mcore_to_pyt_state_dict(sharded_state_dict, True)
        # Load PyT Distributed format
        fsr = CachedMetadataFileSystemReader(checkpoint_dir, cache_metadata=self.cache_metadata)
        checkpoint.load(
            pyt_state_dict,
            fsr,
            planner=MCoreLoadPlanner(
                shapes_validation_sharded_tensors=flexible_shape_sharded_tensors,
                allow_shape_mismatch_sharded_tensors=allow_shape_mismatch_sharded_tensors,
                flatten_state_dict=False,
                flatten_sharded_tensors=False,
            ),
            no_dist=True,
        )

        if self.cache_metadata:
            self.cached_global_metadata = (
                fsr.read_metadata()
            )  # no storage interaction thanks to caching

        pyt_state_dict = cast(
            Dict[str, Union[TorchShardedTensor, List[io.BytesIO]]], pyt_state_dict
        )
        # Unwrap ShardedTensors and return to original state dict
        mcore_state_dict = {
            k: v if not isinstance(v, TorchShardedTensor) else _unwrap_pyt_sharded_tensor(v)
            for k, v in pyt_state_dict.items()
        }
        mcore_state_dict = _replace_sharded_keys_with_state_dict_keys(
            mcore_state_dict, flat_mapping, rename_mapping  # type: ignore[arg-type]
        )
        _restore_dict_types(mcore_state_dict, orig_sharded_state_dict)
        return mcore_state_dict

    def load_tensors_metadata(self, checkpoint_dir: Path, metadata: Metadata = None):
        """Uses tensors metadata stored in the metadata file."""
        if metadata is None:
            fs_reader = FileSystemReader(checkpoint_dir)
            metadata = fs_reader.read_metadata()

        mcore_data = getattr(metadata, 'mcore_data', {})
        sharded_metadata = {}
        for k, tp in metadata.state_dict_metadata.items():
            if not isinstance(tp, TensorStorageMetadata):
                continue  # load only tensors

            # Regular tensor
            sharded_metadata[k] = ShardedTensor.from_rank_offsets(
                k, torch.empty(tp.size, **tp.properties.__dict__, device='meta')
            ).without_data()
        return sharded_metadata

    def load_sharded_metadata(self, checkpoint_dir: Path) -> ShardedStateDict:
        """Uses tensors and objects metadata stored in the metadata file."""
        fs_reader = FileSystemReader(checkpoint_dir)
        metadata = fs_reader.read_metadata()

        sharded_metadata = {}
        for metadata_key, storage_metadata in metadata.state_dict_metadata.items():
            if not isinstance(storage_metadata, BytesStorageMetadata):
                continue
            sh_obj = ShardedObject.empty_from_unique_key(metadata_key)
            sharded_metadata[sh_obj.unique_key] = sh_obj

        sharded_metadata.update(self.load_tensors_metadata(checkpoint_dir, metadata))
        return sharded_metadata

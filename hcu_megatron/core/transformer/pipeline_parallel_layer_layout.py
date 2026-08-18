# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import copy
from functools import lru_cache
from typing import Optional

from megatron.core import parallel_state
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout

from hcu_megatron.training.arguments import get_adaptor_args


class PipelineParallelLayerLayoutDualpipeV:
    """Configuration of custom pipeline parallel layer partitioning."""

    def __init__(self, layout: str | list, pipeline_model_parallel_size: int):
        """Initialize PipelineParallelLayerLayout from a list or a str.
        Format validation will be done here.
        """

        self.input_data = layout
        if isinstance(layout, str):
            layout = PipelineParallelLayerLayout.parse_str_to_list(layout)
        else:
            layout = copy.deepcopy(layout)
        assert all(isinstance(row, list) for row in layout), (
            f"pipeline_model_parallel_layout must be a list of lists, but got"
            f" {[type(row) for row in layout]=}"
        )

        # Check PP size and get VPP size
        assert len(layout) % pipeline_model_parallel_size == 0, (
            f"pipeline_model_parallel_layout must be divisible"
            f" by pipeline_model_parallel_size ({len(layout)=},"
            f" {pipeline_model_parallel_size=})"
        )
        assert len(layout) // pipeline_model_parallel_size == 2, (
            f"pipeline_model_parallel_layout must be equal"
            f" to 2 * pipeline_model_parallel_size ({len(layout)=},"
            f" {pipeline_model_parallel_size=})"
        )

        # Convert 1D layout to 2D layout
        layout = [
            [
                layout[pp_rank],
                layout[-pp_rank-1],
            ]
            for pp_rank in range(pipeline_model_parallel_size)
        ]

        # Convert all strings in pipeline_model_parallel_layout to LayerType
        for pp_rank in range(pipeline_model_parallel_size):
            for vpp_rank in range(2):
                transferred_layout = []
                for layer_type in layout[pp_rank][vpp_rank]:
                    assert isinstance(layer_type, LayerType) or isinstance(layer_type, str), (
                        f"elements in pipeline_model_parallel_layout must be LayerType or str,"
                        f" but got {type(layer_type)}."
                    )
                    if isinstance(layer_type, str):
                        layer_type = layer_type.strip().lower()
                        assert (
                            layer_type in LayerType.__members__
                        ), f"{layer_type} is not a valid LayerType"
                        layer_type = LayerType[layer_type]
                    transferred_layout.append(layer_type)
                layout[pp_rank][vpp_rank] = transferred_layout

        # Flatten the pipeline layout in layer id order.
        flatten_layout = []
        for row in layout:
            flatten_layout.extend(row[0])

        for row in layout[::-1]:
            flatten_layout.extend(row[1])

        # (TODO) keep virtual_pipeline_model_parallel_size, otherwise TransformerConfig.__post_init__ will raise error.
        self.virtual_pipeline_model_parallel_size = 1
        self.pipeline_model_parallel_size = pipeline_model_parallel_size
        self.layout = layout
        self.flatten_layout = flatten_layout

    def validate_layer_layout(self, num_layers: int, mtp_num_layers: int):
        """Check whether the layout is valid."""

        # Check whether the input layer id is valid
        assert all(
            isinstance(x, LayerType) for x in self.flatten_layout
        ), "All layers must be a valid LayerType."

        # Embedding layer and loss layer must be specified
        assert (
            self.flatten_layout[0] == LayerType.embedding
        ), f"The first layer must be embedding, but got {self.flatten_layout[0]}"
        assert (
            self.flatten_layout[-1] == LayerType.loss
        ), f"The last layer must be loss, but got {self.flatten_layout[-1]}"

        # Layer number verification
        assert (
            self.flatten_layout.count(LayerType.embedding) == 1
        ), "Embedding must be specified exactly once"
        assert self.flatten_layout.count(LayerType.loss) == 1, "Loss must be specified exactly once"
        assert self.flatten_layout.count(LayerType.decoder) == num_layers, (
            f"Number of decoder layers {self.flatten_layout.count(LayerType.decoder)}"
            f"must match num_layers {num_layers}"
        )
        # MTP layer verification
        assert self.flatten_layout.count(LayerType.mtp) == mtp_num_layers or (
            mtp_num_layers is None and self.flatten_layout.count(LayerType.mtp) == 0
        ), "Number of mtp layers in layout must match mtp_num_layers"
        for i in range(len(self.flatten_layout)):
            if self.flatten_layout[i] == LayerType.mtp:
                assert (
                    self.flatten_layout[i:].count(LayerType.decoder) == 0
                ), "decoder layers must be placed before MTP layers"
                break
        for pp_rank in range(self.pipeline_model_parallel_size):
            assert (
                LayerType.mtp not in self.layout[pp_rank][0]
            ), f"Currently we restrict that the MTP should be always in the second half."
        for pp_rank in range(self.pipeline_model_parallel_size):
            if LayerType.mtp in self.layout[pp_rank][-1]:
                assert (
                    self.layout[pp_rank][-1].count(LayerType.mtp) == mtp_num_layers
                ), "All of the MTP layers must be in the same one virtual pipeline stage"

        ## Detect MTP standalone usage.
        mtp_standalone = False
        for pp_rank in range(self.pipeline_model_parallel_size):
            if (
                LayerType.mtp in self.layout[pp_rank][-1]
                and pp_rank != 0
            ):
                mtp_standalone = True
                break

        # TODO: remove them in the future once they are supported
        if self.flatten_layout.count(LayerType.encoder) > 0:
            raise NotImplementedError("Encoder layer is not supported for flexible pipeline layout")

        return mtp_standalone

    def get_num_layers_to_build(
        self,
        layer_type: LayerType = LayerType.decoder,
        vp_stage: Optional[int] = None,
        pp_rank: Optional[int] = None,
    ):
        """Get the number of layers to build in the pipeline stage.
           vp_stage: 0: first half, 1: second half
        """
        if pp_rank is None:
            pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        if vp_stage is None:
            vp_stage = 1 - int(getattr(get_adaptor_args(), 'dualpipev_first_chunk', True))

        # Count layer numbers in this stage.
        num_layers_to_build = self.layout[pp_rank][vp_stage].count(layer_type)
        return num_layers_to_build

    def get_layer_offset(
        self,
        layer_type: LayerType = LayerType.decoder,
        vp_stage: Optional[int] = None,
        pp_rank: Optional[int] = None,
    ):
        """Get the layer offset in the pipeline stage"""
        if pp_rank is None:
            pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        if vp_stage is None:
            vp_stage = 1 - int(getattr(get_adaptor_args(), 'dualpipev_first_chunk', True))

        # Calculate the offset by summing up the number of
        # layers in all the previous pipeline stages.
        offset = 0
        for _pp_rank in range(pp_rank if vp_stage == 0 else self.pipeline_model_parallel_size):
            offset += self.layout[_pp_rank][0].count(layer_type)

        if vp_stage == 0:
            return offset

        for _pp_rank in range(pp_rank + 1, self.pipeline_model_parallel_size):
            offset += self.layout[_pp_rank][1].count(layer_type)

        return offset

    def get_layer_id_list(
        self,
        layer_type: LayerType = LayerType.decoder,
        vp_stage: Optional[int] = None,
        pp_rank: Optional[int] = None,
    ):
        """Get the list of layer_id for each layer in the pipeline stage."""

        if vp_stage is None:
            vp_stage = 1 - int(getattr(get_adaptor_args(), 'dualpipev_first_chunk', True))

        offset = self.get_layer_offset(layer_type=layer_type, vp_stage=vp_stage, pp_rank=pp_rank)
        num_layers_to_build = self.get_num_layers_to_build(
            layer_type=layer_type, vp_stage=vp_stage, pp_rank=pp_rank
        )
        return list(range(offset, offset + num_layers_to_build))

    def pretty_repr(self):
        """Pretty representation of the custom layout, showing the layers held by each stage.
        Example:
                            fist half                  second half
        PP rank 0           embedding,decoder*2        loss
        PP rank 1-13        decoder*2                  mtp
        PP rank 14          decoder*2                  decoder*2
        PP rank 15          decoder*2                  decoder*2
        """

        matrix = []
        header = ["", "first half", "second half"]
        matrix.append(header)

        prev_row_repr, prev_row_start_pp_rank = None, None
        for pp_rank in range(self.pipeline_model_parallel_size + 1):
            row_repr = []
            if pp_rank < self.pipeline_model_parallel_size:
                for vpp_rank in range(2):
                    stage = self.layout[pp_rank][vpp_rank]
                    stage_repr = []
                    prev_layer, prev_layer_cnt = None, 0
                    for layer_type in stage + [None]:
                        if layer_type == prev_layer:
                            prev_layer_cnt += 1
                        else:
                            if prev_layer_cnt > 1:
                                stage_repr.append(f"{prev_layer.name}*{prev_layer_cnt}")
                            elif prev_layer_cnt == 1:
                                stage_repr.append(f"{prev_layer.name}")
                            prev_layer, prev_layer_cnt = layer_type, 1
                    if len(stage_repr) == 0:
                        stage_repr.append(f"(empty stage)")
                    row_repr.append(",".join(stage_repr))

            if row_repr != prev_row_repr:
                if prev_row_start_pp_rank == pp_rank - 1:
                    matrix.append([f"PP rank {pp_rank - 1}"] + prev_row_repr)
                elif prev_row_repr is not None:
                    matrix.append(
                        [f"PP rank {prev_row_start_pp_rank}-{pp_rank - 1}"] + prev_row_repr
                    )
                prev_row_repr, prev_row_start_pp_rank = row_repr, pp_rank

        # Indent the matrix to make it more readable
        lens = [max(map(len, col)) for col in zip(*matrix)]
        indents = 8
        fmt = (" " * indents).join('{{:{}}}'.format(x) for x in lens)
        return "\n".join([fmt.format(*row) for row in matrix])

    @staticmethod
    @lru_cache()
    def from_str(layout, pipeline_model_parallel_size):
        """Parse the pipeline model parallel layout from a string."""
        parsed_layout = PipelineParallelLayerLayout(layout, pipeline_model_parallel_size)
        # Pretty print the layout distribution.
        from megatron.training import print_rank_0

        print_rank_0(
            f"Parse pipeline model parallel layout {layout} to:\n" + parsed_layout.pretty_repr(),
        )
        return parsed_layout

    @staticmethod
    def get_num_stages_from_str(layout: str):
        """Get the number of PP * VPP stages from a layout string."""
        layout_list = PipelineParallelLayerLayout.parse_str_to_list(layout)
        assert len(layout_list) % 2 == 0, (
            f"The length of pipeline_model_parallel_layout must be equal to"
            f" 2 * pipeline_model_parallel_size"
        )
        # (TODO) temporarily return pipeline_model_parallel_size, otherwise virtual_pipeline_model_parallel_size
        # will be set to 2 in validate_args
        return len(layout_list) // 2

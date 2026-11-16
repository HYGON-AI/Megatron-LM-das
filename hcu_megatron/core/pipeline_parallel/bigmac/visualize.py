# Copyright (c) 2026 Dots-Infra
# SPDX-License-Identifier: MIT
# Some of this code was adopted from https://github.com/Dots-Infra/BigMac.git

from typing import Optional, Sequence
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from .schedules import (
    PPSchedule,
    GPipe,
    Schedule1F1B,
    Sandwich,
    BigMac,
    OpType,
    VPP1F1BSchedule,
    UnifiedBigMacVPP,
)


CELL_SIZE = 0.5  # inches per time unit

COMPUTE_OPS = {
    OpType.FORWARD,
    OpType.BACKWARD,
    OpType.VEFORWARD,
    OpType.VEBACKWARD,
    OpType.GENFORWARD,
    OpType.GENBACKWARD,
}

# Backward-compatible alias for scripts that reached into this module.
compute_ops = COMPUTE_OPS

COMM_COLORS = {
    OpType.NULLCOMMUNICATION: 'gray',
    OpType.SENDFORWARD: '#CCFFFF',
    OpType.RECVFORWARD: '#CCFFFF',
    OpType.SENDBACKWARD: '#CC99CC',
    OpType.RECVBACKWARD: '#CC99CC',
    OpType.VESENDFORWARD: '#FFFFCC',
    OpType.VERECVFORWARD: '#FFFFCC',
    OpType.VESENDBACKWARD: '#FF9933',
    OpType.VERECVBACKWARD: '#FF9933',
}

BASE_COLORS = {
    OpType.FORWARD: '#64B5F6',
    OpType.BACKWARD: '#1565C0',
    OpType.VEFORWARD: '#81C784',
    OpType.VEBACKWARD: '#2E7D32',
    OpType.GENFORWARD: '#EF9A9A',
    OpType.GENBACKWARD: '#B71C1C',
    OpType.IDLE: 'white',
    **COMM_COLORS,
}

VPP_SOLID_COLORS = {
    OpType.IDLE: 'white',
    OpType.VEFORWARD: '#FFF59D',
    OpType.VEBACKWARD: '#F9A825',
    OpType.GENFORWARD: '#EF9A9A',
    OpType.GENBACKWARD: '#B71C1C',
}


def _ensure_schedule(scheduler, with_comm_op: bool = False):
    if not scheduler.schedules:
        scheduler.generate_schedule()

    if with_comm_op and not hasattr(scheduler, 'schedules_with_comm'):
        scheduler.add_communication_op()

    return scheduler.schedules_with_comm if with_comm_op else scheduler.schedules


def _max_time(schedules, pp_size: int) -> float:
    return max(
        sum(op.duration for op in schedules[rank])
        for rank in range(pp_size)
    )


def _figure_sizes(max_time: float, pp_size: int):
    fig_width = max_time * CELL_SIZE
    fig_height = pp_size * CELL_SIZE
    base_fontsize = CELL_SIZE * 30
    title_fontsize = CELL_SIZE * (max_time + pp_size)
    return fig_width, fig_height, base_fontsize, title_fontsize, base_fontsize


def _configure_axis(ax, pp_size: int, max_time: float, label_fontsize: float):
    ax.set_xlim(0, max_time)
    ax.set_ylim(0, pp_size)
    ax.set_aspect('equal', adjustable='box')
    ax.set_yticks([i + 0.5 for i in range(pp_size)])
    ax.set_yticklabels([f'Rank {i}' for i in range(pp_size)])
    ax.invert_yaxis()
    ax.set_xlabel('Time', fontsize=label_fontsize, fontweight='bold')
    ax.set_ylabel('Pipeline Rank', fontsize=label_fontsize, fontweight='bold')
    ax.tick_params(axis='both', labelsize=label_fontsize * 0.9)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')


def _operation_label(op) -> str:
    return f"{op.microbatch_id}" if op.op_type in COMPUTE_OPS else ""


def _add_schedule_patch(
    ax,
    current_time: float,
    rank: int,
    op,
    color: str,
    fontsize: float,
    text_color: str = 'black',
):
    rect = mpatches.Rectangle(
        (current_time, rank),
        op.duration,
        1.0,
        facecolor=color,
        edgecolor='black',
        linewidth=1.0,
    )
    ax.add_patch(rect)

    if op.op_type != OpType.IDLE:
        ax.text(
            current_time + op.duration / 2,
            rank + 0.5,
            _operation_label(op),
            ha='center',
            va='center',
            fontsize=fontsize,
            fontweight='bold',
            color=text_color,
        )


def _draw_rank_schedules(
    ax,
    schedules,
    pp_size: int,
    color_fn,
    fontsize: float,
    text_color_fn=None,
):
    for rank in range(pp_size):
        current_time = 0
        for op in schedules[rank]:
            text_color = text_color_fn(op) if text_color_fn else 'black'
            _add_schedule_patch(
                ax,
                current_time,
                rank,
                op,
                color_fn(op),
                fontsize,
                text_color=text_color,
            )
            current_time += op.duration


def _has_op_type(schedules, pp_size: int, op_types: Sequence[OpType]) -> bool:
    op_type_set = set(op_types)
    return any(
        op.op_type in op_type_set
        for rank in range(pp_size)
        for op in schedules[rank]
    )


def _vpp_chunk_color(op_type: OpType, chunk_id: int, vpp_size: int) -> str:
    if op_type in VPP_SOLID_COLORS:
        return VPP_SOLID_COLORS[op_type]
    if op_type in COMM_COLORS:
        return COMM_COLORS[op_type]

    intensity = chunk_id / max(vpp_size - 1, 1)
    if op_type == OpType.FORWARD:
        r = int(100 - intensity * (100 - 13))
        g = int(181 - intensity * (181 - 71))
        b = int(246 - intensity * (246 - 161))
        return f'#{r:02x}{g:02x}{b:02x}'
    if op_type == OpType.BACKWARD:
        r = int(129 - intensity * (129 - 27))
        g = int(199 - intensity * (199 - 94))
        b = int(132 - intensity * (132 - 32))
        return f'#{r:02x}{g:02x}{b:02x}'
    return 'gray'


def _save_or_show(save_path: Optional[str]):
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def visualize_schedule(
    scheduler: PPSchedule,
    save_path: Optional[str] = None,
    with_comm_op: bool = False,
):
    # Visualize pipeline schedule as a Gantt chart
    #
    # Args:
    #     scheduler: PPSchedule instance with generated schedule
    #     save_path: Optional path to save the figure
    #     with_comm_op: Optional flag to include communication operations in the visualization
    schedules = _ensure_schedule(scheduler, with_comm_op)
    max_time = _max_time(schedules, scheduler.pp_size)
    fig_width, fig_height, base_fontsize, title_fontsize, label_fontsize = _figure_sizes(
        max_time,
        scheduler.pp_size,
    )
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    _draw_rank_schedules(
        ax,
        schedules,
        scheduler.pp_size,
        color_fn=lambda op: BASE_COLORS[op.op_type],
        fontsize=base_fontsize,
    )
    _configure_axis(ax, scheduler.pp_size, max_time, label_fontsize)

    # Add title
    title = f'{scheduler.__class__.__name__} Schedule\n'
    title += (
        f'PP Size: {scheduler.pp_size}, Microbatches: {scheduler.num_microbatches}, '
        f'Total Time: {max_time:0.1f}'
    )
    ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=title_fontsize * 3)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(
            facecolor=BASE_COLORS[OpType.FORWARD],
            edgecolor=BASE_COLORS[OpType.FORWARD],
            label='LLM Forward',
        ),
        mpatches.Patch(
            facecolor=BASE_COLORS[OpType.BACKWARD],
            edgecolor=BASE_COLORS[OpType.BACKWARD],
            label='LLM Backward',
        ),
        mpatches.Patch(
            facecolor=BASE_COLORS[OpType.VEFORWARD],
            edgecolor=BASE_COLORS[OpType.VEFORWARD],
            label='VE Forward',
        ),
        mpatches.Patch(
            facecolor=BASE_COLORS[OpType.VEBACKWARD],
            edgecolor=BASE_COLORS[OpType.VEBACKWARD],
            label='VE Backward',
        ),
        mpatches.Patch(
            facecolor=BASE_COLORS[OpType.GENFORWARD],
            edgecolor=BASE_COLORS[OpType.GENFORWARD],
            label='Generator Forward',
        ),
        mpatches.Patch(
            facecolor=BASE_COLORS[OpType.GENBACKWARD],
            edgecolor=BASE_COLORS[OpType.GENBACKWARD],
            label='Generator Backward',
        ),
        mpatches.Patch(
            facecolor=BASE_COLORS[OpType.IDLE],
            edgecolor='black',
            label='Idle',
        ),
    ]
    
    if with_comm_op:
        common_legend_elements = [
            mpatches.Patch(
                facecolor=BASE_COLORS[OpType.NULLCOMMUNICATION],
                edgecolor=BASE_COLORS[OpType.NULLCOMMUNICATION],
                label='Null Communication',
            ),
            mpatches.Patch(
                facecolor=BASE_COLORS[OpType.SENDFORWARD],
                edgecolor=BASE_COLORS[OpType.SENDFORWARD],
                label='Send/Recv Forward',
            ),
            mpatches.Patch(
                facecolor=BASE_COLORS[OpType.SENDBACKWARD],
                edgecolor=BASE_COLORS[OpType.SENDBACKWARD],
                label='Send/Recv Backward',
            ),
            mpatches.Patch(
                facecolor=BASE_COLORS[OpType.VESENDFORWARD],
                edgecolor=BASE_COLORS[OpType.VESENDFORWARD],
                label='VE Send/Recv Forward',
            ),
            mpatches.Patch(
                facecolor=BASE_COLORS[OpType.VESENDBACKWARD],
                edgecolor=BASE_COLORS[OpType.VESENDBACKWARD],
                label='VE Send/Recv Backward',
            ),
        ]
        legend_elements.extend(common_legend_elements)
    
    ax.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.9 + base_fontsize * 0.00005),
        framealpha=0.9,
        ncol=5,
        frameon=False,
        fontsize=title_fontsize - 6,
    )
    
    _save_or_show(save_path)
    
    return fig, ax


def visualize_vppschedule(
    scheduler: VPP1F1BSchedule,
    save_path: Optional[str] = None,
    with_comm_op: bool = False,
):
    # Visualize VPP pipeline schedule as a Gantt chart with gradient colors based on chunk_id
    #
    # Args:
    #     scheduler: VPP1F1BSchedule instance with generated schedule
    #     save_path: Optional path to save the figure
    #     with_comm_op: Optional flag to include communication operations in the visualization
    schedules = _ensure_schedule(scheduler, with_comm_op)
    max_time = _max_time(schedules, scheduler.pp_size)
    fig_width, fig_height, base_fontsize, title_fontsize, label_fontsize = _figure_sizes(
        max_time,
        scheduler.pp_size,
    )
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    _draw_rank_schedules(
        ax,
        schedules,
        scheduler.pp_size,
        color_fn=lambda op: _vpp_chunk_color(
            op.op_type,
            getattr(op, 'chunk_id', 0),
            scheduler.vpp_size,
        ),
        fontsize=base_fontsize * 0.8,
    )
    _configure_axis(ax, scheduler.pp_size, max_time, label_fontsize)

    # Add title
    title = f'{scheduler.__class__.__name__} Schedule\n'
    title += f'PP Size: {scheduler.pp_size}, VPP Size: {scheduler.vpp_size}, '
    title += f'Microbatches: {scheduler.num_microbatches}, Total Time: {max_time:.01f}'
    ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=title_fontsize * 3)
    
    legend_elements = []
    if _has_op_type(schedules, scheduler.pp_size, [OpType.VEFORWARD, OpType.VEBACKWARD]):
        legend_elements.append(
            mpatches.Patch(
                facecolor=_vpp_chunk_color(OpType.VEFORWARD, 0, scheduler.vpp_size),
                edgecolor='black',
                label='VE Forward',
            )
        )
        legend_elements.append(
            mpatches.Patch(
                facecolor=_vpp_chunk_color(OpType.VEBACKWARD, 0, scheduler.vpp_size),
                edgecolor='black',
                label='VE Backward',
            )
        )
    
    for chunk_id in range(scheduler.vpp_size):
        legend_elements.append(
            mpatches.Patch(
                facecolor=_vpp_chunk_color(OpType.FORWARD, chunk_id, scheduler.vpp_size),
                edgecolor='black',
                label=f'F Chunk{chunk_id}',
            )
        )
    
    for chunk_id in range(scheduler.vpp_size):
        legend_elements.append(
            mpatches.Patch(
                facecolor=_vpp_chunk_color(OpType.BACKWARD, chunk_id, scheduler.vpp_size),
                edgecolor='black',
                label=f'B Chunk{chunk_id}',
            )
        )
    
    legend_elements.append(
        mpatches.Patch(facecolor='white', edgecolor='black', label='Idle')
    )

    if with_comm_op:
        comm_legend_elements = [
            mpatches.Patch(
                facecolor=COMM_COLORS[OpType.NULLCOMMUNICATION],
                edgecolor=COMM_COLORS[OpType.NULLCOMMUNICATION],
                label='Null Communication',
            ),
            mpatches.Patch(
                facecolor=COMM_COLORS[OpType.SENDFORWARD],
                edgecolor=COMM_COLORS[OpType.SENDFORWARD],
                label='Send/Recv Forward',
            ),
            mpatches.Patch(
                facecolor=COMM_COLORS[OpType.SENDBACKWARD],
                edgecolor=COMM_COLORS[OpType.SENDBACKWARD],
                label='Send/Recv Backward',
            ),
        ]
        legend_elements.extend(comm_legend_elements)
    
    ax.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.0 + base_fontsize * 0.00005),
        framealpha=0.9,
        ncol=min(scheduler.vpp_size * 2 + 1, 10),
        frameon=False,
        fontsize=title_fontsize * 0.8,
    )
    
    _save_or_show(save_path)
    
    return fig, ax


def visualize_unified_schedule(scheduler: UnifiedBigMacVPP, save_path: Optional[str] = None):
    # Visualize Unified BigMac VPP pipeline schedule as a chunk-colored Gantt chart.
    # Includes VE (Vision Encoder) and Generator modules.
    #
    # Args:
    #     scheduler: UnifiedBigMacVPP instance with generated schedule
    #     save_path: Optional path to save the figure
    schedules = _ensure_schedule(scheduler)
    max_time = _max_time(schedules, scheduler.pp_size)
    fig_width, fig_height, base_fontsize, title_fontsize, label_fontsize = _figure_sizes(
        max_time,
        scheduler.pp_size,
    )
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    _draw_rank_schedules(
        ax,
        schedules,
        scheduler.pp_size,
        color_fn=lambda op: _vpp_chunk_color(
            op.op_type,
            getattr(op, 'chunk_id', 0),
            scheduler.vpp_size,
        ),
        fontsize=base_fontsize * 0.8,
        text_color_fn=lambda op: 'white' if op.op_type == OpType.GENBACKWARD else 'black',
    )
    _configure_axis(ax, scheduler.pp_size, max_time, label_fontsize)

    # Add title
    title = f'{scheduler.__class__.__name__} Schedule\n'
    title += f'PP Size: {scheduler.pp_size}, VPP Size: {scheduler.vpp_size}, '
    title += f'Microbatches: {scheduler.num_microbatches}, Total Time: {max_time}'
    ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=title_fontsize * 3)
    
    legend_elements = []

    if _has_op_type(schedules, scheduler.pp_size, [OpType.VEFORWARD, OpType.VEBACKWARD]):
        legend_elements.append(
            mpatches.Patch(
                facecolor=_vpp_chunk_color(OpType.VEFORWARD, 0, scheduler.vpp_size),
                edgecolor='black',
                label='VE Forward',
            )
        )
        legend_elements.append(
            mpatches.Patch(
                facecolor=_vpp_chunk_color(OpType.VEBACKWARD, 0, scheduler.vpp_size),
                edgecolor='black',
                label='VE Backward',
            )
        )
    
    if _has_op_type(schedules, scheduler.pp_size, [OpType.GENFORWARD, OpType.GENBACKWARD]):
        legend_elements.append(
            mpatches.Patch(
                facecolor=_vpp_chunk_color(OpType.GENFORWARD, 0, scheduler.vpp_size),
                edgecolor='black',
                label='GEN Forward',
            )
        )
        legend_elements.append(
            mpatches.Patch(
                facecolor=_vpp_chunk_color(OpType.GENBACKWARD, 0, scheduler.vpp_size),
                edgecolor='black',
                label='GEN Backward',
            )
        )
    
    for chunk_id in range(scheduler.vpp_size):
        legend_elements.append(
            mpatches.Patch(
                facecolor=_vpp_chunk_color(OpType.FORWARD, chunk_id, scheduler.vpp_size),
                edgecolor='black',
                label=f'F Chunk{chunk_id}',
            )
        )
    
    for chunk_id in range(scheduler.vpp_size):
        legend_elements.append(
            mpatches.Patch(
                facecolor=_vpp_chunk_color(OpType.BACKWARD, chunk_id, scheduler.vpp_size),
                edgecolor='black',
                label=f'B Chunk{chunk_id}',
            )
        )
    
    legend_elements.append(
        mpatches.Patch(facecolor='white', edgecolor='black', label='Idle')
    )
    
    ax.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.0 + base_fontsize * 0.00005),
        framealpha=0.9,
        ncol=min(len(legend_elements), 10),
        frameon=False,
        fontsize=title_fontsize * 0.8,
    )
    
    _save_or_show(save_path)
    
    return fig, ax


def visualize_schedule_list(scheduler_list, save_path: Optional[str] = None):
    # Visualize multiple pipeline schedules as vertically stacked Gantt charts in one figure.
    #
    # Args:
    #     scheduler_list: List of tuples (scheduler, label) or just schedulers.
    #                     Each scheduler is a PPSchedule (or subclass) instance.
    #                     label is an optional string used as the subplot title.
    #     save_path: Optional path to save the figure
    items = []
    for i, item in enumerate(scheduler_list):
        if isinstance(item, tuple):
            scheduler, label = item
        else:
            scheduler = item
            label = f"Schedule-{i}"
        items.append((scheduler, label))

    if not items:
        raise ValueError("scheduler_list must not be empty")

    for scheduler, _ in items:
        if not scheduler.schedules:
            scheduler.generate_schedule()

    global_max_time = 0
    for scheduler, _ in items:
        for rank in range(scheduler.pp_size):
            total_time = sum(op.duration for op in scheduler.schedules[rank])
            global_max_time = max(global_max_time, total_time)

    total_rows = sum(scheduler.pp_size for scheduler, _ in items)
    num_schedulers = len(items)

    gap_rows = num_schedulers - 1
    fig_width = global_max_time * CELL_SIZE
    fig_height = (total_rows + gap_rows * 1.5) * CELL_SIZE

    base_fontsize = CELL_SIZE * 30
    label_fontsize = base_fontsize
    title_fontsize = min(CELL_SIZE * (global_max_time + total_rows), base_fontsize * 2)

    fig = plt.figure(figsize=(fig_width, fig_height))

    height_ratios = []
    for i, (scheduler, _) in enumerate(items):
        height_ratios.append(scheduler.pp_size)
        if i < num_schedulers - 1:
            height_ratios.append(0.8)

    gs = gridspec.GridSpec(
        len(height_ratios),
        1,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.05,
    )

    axes = []
    has_ve_ops_global = False

    for idx, (scheduler, label) in enumerate(items):
        gs_idx = idx * 2
        ax = fig.add_subplot(gs[gs_idx])
        axes.append(ax)

        for rank in range(scheduler.pp_size):
            for op in scheduler.schedules[rank]:
                if op.op_type in [OpType.VEFORWARD, OpType.VEBACKWARD]:
                    has_ve_ops_global = True

        for rank in range(scheduler.pp_size):
            schedule = scheduler.schedules[rank]
            current_time = 0

            for op in schedule:
                _add_schedule_patch(
                    ax,
                    current_time,
                    rank,
                    op,
                    BASE_COLORS[op.op_type],
                    base_fontsize * 0.8,
                )

                current_time += op.duration

        ax.set_xlim(0, global_max_time)
        ax.set_ylim(0, scheduler.pp_size)
        ax.set_aspect('equal', adjustable='box')

        ax.set_yticks([i + 0.5 for i in range(scheduler.pp_size)])
        ax.set_yticklabels([f'Rank {i}' for i in range(scheduler.pp_size)])
        ax.invert_yaxis()

        ax.set_ylabel(label, fontsize=label_fontsize, fontweight='bold')
        ax.tick_params(axis='both', labelsize=label_fontsize * 0.9)
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')

        if idx == num_schedulers - 1:
            ax.set_xlabel('Time', fontsize=label_fontsize, fontweight='bold')
        else:
            ax.set_xticklabels([])

    legend_elements = [
        mpatches.Patch(
            facecolor=BASE_COLORS[OpType.FORWARD],
            edgecolor='black',
            label='LLM Forward',
        ),
        mpatches.Patch(
            facecolor=BASE_COLORS[OpType.BACKWARD],
            edgecolor='black',
            label='LLM Backward',
        ),
    ]

    if has_ve_ops_global:
        legend_elements.extend([
            mpatches.Patch(
                facecolor=BASE_COLORS[OpType.VEFORWARD],
                edgecolor='black',
                label='VE Forward',
            ),
            mpatches.Patch(
                facecolor=BASE_COLORS[OpType.VEBACKWARD],
                edgecolor='black',
                label='VE Backward',
            ),
        ])

    legend_elements.append(
        mpatches.Patch(facecolor=BASE_COLORS[OpType.IDLE], edgecolor='black', label='Idle')
    )

    axes[0].legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.0 + base_fontsize * 0.00005),
        framealpha=0.9,
        ncol=min(len(legend_elements), 5),
        frameon=False,
        fontsize=title_fontsize * 0.8,
    )

    _save_or_show(save_path)

    return fig, axes


def visualize_vppschedule_list(scheduler_list, save_path: Optional[str] = None):
    # Visualize multiple VPP pipeline schedules as vertically stacked Gantt charts in one figure.
    #
    # Args:
    #     scheduler_list: List of tuples (scheduler, label) or just schedulers.
    #                     Each scheduler is a VPP1F1BSchedule (or subclass) instance.
    #                     label is an optional string used as the subplot title.
    #     save_path: Optional path to save the figure

    # Normalize input: ensure each item is (scheduler, label)
    items = []
    for i, item in enumerate(scheduler_list):
        if isinstance(item, tuple):
            scheduler, label = item
        else:
            scheduler = item
            # label = scheduler.__class__.__name__
            label = f"Schedule-{i}"
        items.append((scheduler, label))
    
    for scheduler, _ in items:
        if not scheduler.schedules:
            scheduler.generate_schedule()
    
    # Calculate global max time across all schedulers
    global_max_time = 0
    for scheduler, _ in items:
        for rank in range(scheduler.pp_size):
            total_time = sum(op.duration for op in scheduler.schedules[rank])
            global_max_time = max(global_max_time, total_time)
    
    # Calculate total rows (sum of pp_size across all schedulers)
    total_rows = sum(scheduler.pp_size for scheduler, _ in items)
    num_schedulers = len(items)
    
    # Figure dimensions
    # Add spacing between schedulers (1 row gap between each)
    gap_rows = num_schedulers - 1  # number of gaps
    fig_width = global_max_time * CELL_SIZE
    fig_height = (total_rows + gap_rows * 1.5) * CELL_SIZE
    
    # Font sizes
    base_fontsize = CELL_SIZE * 30
    label_fontsize = base_fontsize
    title_fontsize = min(CELL_SIZE * (global_max_time + total_rows), base_fontsize * 2)
    
    # Create figure with subplots sharing x-axis
    # Use gridspec for flexible layout with gaps
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Build height ratios: each scheduler gets pp_size rows, with gap rows in between
    height_ratios = []
    for i, (scheduler, _) in enumerate(items):
        height_ratios.append(scheduler.pp_size)
        if i < num_schedulers - 1:
            height_ratios.append(0.8)  # gap
    
    gs = gridspec.GridSpec(len(height_ratios), 1, figure=fig, height_ratios=height_ratios,
                           hspace=0.05)
    
    axes = []
    # Collect all vpp_sizes to build a unified legend
    max_vpp_size = max(scheduler.vpp_size for scheduler, _ in items)
    has_ve_ops_global = False
    has_gen_ops_global = False
    
    for idx, (scheduler, label) in enumerate(items):
        # Calculate gridspec index (skip gap rows)
        gs_idx = idx * 2  # 0, 2, 4, ...
        
        ax = fig.add_subplot(gs[gs_idx])
        axes.append(ax)
        
        # Check for VE/GEN ops in this scheduler
        for rank in range(scheduler.pp_size):
            for op in scheduler.schedules[rank]:
                if op.op_type in [OpType.VEFORWARD, OpType.VEBACKWARD]:
                    has_ve_ops_global = True
                if op.op_type in [OpType.GENFORWARD, OpType.GENBACKWARD]:
                    has_gen_ops_global = True
        
        # Draw schedule for each rank
        for rank in range(scheduler.pp_size):
            schedule = scheduler.schedules[rank]
            current_time = 0
            
            for op in schedule:
                _add_schedule_patch(
                    ax,
                    current_time,
                    rank,
                    op,
                    _vpp_chunk_color(op.op_type, getattr(op, 'chunk_id', 0), scheduler.vpp_size),
                    base_fontsize * 0.8,
                    text_color='white' if op.op_type == OpType.GENBACKWARD else 'black',
                )
                
                current_time += op.duration
        
        # Set axis properties
        ax.set_xlim(0, global_max_time)
        ax.set_ylim(0, scheduler.pp_size)
        ax.set_aspect('equal', adjustable='box')
        
        # Y-axis labels
        ax.set_yticks([i + 0.5 for i in range(scheduler.pp_size)])
        ax.set_yticklabels([f'Rank {i}' for i in range(scheduler.pp_size)])
        ax.invert_yaxis()
        
        # Add scheduler label on the left side as ylabel
        ax.set_ylabel(label, fontsize=label_fontsize, fontweight='bold')
        
        ax.tick_params(axis='both', labelsize=label_fontsize * 0.9)
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        
        # Only show x-axis label on the bottom subplot
        if idx == num_schedulers - 1:
            ax.set_xlabel('Time', fontsize=label_fontsize, fontweight='bold')
        else:
            ax.set_xticklabels([])
    
    # Add shared legend at the top
    legend_elements = []
    
    if has_ve_ops_global:
        legend_elements.append(
            mpatches.Patch(
                facecolor=VPP_SOLID_COLORS[OpType.VEFORWARD],
                edgecolor='black',
                label='VE Forward',
            )
        )
        legend_elements.append(
            mpatches.Patch(
                facecolor=VPP_SOLID_COLORS[OpType.VEBACKWARD],
                edgecolor='black',
                label='VE Backward',
            )
        )
    
    if has_gen_ops_global:
        legend_elements.append(
            mpatches.Patch(
                facecolor=VPP_SOLID_COLORS[OpType.GENFORWARD],
                edgecolor='black',
                label='GEN Forward',
            )
        )
        legend_elements.append(
            mpatches.Patch(
                facecolor=VPP_SOLID_COLORS[OpType.GENBACKWARD],
                edgecolor='black',
                label='GEN Backward',
            )
        )
    
    for chunk_id in range(max_vpp_size):
        legend_elements.append(
            mpatches.Patch(
                facecolor=_vpp_chunk_color(OpType.FORWARD, chunk_id, max_vpp_size),
                edgecolor='black',
                label=f'F Chunk{chunk_id}',
            )
        )
    
    for chunk_id in range(max_vpp_size):
        legend_elements.append(
            mpatches.Patch(
                facecolor=_vpp_chunk_color(OpType.BACKWARD, chunk_id, max_vpp_size),
                edgecolor='black',
                label=f'B Chunk{chunk_id}',
            )
        )
    
    legend_elements.append(
        mpatches.Patch(facecolor='white', edgecolor='black', label='Idle'))
    
    axes[0].legend(handles=legend_elements, loc='lower center',
                   bbox_to_anchor=(0.5, 1.0 + base_fontsize * 0.00005),
                   framealpha=0.9, ncol=min(len(legend_elements), 10),
                   frameon=False, fontsize=title_fontsize * 0.8)
    
    _save_or_show(save_path)
    
    return fig, axes


# Example usage

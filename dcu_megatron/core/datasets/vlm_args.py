"""VLM 数据集参数定义与数据配置解析。

本文件包含两部分：
  1. _add_dataset_extra_args(parser)  — 注册 CLI 参数
  2. parse_dataset_config(args)       — 解析 JSON 数据配置，填充 args 属性

数据配置 JSON 格式（--px-data-config-path 指向的文件）：
{
    "train_data_infos": {
        "domain_name_1": {
            "path": "/path/to/data_dir",
            "probability": 0.5,
            "sample_rate": 1.0
        },
        "domain_name_2": {
            "path": "/path/to/data_dir",
            "probability": 0.5,
            "sample_rate": 0.8
        }
    },
    "eval_data_infos": {
        "domain_name_1": {
            "path": "/path/to/eval_data_dir",
            "eval_samples_num": 100
        }
    }
}

字段说明：
  path:              数据目录路径（包含 .jsonl 文件及其索引）
  probability:       该 domain 在 global batch 中的采样占比
  sample_rate:       该 domain 的保留率（1.0 = 全量，0.5 = 随机丢弃 50%）
  eval_samples_num:  验证时该 domain 的样本数（-1 表示使用全部）
"""

import json

from dcu_megatron.core.datasets.utils import print_rank_0


# =============================================================================
# CLI 参数注册
# =============================================================================

def _add_dataset_extra_args(parser):
    """向 argparse parser 注册数据集相关 CLI 参数。

    注意：大部分参数的实际值来自 --px-data-config-path 指向的 JSON 配置文件，
    CLI 参数通常作为默认值或覆盖项。
    """

    # ── 数据配置入口（二选一） ──
    data_config_exclusive_group = parser.add_mutually_exclusive_group()
    data_config_exclusive_group.add_argument(
        "--px-data-config-path", type=str, default=None,
        help="VLM SFT 数据配置 JSON 文件路径"
    )

    group = parser.add_argument_group(title='dataset extra args')

    # ── domain 配置由 JSON 配置文件设定，不需要 CLI 参数 ──
    # px_domain_probabilities / px_train_data_domain_names / px_eval_data_domain_names
    # 均由 parse_dataset_config() 从 config JSON 的 key 自动读取并覆盖

    # ── 数据加载通用参数 ──
    group.add_argument(
        "--px-shuffle-buffer-size", type=int, default=1000000,
        help="HF datasets shuffle buffer 大小"
    )
    group.add_argument(
        "--px-dataloader-prefetch-factor", type=int, default=4,
        help="DataLoader prefetch_factor"
    )
    group.add_argument(
        '--px-top-domains-to-cut', type=int, default=1,
        help='domain 调度微调时修改前 N 个最大 domain 的配额'
    )

    # ── 验证集参数 ──
    group.add_argument(
        "--px-eval-data-path", nargs='*', default=None, type=str,
        help="验证集数据路径列表"
    )
    group.add_argument(
        "--px-eval-samples-per-domain", type=int, nargs='*', default=None,
        help="每个 domain 的 eval 样本数"
    )

    # ── Pareto 采样（可选） ──
    group.add_argument(
        "--apply-pareto-sampling", action='store_true',
        help="是否启用 Pareto 采样"
    )
    group.add_argument(
        "--px-train-apply-pareto", type=int, nargs='*', default=[],
        help="各 domain 是否启用 Pareto"
    )
    group.add_argument(
        "--px-train-pareto-alpha", type=float, nargs='*', default=[],
        help="各 domain 的 Pareto alpha"
    )
    group.add_argument(
        "--px-train-pareto-scale", type=float, nargs='*', default=[],
        help="各 domain 的 Pareto scale"
    )
    group.add_argument(
        "--train-pareto-score-scale", type=float, nargs='*', default=[],
        help="各 domain 的 Pareto score scale"
    )

    return parser


# =============================================================================
# 数据配置解析
# =============================================================================

def parse_dataset_config(args):
    """解析数据配置并填充 args 属性。

    从 --px-data-config-path 指向的 JSON 文件中读取 train/eval 配置，
    将 path / probability / sample_rate / domain_name 等填充到 args 上，
    供 build_train_valid_test_datasets() 使用。

    同时按 data_path 排序所有 domain 相关列表，确保各 domain 参数对应关系正确。
    """
    px_data_config_path = args.px_data_config_path
    if not px_data_config_path:
        raise ValueError(
            "--px-data-config-path must be specified for VLM SFT training"
        )

    with open(px_data_config_path, 'r') as f:
        data_config = json.load(f)

    # ── 解析训练集配置 ──
    train_data_path = []
    train_probability = []
    train_sample_rate = []
    train_data_domain_names = []
    train_apply_pareto = []
    train_pareto_alpha = []
    train_pareto_scale = []
    train_pareto_score_scale = []

    if "train_data_infos" not in data_config:
        raise ValueError(f"train_data_infos not found in {px_data_config_path}")

    for key, values in data_config["train_data_infos"].items():
        train_data_path.append(values["path"])
        train_probability.append(float(values["probability"]))
        train_sample_rate.append(float(values["sample_rate"]))
        train_data_domain_names.append(key)
        if args.apply_pareto_sampling:
            train_apply_pareto.append(int(values.get("apply_pareto", 0)))
            train_pareto_alpha.append(float(values.get("pareto_alpha", 9.0)))
            train_pareto_scale.append(float(values.get("pareto_scale", 1.0)))
            train_pareto_score_scale.append(
                float(values.get("pareto_score_scale", 1.0))
            )

    args.data_path = train_data_path
    args.px_domain_probabilities = train_probability
    # sample_rate → retention_rates_per_domain: 控制 domain 级别的数据保留比例
    args.px_retention_rates_per_domain = train_sample_rate
    args.px_train_data_domain_names = train_data_domain_names
    args.px_train_apply_pareto = train_apply_pareto
    args.px_train_pareto_alpha = train_pareto_alpha
    args.px_train_pareto_scale = train_pareto_scale
    args.train_pareto_score_scale = train_pareto_score_scale

    # ── 解析验证集配置 ──
    eval_data_path = []
    eval_sample_nums_per_domain = []
    eval_data_domain_names = []

    if "eval_data_infos" in data_config and args.eval_iters > 0:
        for key, values in data_config["eval_data_infos"].items():
            eval_data_path.append(values["path"])
            eval_sample_nums_per_domain.append(values.get("eval_samples_num", -1))
            eval_data_domain_names.append(key)
        args.px_eval_data_path = eval_data_path
        args.px_eval_samples_per_domain = eval_sample_nums_per_domain
        args.px_eval_data_domain_names = eval_data_domain_names

    # ── 排序：确保所有 domain 相关列表按 data_path 字典序对齐 ──
    if args.px_domain_probabilities is not None:
        args.px_domain_probabilities = [
            x for _, x in sorted(zip(args.data_path, args.px_domain_probabilities))
        ]
        args.px_retention_rates_per_domain = [
            x for _, x in sorted(zip(args.data_path, args.px_retention_rates_per_domain))
        ]
        args.px_train_data_domain_names = [
            x for _, x in sorted(zip(args.data_path, args.px_train_data_domain_names))
        ]
        if args.apply_pareto_sampling:
            args.px_train_apply_pareto = [
                x for _, x in sorted(zip(args.data_path, args.px_train_apply_pareto))
            ]
            args.px_train_pareto_alpha = [
                x for _, x in sorted(zip(args.data_path, args.px_train_pareto_alpha))
            ]
            args.px_train_pareto_scale = [
                x for _, x in sorted(zip(args.data_path, args.px_train_pareto_scale))
            ]
            args.train_pareto_score_scale = [
                x for _, x in sorted(zip(args.data_path, args.train_pareto_score_scale))
            ]
    args.data_path = sorted(args.data_path)

    # 验证 domain 数一致
    assert len(args.data_path) == len(args.px_domain_probabilities), \
        f"data_path ({len(args.data_path)}) != domain_probabilities ({len(args.px_domain_probabilities)})"

    # ── 验证集排序 ──
    if args.px_eval_data_path:
        args.px_eval_samples_per_domain = [
            x for _, x in sorted(zip(args.px_eval_data_path, args.px_eval_samples_per_domain))
        ]
        args.px_eval_data_domain_names = [
            x for _, x in sorted(zip(args.px_eval_data_path, args.px_eval_data_domain_names))
        ]
        args.px_eval_data_path = sorted(args.px_eval_data_path)

        assert len(args.px_eval_data_path) == len(args.px_eval_data_domain_names)
        assert len(args.px_eval_data_path) == len(args.px_eval_samples_per_domain)

        # 计算每个 domain 的 eval iter 数
        if all(c >= 0 for c in args.px_eval_samples_per_domain):
            eval_iters_per_domain = []
            for eval_sample_nums in args.px_eval_samples_per_domain:
                eval_iters_per_domain.append(eval_sample_nums // args.global_batch_size)
            args.px_eval_iters_per_domain = eval_iters_per_domain

    # 初始化消费进度（断点续训用）
    if not hasattr(args, "train_data_consuming_progresses"):
        args.train_data_consuming_progresses = {}

    print_rank_0(
        f"parse_dataset_config: "
        f"data_path={args.data_path} "
        f"px_domain_probabilities={args.px_domain_probabilities} "
        f"px_retention_rates_per_domain={args.px_retention_rates_per_domain} "
        f"train_data_domain_names={args.px_train_data_domain_names} "
        f"px_eval_data_path={args.px_eval_data_path} "
    )

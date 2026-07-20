"""VLM 训练参数与数据配置解析。

本文件包含两部分：
  1. add_vlm_extra_args(parser)  — 注册 VLM 训练相关 CLI 参数（模型 + 数据）
  2. parse_dataset_config(args)  — 解析 JSON 数据配置，填充 args 属性

数据配置 JSON 格式（--vlm-data-config-path 指向的文件）：
{
    "train_data_infos": {
        "domain_name_1": {
            "path": "/path/to/data_dir",
            "probability": 0.5
        }
    },
    "eval_data_infos": {
        "domain_name_1": {
            "path": "/path/to/eval_data_dir"
        }
    }
}

字段说明：
  path:         数据目录路径（包含 .jsonl 文件及其索引）
  probability:  该 domain 在 global batch 中的采样占比
"""

import json

from hcu_megatron.core.datasets.utils import print_rank_0


# =============================================================================
# CLI 参数注册
# =============================================================================

def add_vlm_extra_args(parser):
    """向 argparse parser 注册 VLM 训练相关 CLI 参数。

    包括模型架构选择、图像预处理参数、数据加载参数等。
    大部分数据 domain 参数的实际值来自 --vlm-data-config-path 指向的 JSON 配置文件，
    CLI 参数通常作为默认值或覆盖项。
    """

    # ── 模型架构 & 预处理 ──
    model_group = parser.add_argument_group(title='vlm model arguments')
    model_group.add_argument(
        "--model-arch", type=str, default="qwen2vl",
        choices=["qwen2vl", "qwen2.5vl", "qwen3vl", "gemma3vl"],
        help="model architecture, which determines the default processor and tokenizer if not specified"
    )
    model_group.add_argument("--processor-path", type=str, default=None, help="")
    model_group.add_argument("--tarfile-path", type=str, default="/", help="")
    model_group.add_argument("--min-pixels-num", type=int, default=None, help="min image width * height")
    model_group.add_argument("--max-pixels-num", type=int, default=None, help="max image width * height")
    model_group.add_argument('--spatial-merge-size', type=int, default=2, help='spatial merge size')
    model_group.add_argument("--mask-history", action='store_true', help="多轮对话只取最后一轮对话为label")

    # ── 微调冻结开关(与 Bridge provider 上的同名字段对齐,由 _bridge_apply_vlm_overrides 覆盖) ──
    model_group.add_argument("--freeze-language-model", action='store_true', default=False,
                             help="Freeze language model weights during fine-tuning")
    model_group.add_argument("--freeze-vision-model", action='store_true', default=False,
                             help="Freeze vision encoder weights during fine-tuning")
    model_group.add_argument("--freeze-vision-projection", action='store_true', default=False,
                             help="Freeze vision-to-language projection weights during fine-tuning")

    # ── 数据配置入口 ──
    parser.add_argument(
        "--vlm-data-config-path", type=str, default=None,
        help="VLM SFT 数据配置 JSON 文件路径"
    )

    # ── domain 配置由 JSON 配置文件设定，不需要 CLI 参数 ──
    # vlm_domain_probabilities / vlm_train_data_domain_names / vlm_eval_data_domain_names
    # 均由 parse_dataset_config() 从 config JSON 的 key 自动读取并覆盖

    # ── 数据加载通用参数 ──
    data_group = parser.add_argument_group(title='vlm dataset arguments')
    data_group.add_argument(
        "--vlm-shuffle-buffer-size", type=int, default=1000000,
        help="HF datasets shuffle buffer 大小"
    )
    data_group.add_argument(
        "--vlm-dataloader-prefetch-factor", type=int, default=4,
        help="DataLoader prefetch_factor"
    )
    data_group.add_argument(
        "--vlm-top-domains-to-cut", type=int, default=1,
        help="domain 调度微调时修改前 N 个最大 domain 的配额"
    )

    return parser


# =============================================================================
# 数据配置解析
# =============================================================================

def parse_dataset_config(args):
    """解析数据配置并填充 args 属性。

    从 --vlm-data-config-path 指向的 JSON 文件中读取 train/eval 配置，
    按 domain 拆分为若干个平行列表（`path` / `probability` / `domain_name`），
    挂到 args 上供 build_train_valid_test_datasets() 使用。

    JSON 的 dict 迭代顺序保证了每个 domain 的所有字段位置对齐；末尾按
    `data_path` 字典序对所有列表整体重排，得到稳定顺序（这一点在断点续训里被
    `train_data_consuming_progresses` 按 domain_id 索引时至关重要）。
    """
    vlm_data_config_path = args.vlm_data_config_path
    if not vlm_data_config_path:
        raise ValueError(
            "--vlm-data-config-path must be specified for VLM SFT training"
        )

    with open(vlm_data_config_path, 'r') as f:
        data_config = json.load(f)

    # ── 解析训练集配置 ──
    train_data_path = []
    train_probability = []
    train_data_domain_names = []

    if "train_data_infos" not in data_config:
        raise ValueError(f"train_data_infos not found in {vlm_data_config_path}")

    for key, values in data_config["train_data_infos"].items():
        train_data_path.append(values["path"])
        train_probability.append(float(values["probability"]))
        train_data_domain_names.append(key)

    args.data_path = train_data_path
    args.vlm_domain_probabilities = train_probability
    args.vlm_train_data_domain_names = train_data_domain_names

    # ── 解析验证集配置 ──
    # Always default these so downstream `if args.vlm_eval_data_path:` works even
    # when the JSON omits eval_data_infos or --eval-iters is 0.
    args.vlm_eval_data_path = None
    args.vlm_eval_data_domain_names = None
    if "eval_data_infos" in data_config and args.eval_iters > 0:
        eval_data_path = []
        eval_data_domain_names = []
        for key, values in data_config["eval_data_infos"].items():
            eval_data_path.append(values["path"])
            eval_data_domain_names.append(key)
        args.vlm_eval_data_path = eval_data_path
        args.vlm_eval_data_domain_names = eval_data_domain_names

    # ── 按 data_path 字典序整体重排（domain_id 稳定，断点续训需要）──
    order = sorted(range(len(args.data_path)), key=lambda i: args.data_path[i])
    args.data_path = [args.data_path[i] for i in order]
    args.vlm_domain_probabilities = [args.vlm_domain_probabilities[i] for i in order]
    args.vlm_train_data_domain_names = [args.vlm_train_data_domain_names[i] for i in order]

    if args.vlm_eval_data_path:
        eval_order = sorted(
            range(len(args.vlm_eval_data_path)), key=lambda i: args.vlm_eval_data_path[i]
        )
        args.vlm_eval_data_path = [args.vlm_eval_data_path[i] for i in eval_order]
        args.vlm_eval_data_domain_names = [args.vlm_eval_data_domain_names[i] for i in eval_order]

    # 初始化消费进度（断点续训用）
    if not hasattr(args, "train_data_consuming_progresses"):
        args.train_data_consuming_progresses = {}

    print_rank_0(
        f"parse_dataset_config: "
        f"data_path={args.data_path} "
        f"vlm_domain_probabilities={args.vlm_domain_probabilities} "
        f"train_data_domain_names={args.vlm_train_data_domain_names} "
        f"vlm_eval_data_path={args.vlm_eval_data_path} "
    )

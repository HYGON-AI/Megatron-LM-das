# Megatron-LM-das CI 说明

CI 按 verl-das 的边界拆为两条独立流水线：PR 只执行单元测试，Nightly 只执行
定时或手动发起的真实训练验证。两条流水线都使用固定 digest 的 HCU 容器。

## 流水线

| Workflow | 触发 | 测试范围 |
|---|---|---|
| `PR Test (HCU)` | `pull_request_target`、手动触发 | 完整 `tests/unit_tests` |
| `Nightly Test (HCU)` | 每天 19:00 UTC、手动触发 | Qwen3-8B pretrain 10 steps、Qwen3-8B SFT 10 steps |

Nightly 的手动入口支持 `all`、`pretrain`、`sft`。pretrain 与 SFT 是两个独立 job，
分别显示结果、上传日志，并在每个训练阶段后恢复 runner 工作区属主。即使 pretrain
失败，属主恢复成功后仍会继续运行 SFT，最终由 `Finish` 汇总两项结论。

## Qwen3-8B 训练入口

Nightly 不维护另一套模型参数脚本，直接运行仓库已有的
`examples/qwen3/train_qwen3_8B.sh`。该 example 接受以下环境变量：

| 变量 | Nightly 值 | 说明 |
|---|---|---|
| `TRAINING_MODE` | `pretrain` / `sft` | 选择 indexed dataset 或 SFT JSONL 参数 |
| `TRAIN_ITERS` | `10` | 训练步数 |
| `DATA_PATH` | 仓库变量注入 | pretrain dataset prefix 或 SFT 目录 |
| `TOKENIZER_MODEL_PATH` | 仓库变量注入 | Qwen3-8B Hugging Face 模型目录 |
| `LAUNCH_BACKEND` | `torchrun` | 单节点 8 卡启动方式 |

example 通过 Megatron Bridge 加载 Hugging Face 权重；pretrain 使用 `.bin/.idx`
dataset prefix，SFT 使用目录下的 `train.jsonl` 和 `valid.jsonl`。

## 仓库变量

在 Settings → Secrets and variables → Actions → Variables 中配置：

| 变量 | PR | Nightly | 说明 |
|---|---|---|---|
| `DAS_HCU_CI_RUNNER_LABEL` | 必填 | 必填 | 专用 HCU runner 标签 |
| `DAS_HCU_CI_IMAGE` | 必填 | 必填 | 训练镜像滚动 tag，形如 `<harbor>/megatron:0.18.2-latest` |
| `DAS_HCU_ASSET_ROOT` | - | 必填 | 模型和数据共同根目录，只读挂载 |
| `DAS_QWEN3_8B_MODEL_PATH` | - | 必填 | Qwen3-8B Hugging Face 模型绝对路径 |
| `DAS_QWEN3_PRETRAIN_DATA_PATH` | - | pretrain 必填 | Megatron indexed dataset 绝对前缀 |
| `DAS_QWEN3_SFT_DATA_PATH` | - | SFT 必填 | 含 `train.jsonl`、`valid.jsonl` 的绝对目录 |
| `DAS_HCU_MEGATRON_WHEEL` | 可选 | 可选 | hcu-megatron wheel 路径或 URL |

模型与数据路径必须位于 `DAS_HCU_ASSET_ROOT` 下。资产根以只读 volume 挂载，训练
日志、TensorBoard 与临时文件写入 CI 自有目录，不写回模型或数据目录。

## 依赖与子模块

PR 单测、Nightly pretrain、Nightly SFT 各自在容器中执行：

```bash
python3 -m pip install -r requirements/requirements.txt
```

随后 `tests/das/ci/prepare_workspace.sh` 初始化并核对固定的 Megatron-LM、Energon、
Bridge 子模块，再设置 `PYTHONPATH`。PR 将 `BUCKET=tests/unit_tests` 传给现有
`tests/unit_tests/run_ci_test.sh`，因此该目录新增的测试会自动纳入。

## 安全模型

1. PR 使用 `pull_request_target`，workflow 控制逻辑来自目标分支；checkout 显式使用
   PR head repository 与 head SHA。
2. PR 的控制面 job 运行在 GitHub-hosted runner；测试和属主恢复使用专用、隔离的
   self-hosted HCU runner。
3. workflow 权限为只读，不向 PR 测试注入仓库 secrets。
4. Nightly 只运行目标分支代码，不接受 PR head，并使用只读真实资产。
5. 每个容器测试阶段后都恢复工作区属主，避免持久 runner 的后续 checkout 失败。

## 镜像

`.github/workflows/docker-image.yml` 负责构建并推送训练镜像，每次构建同时推送带时间戳
的 tag 和滚动 tag `0.18.2-latest`。

测试 workflow 通过仓库变量 `DAS_HCU_CI_IMAGE` 消费该滚动 tag，不再需要在每次构建后手工
更新 digest。

HCU runner 主机上有一个每天 01:30 触发的 systemd user timer
(`docker-pull-megatron.timer`)，负责把滚动 tag 拉到本地，并只保留最近 3 个构建以控制磁盘
占用。构建 workflow 在 00:40 前后完成推送，因此拉取时取到的即为当日镜像。

若需临时回退到某个历史构建，把 `DAS_HCU_CI_IMAGE` 改成对应的时间戳 tag 即可。

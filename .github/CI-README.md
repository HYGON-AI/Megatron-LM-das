# Megatron-LM-das CI 说明

CI 使用单主流水线、配置决策、测试执行和聚合门四层结构，提供 PR 单元测试与
Nightly Qwen3-8B 真实训练验证。测试环境由固定 digest 的 HCU 容器提供。

## 触发与测试范围

| 触发 | 测试范围 |
|---|---|
| `pull_request_target`（目标分支 `core_v0.18.2`） | 完整运行仓库 `tests/unit_tests` |
| `schedule`（每天 19:00 UTC） | 完整单元测试 + Qwen3-8B 真实数据预训练 + Qwen3-8B 真实数据 SFT |
| `workflow_dispatch` | `pr` / `unit-only` 只运行单元测试；`nightly` 运行完整 Nightly |

Job 链为：`authorize` → `check-changes` → `configure` →
`linting` / `validate-config` → `restore-before-unit` → `unit-tests` →
`restore-after-unit` → `nightly-qwen3-8b` → `restore-after-nightly` → `finish`。

- PR 不运行训练任务，只运行 `tests/unit_tests`。
- Nightly 在同一个隔离 runner 上顺序运行预训练和 SFT，两个测试分别上传日志。
- 模型和数据目录只读挂载；训练不向资产目录写 checkpoint 或日志。
- Nightly 通过 Megatron Bridge 从 Hugging Face 格式目录加载 Qwen3-8B 权重。

## 安全模型

1. PR 使用 `pull_request_target`，workflow 控制逻辑来自目标分支；checkout 显式使用
   PR head repository 和 head SHA。
2. 控制面 job 运行在 GitHub-hosted `ubuntu-latest`；只有测试和属主恢复 job 使用
   专用 self-hosted HCU runner。
3. 测试 token 仅有只读权限，不向 PR 测试 job 注入仓库 secrets。
4. PR 代码会进入带设备和 `--privileged` 的容器，因此 runner 必须专用、隔离且
   不保存生产凭据。
5. `pull_request_target` 只执行目标分支已有的 workflow；本 PR 中的新逻辑需要合入
   后才能通过该事件验证。

## 仓库变量

Settings → Secrets and variables → Actions → **Variables**：

| 变量 | 必填 | 说明 |
|---|---|---|
| `DAS_HCU_CI_RUNNER_LABEL` | 是 | 专用 HCU runner 标签 |
| `DAS_HCU_CI_IMAGE` | 是 | 测试镜像，必须使用 `仓库@sha256:<64 位十六进制>` 固定 digest |
| `DAS_HCU_ASSET_ROOT` | Nightly 必填 | runner 上同时包含 Qwen3-8B 模型与真实数据集的绝对路径；以只读方式挂载到容器 |
| `DAS_QWEN3_8B_MODEL_PATH` | 可选 | Qwen3-8B Hugging Face 模型目录；未填时在资产根目录内按 `config.json` 唯一识别 |
| `DAS_QWEN3_PRETRAIN_DATA_PATH` | 可选 | Megatron indexed dataset 前缀；未填时在资产根目录内按配对 `.bin/.idx` 唯一识别 |
| `DAS_QWEN3_SFT_DATA_PATH` | 可选 | 同时包含 `train.jsonl` 和 `valid.jsonl` 的目录；未填时在资产根目录内唯一识别 |
| `DAS_HCU_MEGATRON_WHEEL` | 可选 | hcu-megatron 预编译 wheel 路径或 URL |

如果自动识别得到零个或多个候选，Nightly 会明确失败并列出候选；管理员应填写对应
精确路径变量，避免测试静默选择错误的模型或数据集。

## Nightly 参数

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `DAS_QWEN3_TRAIN_ITERS` | 5 | 预训练和 SFT 各自执行的迭代数 |
| `DAS_QWEN3_SEQ_LENGTH` | 1024 | 序列长度 |
| `DAS_QWEN3_TP` | 2 | Tensor Parallel 大小 |
| `DAS_QWEN3_PP` | 2 | Pipeline Parallel 大小 |
| `DAS_QWEN3_MICRO_BATCH_SIZE` | 1 | micro batch size |
| `DAS_QWEN3_GLOBAL_BATCH_SIZE` | 8 | global batch size |
| `DAS_HCU_CI_TMP_ROOT` | `/tmp/das-hcu-ci` | CI 临时目录根 |

## 单元测试与子模块

PR 和 Nightly 都将 `BUCKET=tests/unit_tests` 传给现有
`tests/unit_tests/run_ci_test.sh`，因此目标分支后续新增到该目录的测试也会自动纳入，
无需维护静态 bucket 列表。

`tests/das/ci/verify_submodules.py` 固定 Megatron-LM、Energon 和 Bridge 三个
submodule 的 gitlink SHA，并在测试前核对 gitlink 与 checkout。升级 submodule 时需
同步更新 `EXPECTED_SUBMODULES`。

测试依赖固定为 `pytest-mock==3.14.0` 和 `coverage==7.6.1`。覆盖率使用
`--parallel-mode`，各 worker 独立写文件后再合并。

## 与官方 CI 架构的对应关系

| 官方结构 | 本仓库实现 |
|---|---|
| pre-flight / configure | `authorize`、`check-changes`、`configure`、`validate-config` |
| lint | GitHub-hosted `linting` |
| unit test matrix / recipe | 完整 `tests/unit_tests` 目录，由现有 `run_ci_test.sh` 启动 |
| nightly integration | Qwen3-8B 真实权重预训练和 SFT |
| aggregation gate | `finish` 汇总并校验所有必需 job 结论 |

本仓库将镜像构建与测试解耦，并使用专用 HCU runner；这是运行环境差异，不改变
“控制面决策 → 测试执行 → 聚合门”的总体结构。

## CI 镜像

`.github/workflows/docker-image.yml` 负责构建并推送训练镜像。测试工作流不直接消费
可变 tag；镜像构建成功后，应验证新镜像并由管理员将完整 digest 更新到
`DAS_HCU_CI_IMAGE`。

## 常见问题

- **PR 没有触发新逻辑**：确认 workflow 已进入目标分支。
- **Nightly 报资产候选不唯一**：填写三个 Qwen3 精确路径变量中的对应项。
- **GPU job 一直排队**：检查专用 runner 是否在线且标签与变量一致。
- **runner 工作区属主异常**：前置和后置恢复 job 会在安全路径检查后修复属主。
- **镜像漂移**：测试变量必须固定 digest，不能使用 `latest` 或普通 tag。

## 注册 runner

```bash
gh api -X POST repos/HYGON-AI/Megatron-LM-das/actions/runners/registration-token \
  -q .token

mkdir -p ~/actions-runner-das && cd ~/actions-runner-das
curl -o actions-runner.tar.gz -L https://github.com/actions/runner/releases/download/v2.327.0/actions-runner-linux-x64-2.327.0.tar.gz
tar xzf actions-runner.tar.gz
./config.sh --url https://github.com/HYGON-AI/Megatron-LM-das \
  --token <TOKEN> \
  --name nmz36-hygon-hcu-megatron \
  --labels self-hosted,Linux,X64,hcu,bw1100,hcu-ci-pr,nmz36 \
  --unattended --replace
nohup ./run.sh > runner.log 2>&1 &
```

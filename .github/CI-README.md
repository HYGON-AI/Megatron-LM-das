# Megatron-LM-das CI 说明

CI 框架对齐 [NVIDIA/Megatron-LM 官方](https://github.com/NVIDIA/Megatron-LM) 的
GitHub Actions 架构(单主流水线 + 配置决策中心 + 测试矩阵 + 聚合门),运行细节
遵循 HYGON verl-das 既有惯例(HCU 容器化、同仓库授权、变更分类、工作区恢复、
聚合总结)。v1 提供 **PR 级测试** 与 **Nightly 测试** 两个基础档次。

## 流水线结构(`.github/workflows/cicd-main.yml`)

| 触发 | 行为 |
|---|---|
| `pull_request`(core_v0.18.2) | 控制面检查自动跑;GPU 测试(L0 单元)需人工审批后执行;打 label `Run functional tests` → L1(+功能冒烟) |
| `schedule`(每天 19:00 UTC) | Nightly: 单元测试 ×2 + 功能冒烟 20 iter(不经过审批门) |
| `workflow_dispatch` | `test_case`: `pr` / `nightly` / `unit-only` 手动选择 |

Job 链: `authorize`(fork 判定)→ `check-changes`(docs-only 白名单)→ `configure`
(L0/L1 决策)→ `linting` / `validate-config` → `gpu-gate`(PR 人工审批门)→
`unit-tests`(矩阵 2 bucket, 8 卡分布式 pytest)→ `restore-after-unit` →
`functional-smoke`(合成数据 + 极小模型 pretrain)→ `restore-after-smoke` →
`finish`(聚合门, 输出总结)。

- **控制面 job**(authorize/check-changes/configure/lint/validate/finish/gpu-gate)
  全部运行在 GitHub-hosted `ubuntu-latest`,不依赖 self-hosted runner;
- **GPU 测试 job**(unit-tests/functional-smoke/restore-\*)使用
  `${{ vars.DAS_HCU_CI_RUNNER_LABEL }}` 注入的 self-hosted HCU runner。

## 安全模型与部署必需项 ⚠

1. **environment 审批(必须配置)**: 管理员必须在仓库
   Settings → Environments 为 `das-hcu-gpu` 配置 **required reviewers**。
   未配置时 GitHub 对无保护环境默认放行,PR 的 GPU 测试将不经审批直接执行。
   这是 PR 触发 GPU 测试唯一的审批闸门。
2. **fork PR**: `authorize` 拒绝 fork PR 的 GPU 测试(仅跑控制面);
   fork PR 的 secrets 天然不可见。
3. **控制面与 GPU 分离**: PR head 代码只在 GPU 测试 job 中执行,且必须通过
   审批门;控制面永远跑在 GitHub-hosted runner 上。
4. **权限基线**: workflow `permissions` 为只读(contents/ issues)。

## 仓库配置(必须填充)

Settings → Secrets and variables → Actions → **Variables**:

| 变量 | 必填 | 说明 |
|---|---|---|
| `DAS_HCU_CI_RUNNER_LABEL` | ✅ | self-hosted runner 标签(如 `hcu-ci`),未配置时 validate-config 明确失败并给出指引 |
| `DAS_HCU_CI_IMAGE` | ✅ | 测试容器镜像(HCU 环境),**必须用 `@sha256:<64 位十六进制>` 固定 digest**(与 verl-das 一致, 防镜像漂移);未配置/未固定时 validate-config 明确失败 |
| `DAS_HCU_MEGATRON_WHEEL` | 可选 | hcu-megatron 预编译 wheel 路径/URL,已注入全部测试 job;prepare_workspace 自动 `pip install` |
| `DAS_HCU_MODEL_ROOT` | 预留 | 模型根目录;**接入时需取消注释 workflow 中 functional-smoke 的挂载行**(YAML 无法条件化 volume) |
| `DAS_HCU_DATA_ROOT` | 预留 | 数据根目录,同上 |

**子模块版本 pin**: `tests/das/ci/verify_submodules.py` 固定三个子模块
(Megatron-LM / Energon / Bridge)的 gitlink SHA, 在每次测试前做 gitlink +
checkout 双重核对;升级子模块时需同步更新该文件的 `EXPECTED_SUBMODULES`。

## 测试脚本接口(环境变量, 均有默认值)

| 变量 | 默认 | 说明 |
|---|---|---|
| `DAS_SMOKE_TRAIN_ITERS` | 5(L1)/20(nightly) | 功能冒烟训练迭代数 |
| `DAS_TRANSFORMER_IMPL` | `transformer_engine` | 功能冒烟 transformer 实现(与生产一致; `local` 分支当前有 gpt_builders.py 参数错位问题) |
| `DAS_TOY_VOCAB_SIZE` | 256 | 合成数据 vocab 大小(eod = vocab_size - 1) |
| `DAS_TOY_NUM_DOCS` | 2000 | 合成数据文档数 |
| `DAS_HCU_CI_TMP_ROOT` | `/tmp/das-hcu-ci` | 运行期临时目录根 |
| `DAS_COVERAGE_DISABLED` | 0 | 镜像缺 coverage 时自动置 1, 降级纯 pytest |
| `DAS_HCU_MEGATRON_WHEEL` | 未配置 | wheel 路径/URL,提供 `fused_weight_gradient_mlp_cuda` 等编译算子;未配置时 sitecustomize 注入 import 垫片(调用即报 NotImplementedError) |
| `MEGATRON_PATH` / `DTK_ENV` / `CONDA_HOME` | 见脚本 | `run_ci_test.sh` 环境变量化占位符 |

测试依赖版本锁定: `pytest-mock==3.14.0` / `coverage==7.6.1`(运行时按需安装 +
Dockerfile 固化, 双路径一致)。覆盖率采集使用 `coverage run --parallel-mode`,
8 个 worker 各写独立文件, 最后 `coverage combine` 合并。

## 与官方(Megatron-LM)启动方式对齐

| 维度 | 官方 cicd-main.yml | 本仓库 cicd-main.yml | 说明 |
|---|---|---|---|
| 流水线骨架 | pre-flight → configure → lint → 容器构建 → unit/integration 矩阵 → 聚合门 | authorize → check-changes → configure → lint → validate → gpu-gate → unit/smoke → finish | 决策中心 + 测试矩阵 + 聚合门三段式一致 |
| 单元测试启动 | `run_ci_test.sh` 内 `python -m torch.distributed.run`(8 卡) | 同左(原样保留 das 自己的 run_ci_test.sh) | 完全一致 |
| 训练启动 | recipe 脚本 → pretrain_xxx.py | `LAUNCH_BACKEND=torchrun python3 -m torch.distributed.run pretrain_gpt.py` | das 仓库自身约定(见 run_qwen.sh);torchrun 启动必须带 `LAUNCH_BACKEND=torchrun`,否则 parse_args 会用 `--rank` 默认值 -1 覆写 RANK |
| PR 触发 | push `pull-request/*` 分支(需镜像 bot)+ merge_group | 普通 `pull_request` + environment 人工审批门 | 无 PR 镜像 bot 基础设施;fork PR 拒绝 GPU |
| Nightly | schedule 0 0 \* \* \* | schedule 0 19 \* \* \* | 时间可改 |
| 容器 | CI 内动态 build(Dockerfile.ci.dev/lts) | 预置镜像,经 `DAS_HCU_CI_IMAGE` 注入(预留 Dockerfile.ci.hcu) | 用户填充镜像变量即可 |
| runner | 动态矩阵(aws-h100/gb200) | 单标签 `${{ vars.DAS_HCU_CI_RUNNER_LABEL }}` | 多机扩展时可仿官方加 matrix |
| 测试选择 | recipe parser + scope/cadence | L0/L1 label + workflow_dispatch 选择 | v1 简化,后续可对齐 recipe 体系 |

## 常见问题

- **缺镜像/缺 runner**: 控制面会先跑完并给出 `validate-config` 错误指引;
  GPU job 找不到 runner 时排队等待(属正常现象, 注册 runner 后自动续跑)。
- **PR 的 GPU 测试一直等待**: 检查是否已配置 `das-hcu-gpu` environment 的
  required reviewers,并在 PR 的 workflow run 页面完成审批。
- **runner 工作区属主**: 容器 job 以 root 写入, 由 `restore-after-*` job 自动
  chown 恢复;子模块 `Megatron-LM/tests` 改名由 `trap` 保证成功/失败都恢复。
- **与 verl-das CI 并发**: 与 verl-das 共用机器时通过不同 label 隔离;
  并发高峰可能排队, 后续可加互斥。
- **分支**: 仓库默认分支 `core_v0.18.2`, workflow 与 schedule 均挂在该分支。

## 注册 runner(一次性, 需仓库 admin)

```bash
# 获取注册 token(nmz4, 需 repo admin)
gh api -X POST repos/HYGON-AI/Megatron-LM-das/actions/runners/registration-token \
  -q .token

# 在 bw18 上注册(标签含 self-hosted + hcu + 与 DAS_HCU_CI_RUNNER_LABEL 一致的自定义标签)
mkdir -p ~/actions-runner-das && cd ~/actions-runner-das
curl -o actions-runner.tar.gz -L https://github.com/actions/runner/releases/download/v2.327.0/actions-runner-linux-x64-2.327.0.tar.gz
tar xzf actions-runner.tar.gz
./config.sh --url https://github.com/HYGON-AI/Megatron-LM-das \
  --token <TOKEN> \
  --name nmz18-hygon-hcu-megatron-lm-das \
  --labels self-hosted,Linux,X64,hcu,hcu-ci \
  --unattended --replace
nohup ./run.sh > runner.log 2>&1 &
```

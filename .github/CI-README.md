# Megatron-LM-das CI 说明

CI 框架对齐 [NVIDIA/Megatron-LM 官方](https://github.com/NVIDIA/Megatron-LM) 的
GitHub Actions 架构(单主流水线 + 配置决策中心 + 测试矩阵 + 聚合门),运行细节
遵循 HYGON verl-das 既有惯例(HCU 容器化、精确 PR head 检出、变更分类、
工作区恢复、聚合总结)。v1 提供 **PR 级测试** 与 **Nightly 测试** 两个基础档次。

## 流水线结构(`.github/workflows/pr-test-hcu.yml`)

| 触发 | 行为 |
|---|---|
| `pull_request_target`(core_v0.18.2) | 从默认分支加载受控 workflow,精确检出 PR head;同仓库与 fork PR 均执行 L0 单元测试;打 label `Run functional tests` → L1(+功能冒烟) |
| `schedule`(每天 19:00 UTC) | Nightly: 单元测试 ×2 + 功能冒烟 20 iter(不经过审批门) |
| `workflow_dispatch` | `test_case`: `pr` / `nightly` / `unit-only` 手动选择 |

Job 链: `authorize`(记录 PR 来源)→ `check-changes`(docs-only 白名单)→ `configure`
(L0/L1 决策)→ `linting` / `validate-config` →
`unit-tests`(矩阵 2 bucket, 8 卡分布式 pytest)→ `restore-after-unit` →
`functional-smoke`(合成数据 + 极小模型 pretrain)→ `restore-after-smoke` →
`finish`(聚合门, 输出总结)。

- **控制面 job**(authorize/check-changes/configure/lint/validate/finish)
  全部运行在 GitHub-hosted `ubuntu-latest`,不依赖 self-hosted runner;
- **GPU 测试 job**(unit-tests/functional-smoke/restore-\*)使用
  `${{ vars.DAS_HCU_CI_RUNNER_LABEL }}` 注入的 self-hosted HCU runner。

## 安全模型与部署必需项 ⚠

1. **默认分支控制面**: PR 使用 `pull_request_target`,授权和调度逻辑始终来自
   默认分支;每个 checkout 都显式使用
   `pull_request.head.repo.full_name` + `pull_request.head.sha`,不执行模糊分支引用。
2. **fork PR 开放**: 与 verl-das `e1d01629` 一致,同仓库和 fork PR 都可执行
   HCU 测试。由于 PR 代码会进入带设备和 `--privileged` 的容器,runner 必须是
   **专用、隔离、无生产凭据**的 CI 机器,不能与通用或生产任务共用。
3. **权限基线**: 测试 workflow 的 token 仅有 `contents: read` / `issues: read`,
   不向 PR 测试 job 注入仓库 secrets。
4. **bootstrap 限制**: `pull_request_target` 只执行默认分支上的 workflow。
   因此本框架需先经审核合入,后续 PR 才能验证新触发逻辑;当前 PR 不能自证
   尚未合入的 workflow 修改。

## 仓库配置(必须填充)

Settings → Secrets and variables → Actions → **Variables**:

| 变量 | 必填 | 说明 |
|---|---|---|
| `DAS_HCU_CI_RUNNER_LABEL` | ✅ | self-hosted runner 标签(如 `hcu-ci`),未配置时 validate-config 明确失败并给出指引 |
| `DAS_HCU_CI_IMAGE` | ✅ | `.github/workflows/docker-image.yml` 推送的 Megatron 训练镜像;必须转换为 `镜像仓库@sha256:<64 位十六进制>` 固定 digest(与 verl-das 一致,防镜像漂移);未配置/未固定时 validate-config 明确失败 |
| `DAS_HCU_MEGATRON_WHEEL` | 可选 | hcu-megatron 预编译 wheel 路径/URL,已注入全部测试 job;prepare_workspace 自动 `pip install` |
| `DAS_HCU_MODEL_ROOT` | 预留 | 模型根目录;**接入时需取消注释 workflow 中 functional-smoke 的挂载行**(YAML 无法条件化 volume) |
| `DAS_HCU_DATA_ROOT` | 预留 | 数据根目录,同上 |

**子模块版本 pin**: `tests/das/ci/verify_submodules.py` 固定三个子模块
(Megatron-LM / Energon / Bridge)的 gitlink SHA, 在每次测试前做 gitlink +
checkout 双重核对;升级子模块时需同步更新该文件的 `EXPECTED_SUBMODULES`。

## CI 镜像 workflow

默认分支 `dd8740c4` 已提供 `.github/workflows/docker-image.yml` 与
`docker/Dockerfile`,workflow 会构建并推送时间戳 tag 到其配置的 Harbor 仓库。
测试主流水线不直接消费可变 tag:镜像推送成功后应解析 registry 返回的 digest,
再由管理员把完整的 `仓库@sha256:...` 写入 `DAS_HCU_CI_IMAGE`。

镜像构建与测试流水线目前没有自动跨 workflow 传递 digest;这是有意保留的发布
边界,可避免 PR 测试静默切换到刚生成、尚未审核的镜像。

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

| 维度 | 官方 cicd-main.yml | 本仓库 pr-test-hcu.yml | 说明 |
|---|---|---|---|
| 流水线骨架 | pre-flight → configure → lint → 容器构建 → unit/integration 矩阵 → 聚合门 | authorize → check-changes → configure → lint → validate → unit/smoke → finish | 决策中心 + 测试矩阵 + 聚合门三段式一致 |
| 单元测试启动 | `run_ci_test.sh` 内 `python -m torch.distributed.run`(8 卡) | 同左(原样保留 das 自己的 run_ci_test.sh) | 完全一致 |
| 训练启动 | recipe 脚本 → pretrain_xxx.py | `LAUNCH_BACKEND=torchrun python3 -m torch.distributed.run pretrain_gpt.py` | das 仓库自身约定(见 run_qwen.sh);torchrun 启动必须带 `LAUNCH_BACKEND=torchrun`,否则 parse_args 会用 `--rank` 默认值 -1 覆写 RANK |
| PR 触发 | push `pull-request/*` 分支(需镜像 bot)+ merge_group | `pull_request_target` + 精确 head repo/SHA | 触发与 fork 授权对齐 verl-das;隔离 runner 是前提 |
| Nightly | schedule 0 0 \* \* \* | schedule 0 19 \* \* \* | 时间可改 |
| 容器 | CI 内动态 build(Dockerfile.ci.dev/lts) | `docker-image.yml` 构建,经 `DAS_HCU_CI_IMAGE` 注入固定 digest | 镜像发布与测试解耦 |
| runner | 动态矩阵(aws-h100/gb200) | 单标签 `${{ vars.DAS_HCU_CI_RUNNER_LABEL }}` | 多机扩展时可仿官方加 matrix |
| 测试选择 | recipe parser + scope/cadence | L0/L1 label + workflow_dispatch 选择 | v1 简化,后续可对齐 recipe 体系 |

## 常见问题

- **缺镜像/缺 runner**: 控制面会先跑完并给出 `validate-config` 错误指引;
  GPU job 找不到 runner 时排队等待(属正常现象, 注册 runner 后自动续跑)。
- **新 PR 没有触发新逻辑**: 检查 `pr-test-hcu.yml` 是否已经合入默认分支;
  `pull_request_target` 不会从 PR head 加载 workflow。
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

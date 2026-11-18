# UltraEP

## 简介

UltraEP 是基于冗余专家副本的 MoE 在线专家并行负载均衡方案。

在 MoE 训练里，路由器把 token 分给各个 expert 时经常出现热点：少数专家收到远多于平均值的 token，其所在 EP rank 通信/计算成为整个 all-to-all 的瓶颈。UltraEP 的思路是**在每个 EP rank 上多放 N 个「副本专家」（replica experts）**，路由时通过 `ultra_ep` C++ 运行时把热点逻辑专家 round-robin 到多个物理副本，从而把该逻辑专家承担的 token 分摊到多个 rank 上。

副本权重的存放、副本↔master 的权重同步、副本梯度的归约都由 `ultra_ep` C++ 运行时通过共享 GPU buffer 处理，与 Megatron 的 DDP / DistributedOptimizer 通过 autograd Function 时序衔接，对上层训练逻辑透明。

## 实现内容

### 新增

| 路径 | 说明 |
|------|------|
| `hcu_megatron/core/transformer/moe/eplb_manager.py` | 封装 `ultra_ep.Manager`，管理专家布局、虚拟 layer 槽、步边界启停和副本通信的跨 rank 完成依赖 |
| `hcu_megatron/features_manager/moe/ultraep_feature.py` | `UltraEPFeature`：CLI 参数注册、参数校验、patch 注册 |
| `hcu_megatron/training/checkpointing.py` | native torch 格式 checkpoint 过滤 replica 专家的 wrapper |
| `hcu_megatron/training/ultraep_autotune.py` | 完整 `train_step` 计时、autotune 回调与自适应启停的训练桥 |
| `hcu_megatron/training/ultraep_adaptive.py` | 有界自适应选择状态机，处理预热、配对采样、收益判断和校准预算 |

### 修改

| 路径 | 说明 |
|------|------|
| `hcu_megatron/core/transformer/moe/moe_layer.py` | MoE 初始化/前向包装、autograd 回调、逻辑/物理专家执行配置切换及副本通信接线 |
| `hcu_megatron/features_manager/__init__.py` | 将 `UltraEPFeature` 加入 `ADAPTOR_FEATURES` |
| `hcu_megatron/core/distributed/distributed_data_parallel.py` | UltraEP 激活时延后 master 梯度 ready，副本归约后手动触发；baseline 恢复原 DDP hook |
| `hcu_megatron/core/distributed/param_and_grad_buffer.py` | 新增 3 个 wrapper：过滤 replica 参数出 DDP bucket、DDP init 期间屏蔽 replica、过滤 `full_param_layout` 计算 |
| `hcu_megatron/core/transformer/moe/experts.py` | 新增两个 sharded checkpoint wrapper，只导出 master 专家的 metadata |

### Patch 目标

MoE 初始化与前向复用基础 wrapper；其他 UltraEP 专用 patch 仅在
`--moe-enable-ultraep` 开启时注册：

- `MoELayer.__init__` / `.forward`（由基础 MoE wrapper 统一注册，内部按配置启用 UltraEP）
- `_ParamAndGradBuffer.__init__`
- `DistributedDataParallel.__init__`
- `DistributedDataParallel._make_backward_post_hook`
- `DistributedOptimizer.compute_full_param_layout`
- `_get_param_groups`
- `TEGroupedLinear._sharded_state_dict_grouped`
- `TEGroupedMLP.sharded_state_dict`
- `generate_state_dict`
- `hcu_megatron.training.training.train_step`（开启 autotune 或 adaptive 时注册）

`ultraep_backward_post_hook_wrapper` 统一包装 Megatron 或 HCU 的 DDP hook。
UltraEP 激活时，在副本梯度归约后通知 DDP ready；自适应关闭时调用原 hook。

## 使用方式

### 1. 安装 `ultra_ep` C++ 扩展

在容器内安装（DTK 平台专用 wheel）：

```bash
pip install "ultra_ep-1.0.0+<hash>-cp310-cp310-linux_x86_64.whl"
```

当前环境已验证 wheel 为 `ultra_ep 1.0.0+636be96`。可用下面的命令同时检查版本和自动调优 API：

```bash
python - <<'PY'
from importlib.metadata import version
import ultra_ep

required = (
    'autotune_iteration_end',
    'autotune_needs_iteration_time',
    'wait_grad_reduce',
    'wait_weight_sync',
)
assert hasattr(ultra_ep, 'AutotuneConfig')
assert all(hasattr(ultra_ep.Manager, name) for name in required)
print(version('ultra_ep'), ultra_ep.__file__)
PY
```

### 2. 训练脚本增加参数

在 `MOE_ARGS` 里追加：

```bash
--moe-enable-ultraep
--moe-num-redundant-experts-per-rank 2
--moe-ultraep-autotune
```

- `--moe-enable-ultraep`：启用 UltraEP（默认关闭）
- `--moe-num-redundant-experts-per-rank N`：每 EP rank 的副本专家数，`N ≥ 1`
- `--moe-ultraep-autotune`：在真实训练 iteration 中依次调优 Weight Sync 和 Grad Reduce
- `--moe-ultraep-autotune-start-iteration N`：本地调优起始 iteration，默认 3
- `--moe-ultraep-autotune-grad-reduce-max-sms N`：可选的 Grad Reduce SMS 正偶数上限

物理专家数计算：`num_global_physical = num_moe_experts + ep_size × N`。

例：`num_moe_experts=128, ep_size=8, N=2` → 物理专家数 = 144，每 rank 拥有 18 个物理专家（16 master + 2 replica）。

### 3. 自动调优说明

自动调优需要包含 `AutotuneConfig`、`autotune_iteration_end`、`wait_grad_reduce` 和 `wait_weight_sync` API 的 UltraEP runtime。项目会在启用参数后检查这些 API；旧 runtime 仍可运行基础 UltraEP，但开启自动调优时会给出明确错误。

开启后无需在启动脚本中固定以下性能变量：

```text
ULTRA_EP_WEIGHT_SYNC_PLAN_MODE
ULTRA_EP_WEIGHT_SYNC_HIP_COPY_MODE
ULTRA_EP_WEIGHT_SYNC_LDS_WAVES_PER_DEST
ULTRA_EP_WEIGHT_SYNC_THREADS_PER_BLOCK
ULTRA_EP_WEIGHT_SYNC_CTA_MULTIPLIER
ULTRA_EP_GRAD_REDUCE_NUM_SMS
```

`ULTRA_EP_GRAD_REDUCE_DETERMINISTIC` 及 `ULTRA_EP_QUOTA_*`、`ULTRA_EP_BALANCE_THRESHOLD` 不属于本次搜索范围；省略它们表示使用 runtime 默认值。严格确定性或特殊 placement 场景仍需显式配置。

调优只在前期校准阶段按需同步设备并测量完整 iteration；结束后 `train_step` 走直接快速路径。

### 4. 自适应启用

在已启用 UltraEP 的 `MOE_ARGS` 中追加：

```bash
--moe-ultraep-adaptive
```

功能默认关闭。开启后先预热 baseline，再交替比较 baseline 与 UltraEP 的完整
`train_step` 耗时，选择后固定启用或关闭。可与 `--moe-ultraep-autotune` 联用：
先调优通信参数，再进行比较；未开启 autotune 时使用默认或手动指定的通信参数。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--moe-ultraep-adaptive-samples` | 3 | 每种模式的有效样本数，奇数且至少 3 |
| `--moe-ultraep-adaptive-max-samples` | 5 | 补样上限，奇数且不小于初始样本数 |
| `--moe-ultraep-adaptive-min-speedup` | 0.01 | 启用所需的最小完整步耗时改善比例 |
| `--moe-ultraep-adaptive-max-calibration-steps` | 48 | 校准总步数上限，包含预热、调优、切换和无效步 |

- 各 rank 完整步耗时先取 MAX，再按样本中位数计算收益。默认收益达到 1% 且
  3 组配对均胜出时提前启用；结果有分歧时补到最多 5 组，最终按中位收益决定。
- 无效步不计入评分但消耗预算。预算耗尽时只使用完整配对；样本不足或收益
  未达门槛则选择 baseline，校准期间负载配置变化也回退 baseline。
- 支持 TP/PP，要求 CP1/ETP1、无 bias 的 BF16 TE SwiGLU grouped experts、alltoall，
  要求开启梯度累积融合。不支持 EP overlap、CUDA graph、重计算、延迟 wgrad、
  shared-expert overlap、latent MoE、MTP 和 TE op fuser。
- 未开启 UltraEP 时自动关闭 adaptive。校准结束后停止额外计时、同步和决策
  collective，rank 0 输出选择结果；重启后重新校准。回退后保留副本显存。

示例使用 `examples/qwen3/train_qwen3_30B_A3B_ultraep.sh`，启动方式沿用
`examples/qwen3/run_qwen.sh`；回归测试见 `tests/unit_tests/moe/test_ultraep_framework.py`。

### 5. Dispatcher 要求

基础 UltraEP 使用 `alltoall` 或 `flex`，不支持 `allgather`；自适应当前仅支持
`alltoall`，见上面的自适应支持范围。

### 6. Checkpoint

保存的 checkpoint **只包含 master 专家权重**，与关闭 UltraEP 时的 checkpoint 二进制兼容。不同 `N` 值之间的 checkpoint 可以互相加载。

## Forward / Backward 时序

```
forward:
  route → update_placement → weight_sync(async) → reroute(logical→physical)
  → preprocess → dispatch → wait_weight_sync → experts → combine → postprocess
  → _EPLBWeightSyncFunction.apply    # 非 recompute 路径

backward（forward 逆序触发）:
  _EPLBWeightSyncFunction.backward           # weight_sync，当前流等待完成
  MoE backward                               # replica 梯度写共享 buffer，master 写 main_grad
  _EPLBReplicaGradReduceStartFunction.bw     # 异步 grad_reduce
  _EPLBReplicaGradReduceFinishFunction.bw    # 等待完成 → 手动 register_grad_ready
```

## 快速回归

单节点 8卡 跑通 `examples/qwen3/train_qwen3_30B_A3B.sh` 50 iter，loss 曲线与 baseline（不开 UltraEP）一致：

| 配置 | iter50 lm loss | throughput (TFLOP/s/GPU) |
|------|----------------|--------------------------|
| baseline | 1.101745E+01 | ~40 |
| ultraep (N=2) | 1.101753E+01 | ~51 |

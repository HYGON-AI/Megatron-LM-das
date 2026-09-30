# UltraEP

## 简介

UltraEP 是基于冗余专家副本的 MoE 在线专家并行负载均衡方案。

在 MoE 训练里，路由器把 token 分给各个 expert 时经常出现热点：少数专家收到远多于平均值的 token，其所在 EP rank 通信/计算成为整个 all-to-all 的瓶颈。UltraEP 的思路是**在每个 EP rank 上多放 N 个「副本专家」（replica experts）**，路由时通过 `ultra_ep` C++ 运行时把热点逻辑专家 round-robin 到多个物理副本，从而把该逻辑专家承担的 token 分摊到多个 rank 上。

副本权重的存放、副本↔master 的权重同步、副本梯度的归约都由 `ultra_ep` C++ 运行时通过共享 GPU buffer 处理，与 Megatron 的 DDP / DistributedOptimizer 通过 autograd Function 时序衔接，对上层训练逻辑透明。

## 实现内容

### 新增

| 路径 | 说明 |
|------|------|
| `hcu_megatron/core/transformer/moe/eplb_manager.py` | 封装 `ultra_ep.Manager`，负责物理专家数量计算、per-microbatch 虚拟 layer 槽分配、routing_map 从逻辑空间 reroute 到物理空间 |
| `hcu_megatron/core/transformer/moe/moe_layer_ultraep.py` | 三个 autograd Function 控制 backward 时序 + 5 个注入到 `MoELayer` 的实例方法 + `MoELayer.__init__` / `.forward` 的 wrapper |
| `hcu_megatron/features_manager/moe/ultraep_feature.py` | `UltraEPFeature`：CLI 参数注册、参数校验、patch 注册 |
| `hcu_megatron/training/checkpointing.py` | native torch 格式 checkpoint 过滤 replica 专家的 wrapper |
| `hcu_megatron/training/ultraep_autotune.py` | 仅在校准需要时测量完整 `train_step`，并把 iteration 耗时上报给 UltraEP runtime |

### 修改

| 路径 | 说明 |
|------|------|
| `hcu_megatron/features_manager/__init__.py` | 将 `UltraEPFeature` 加入 `ADAPTOR_FEATURES` |
| `hcu_megatron/core/distributed/distributed_data_parallel.py` | DDP backward hook 对 `is_eplb_master` 参数早返回，master 梯度 ready 由 UltraEP backward Function 手动触发 |
| `hcu_megatron/core/distributed/param_and_grad_buffer.py` | 新增 3 个 wrapper：过滤 replica 参数出 DDP bucket、DDP init 期间屏蔽 replica、过滤 `full_param_layout` 计算 |
| `hcu_megatron/core/transformer/moe/experts.py` | 新增两个 sharded checkpoint wrapper，只导出 master 专家的 metadata |

### Patch 目标

`--moe-enable-ultraep=True` 时注册以下基础 patch，未启用时零开销：

- `MoELayer.__init__` / `.forward`
- `_ParamAndGradBuffer.__init__`
- `DistributedDataParallel.__init__`
- `DistributedOptimizer.compute_full_param_layout`
- `_get_param_groups`
- `TEGroupedLinear._sharded_state_dict_grouped`
- `TEGroupedMLP.sharded_state_dict`
- `generate_state_dict`
- `hcu_megatron.training.training.train_step`（仅在同时开启 `--moe-ultraep-autotune` 时注册）

## 使用方式

### 1. 安装 `ultra_ep` C++ 扩展

在容器内安装（DTK 平台专用 wheel）：

```bash
pip install ultra_ep-1.0.0+<hash>-cp310-cp310-linux_x86_64.whl
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

### 4. 最新 runtime 验证结果

2026-09-21 在当前单节点环境完成以下验证：

| 检查项 | 结果 |
|------|------|
| UltraEP wheel | `1.0.0+636be96` |
| 软件/设备 | PyTorch `2.10.0`、HIP `6.3.26113`、8 × BW（80 SMS/卡） |
| API 契约 | `AutotuneConfig`、Manager `autotune` 参数及四个调优/等待接口全部通过 |
| Megatron 接线 | CLI 校验、`AutotuneConfig` 参数传递、train-step 快速/计时路径全部通过 |
| 8-rank 硬件烟测 | RCCL、GDA rocSHMEM、autotune Manager 构造、预热回调及 placement 聚合通过 |
| 静态检查 | Python compile、Shell 语法、`git diff --check` 通过 |

硬件烟测使用极小 expert 张量，不等同于重新执行 50 iteration 的完整模型性能测试；完整负载的历史结果见下方“快速回归”。

### 5. Dispatcher 要求

`--moe-token-dispatcher-type` 必须是 `alltoall` 或 `flex`。`allgather` 不支持。

### 6. Checkpoint

保存的 checkpoint **只包含 master 专家权重**，与关闭 UltraEP 时的 checkpoint 二进制兼容。不同 `N` 值之间的 checkpoint 可以互相加载。

## Forward / Backward 时序

```
forward:
  route → update_placement → weight_sync(async) → reroute(logical→physical)
  → preprocess → dispatch → wait_weight_sync → experts → combine → postprocess
  → _EPLBWeightSyncFunction.apply    # 非 recompute 路径

backward（forward 逆序触发）:
  _EPLBWeightSyncFunction.backward           # 异步 weight_sync
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

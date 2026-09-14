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

### 修改

| 路径 | 说明 |
|------|------|
| `hcu_megatron/features_manager/__init__.py` | 将 `UltraEPFeature` 加入 `ADAPTOR_FEATURES` |
| `hcu_megatron/core/distributed/distributed_data_parallel.py` | DDP backward hook 对 `is_eplb_master` 参数早返回，master 梯度 ready 由 UltraEP backward Function 手动触发 |
| `hcu_megatron/core/distributed/param_and_grad_buffer.py` | 新增 3 个 wrapper：过滤 replica 参数出 DDP bucket、DDP init 期间屏蔽 replica、过滤 `full_param_layout` 计算 |
| `hcu_megatron/core/transformer/moe/experts.py` | 新增两个 sharded checkpoint wrapper，只导出 master 专家的 metadata |

### Patch 目标

`--moe-enable-ultraep=True` 时注册以下 patch，未启用时零开销：

- `MoELayer.__init__` / `.forward`
- `_ParamAndGradBuffer.__init__`
- `DistributedDataParallel.__init__`
- `DistributedOptimizer.compute_full_param_layout`
- `_get_param_groups`
- `TEGroupedLinear._sharded_state_dict_grouped`
- `TEGroupedMLP.sharded_state_dict`
- `generate_state_dict`

## 使用方式

### 1. 安装 `ultra_ep` C++ 扩展

在容器内安装（DTK 平台专用 wheel）：

```bash
pip install ultra_ep-1.0.0+<hash>-cp310-cp310-linux_x86_64.whl
```

验证：

```bash
python -c "import ultra_ep; print(ultra_ep.__file__)"
```

### 2. 训练脚本增加参数

在 `MOE_ARGS` 里追加：

```bash
--moe-enable-ultraep
--moe-num-redundant-experts-per-rank 2
```

- `--moe-enable-ultraep`：启用 UltraEP（默认关闭）
- `--moe-num-redundant-experts-per-rank N`：每 EP rank 的副本专家数，`N ≥ 1`

物理专家数计算：`num_global_physical = num_moe_experts + ep_size × N`。

例：`num_moe_experts=128, ep_size=8, N=2` → 物理专家数 = 144，每 rank 拥有 18 个物理专家（16 master + 2 replica）。

### 3. Dispatcher 要求

`--moe-token-dispatcher-type` 必须是 `alltoall` 或 `flex`。`allgather` 不支持。

### 4. Checkpoint

保存的 checkpoint **只包含 master 专家权重**，与关闭 UltraEP 时的 checkpoint 二进制兼容。不同 `N` 值之间的 checkpoint 可以互相加载。

## Forward / Backward 时序

```
forward:
  route → update_placement → weight_sync(async) → reroute(logical→physical)
  → preprocess → wait_weight_sync → dispatch → experts → combine → postprocess
  → _EPLBWeightSyncFunction.apply    # 非 recompute 路径

backward（forward 逆序触发）:
  _EPLBWeightSyncFunction.backward           # 异步 weight_sync
  MoE backward                               # replica 梯度写共享 buffer，master 写 main_grad
  _EPLBReplicaGradReduceStartFunction.bw     # 异步 grad_reduce
  _EPLBReplicaGradReduceFinishFunction.bw    # 等待完成 → 手动 register_grad_ready
```

## 快速回归

单节点 8×DCU 跑通 `examples/qwen3/train_qwen3_30B_A3B.sh` 50 iter，loss 曲线与 baseline（不开 UltraEP）一致：

| 配置 | iter50 lm loss | throughput (TFLOP/s/GPU) |
|------|----------------|--------------------------|
| baseline | 1.101745E+01 | ~40 |
| ultraep (N=2) | 1.101753E+01 | ~51 |

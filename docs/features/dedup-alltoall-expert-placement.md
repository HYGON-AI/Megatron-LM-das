# 去重 Dispatcher 与相关专家放置

## 简介

在 MoE 的 top-k 路由中，一个 token 会被分配给多个专家。Megatron 原生 `alltoall` dispatcher 按 `(token, expert)` 发送数据；即使多个专家位于同一个 EP rank，仍会重复发送该 token 的 hidden states。

本特性通过两项可独立开启、也可组合使用的优化减少通信：
去重 dispatcher可直接使用，相关专家放置需要先训练一定轮数（1000-2000），采集路由信息并离线拟合放置后，使用其生成的json文件即可使用。

- **去重 dispatcher**：按 `(token, 目标 EP rank)` 发送数据，同一 token 发往同一 rank 时只发送一次；接收端展开给本地专家计算，合并本地结果后再回传。
- **相关专家放置（Correlated Expert Placement）**：根据实际路由中专家被共同选中的频率，离线调整专家到 EP rank 的映射，让经常共同命中的专家尽量位于同一 rank，进一步增加去重机会。

两者保持逻辑专家的计算和路由权重含义不变，不减少每个 token 需要执行的专家计算，也不引入专家副本。浮点求和顺序会变化，因此结果不保证逐位一致。

## 实现原理

### 1. 按目标 rank 去重

假设一个 token 选择了 4 个专家，其中 3 个位于 rank 0，另 1 个位于 rank 1：

| Dispatcher | 发往 rank 0 的 hidden 行数 | 发往 rank 1 的 hidden 行数 | 合计 |
|---|---|---|---|
| 原生 alltoall | 3 | 1 | 4 |
| 去重 alltoall | 1 | 1 | 2 |

对于固定 top-k、无丢弃和额外 padding 的路由，若每个 token 平均命中 `d` 个不同的 EP rank，则 hidden 行数由 `T × k` 降为 `T × d`，行数去重比例为 `1 - d/k`。这里包含留在本 rank 的行，不能直接等同于跨卡或跨节点流量；实际字节量还取决于附带的路由元数据。

前向流程如下，反向按对应索引回传 hidden states 和路由概率的梯度：

```mermaid
flowchart LR
    A[路由结果] --> B[按目标 rank 合并并打包]
    B --> C[Dispatch All-to-All]
    C --> D[展开到本地专家并计算]
    D --> E[按接收行合并专家输出]
    E --> F[Combine All-to-All]
    F --> G[按原 token 汇总]
```

每个发送行携带 `[hidden states | 本 rank 的专家概率]`。概率按原 dtype 的位模式重解释为通信 buffer 的列，与 hidden states 一起进行主数据 All-to-All；在常用 BF16 hidden、FP32 probs 配置下，不会把概率数值舍入为 BF16。未选中专家用 `-inf` 标记，接收端通过 `> -inf` 恢复 mask，能够区分“未选中”和“选中但概率为 0”的 padding 项。行数等计数信息仍需单独交换。

实现使用一维 cumsum 和 scatter 构建压缩索引，通过 `gather_sum`、`gather_elem` 自定义算子完成展开、聚合及反向搬运。聚合按输出行 gather 求和，常用低精度输入以 FP32 累加，避免 atomic 累加和额外的清零、类型转换 kernel。此路径不调用 TE permute/unpermute，`--moe-permute-fusion` 对它不起作用。动态 All-to-All 的 split sizes 仍需要每层一次设备到主机同步。

### 2. 根据专家相关性调整放置

离线工具对每个 MoE 层统计专家共同命中的次数，以增加同 rank 内共同命中的专家对为优化目标，在保持每 rank 专家数量为 `E/P` 的约束下交换专家。多节点配置先划分到节点，再划分到节点内各 rank；该目标是通信收益的代理指标，最终需要用独立路由样本评估。

放置文件中的 `rank_to_experts[r]` 指定 EP rank `r` 承载的逻辑专家。运行时将各 rank 的列表顺序拼接成 `phys2log`，并把 router 输出的 `probs`、`routing_map` 从逻辑顺序重排为物理顺序。dispatcher 仍按连续物理槽位工作，router、aux loss 和 expert bias 仍按逻辑专家编号计算。

相关专家放置本身不会合并发送行。配合去重 dispatcher 或具备去重能力的 `flex`/DeepEP 才能减少重复传输；单独配合原生 `alltoall` 仍按 `(token, expert)` 发送。

## 使用方式

### 1. 仅开启去重 dispatcher

在已有 MoE 训练脚本中添加以下参数，无需路由采集或放置文件：

```shell
--moe-token-dispatcher-type alltoall
--moe-dedup-alltoall
```

`--moe-dedup-alltoall` 默认为关闭；开启后，用 `MoEDedupAlltoAllTokenDispatcher` 替换原生 `MoEAlltoAllTokenDispatcher`。

原生 index kernel 会在模型构造时通过 PyTorch C++ extension JIT 编译/加载，需要可用的 CUDA/HIP 编译环境。编译失败会告警并回退到等价的 PyTorch 实现；

### 2. 采集路由

若还需启用相关专家放置，先在已有训练任务的启动环境中设置：

```shell
export EC_TRACE_ITERS=250,252,500,502,980,984,988,992,996
export EC_TRACE_DIR=/path/to/ec_traces
# 随后按原有方式启动或续跑训练，确保覆盖指定迭代。
```

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `EC_TRACE_ITERS` | 未设置，关闭采集 | 逗号分隔的迭代号，与训练 `train_step` 的 iteration 参数一致 |
| `EC_TRACE_DIR` | `ec_traces` | 路由输出目录 |


在仓库根目录执行以下命令，按全局层号合并路由。示例为 PP=1，实际使用时将 `--pp` 改为采集时的 PP 大小：

```shell
python hcu_megatron/training/route_collector.py merge \
    --dir /path/to/ec_traces \
    --pp 1 \
    --iters 250,252,500,502,980,984,988,992,996
```

输出为目录下的 `s980.pt`、`s984.pt` 等文件，每个文件包含按全局层序排列的各 MoE 层路由。

### 3. 离线拟合放置

下面以 128 专家、EP=16、EP 组跨 2 节点为例，用 988/992/996 步拟合，用 980/984 步独立评估：

```shell
python tools/expert_placement/fit_placement.py \
    --fit /path/to/ec_traces/s988.pt \
          /path/to/ec_traces/s992.pt \
          /path/to/ec_traces/s996.pt \
    --eval /path/to/ec_traces/s980.pt \
           /path/to/ec_traces/s984.pt \
    --num-experts 128 \
    --ep 16 \
    --nodes 2 \
    --out /path/to/ec_traces/placement.json
```

工具输出连续放置与相关放置的逐层指标及均值，并生成训练可直接加载的 JSON 文件。

| 拟合参数 | 默认值 | 说明 |
|---|---|---|
| `--num-experts`、`--ep` | 必填 | 逻辑专家数、目标 EP 大小，必须匹配训练配置 |
| `--nodes` | `1` | EP 组跨越的节点数；需整除 EP 大小，并假设连续的 EP rank 块对应同一节点 |
| `--init` | `contiguous` | 初始放置，可选 `contiguous` / `greedy` |
| `--max-swaps` | `1000` | 每个划分阶段的最大交换次数 |
| `--max-load-ratio` | 不限制 | 按拟合样本的专家 token 负载约束候选交换，如 `1.1`；不保证新样本的实际接收/计算负载上限 |
| `--router-groups` | `1` | 分组路由的专家组数，应与训练的 `--moe-router-num-groups` 一致 |
| `--eval` | 未设置 | 独立评估样本；省略时在拟合样本上评估，结果可能偏乐观 |

评估时同时关注 `gpus_per_token`、`dedup_pct`、`nodes_per_token`，以及 `recv_imbalance`（接收行数的最大值/均值）、`compute_imbalance`（专家计算行数的最大值/均值）。去重比例提高时，热点 rank 的负载也可能上升，不能仅凭平均通信行数选择放置。

### 4. 加载放置并训练

在原有训练参数上添加：

```shell
--moe-token-dispatcher-type alltoall
--moe-dedup-alltoall
--moe-expert-placement-path /path/to/ec_traces/placement.json
--ckpt-format torch_dist
```

如果使用 `--use-distributed-optimizer`，在生成供后续切换放置的 checkpoint 及加载时，还需添加：

```shell
--dist-ckpt-optim-fully-reshardable
```

放置路径默认不设置。所有训练进程需能读取该文件；文件的专家数、EP 大小、MoE 层数必须匹配模型，每个逻辑专家必须恰好出现一次，每 rank 必须有相同数量的专家。文件按 decoder 中 MoE 层的顺序排列，MTP 层保持连续放置。

放置在模型初始化时加载，不会在训练过程中自动更新。路由相关性随训练变化，长训练可重新采样、拟合，再通过保存和重启加载 checkpoint 切换放置。

### 5. Checkpoint 与已有权重

启用放置后，`torch_dist` checkpoint 的专家轴仍按逻辑专家编号保存，可在连续放置、放置 A、放置 B 之间加载。使用分布式优化器时，完整重分片选项保证优化器状态随逻辑专家正确映射；普通按物理 bucket 保存的状态不能直接用于切换放置。


## 支持范围与限制

| 项目 | 说明 |
|---|---|
| Dispatcher 类型 | 去重开关仅支持 `alltoall`；放置也可配合已有的 `flex`/DeepEP 去重路径 |
| 路由 | 不支持 `--moe-pad-expert-input-to-capacity` |
| CUDA graph | 去重 dispatcher 当前不支持，要求 `cuda_graph_impl=none` |
| 分组路由 | 放置只能在每个逻辑 router group 原本覆盖的 EP rank 范围内调整，越界启动报错 |
| Checkpoint | 放置读写仅支持 `torch_dist`，不支持 `singleton_local_shards`；分布式优化器还要求完整重分片 |


## 验证与排查

启动后，rank 0 首次输出以下形式的日志，可确认去重路径已启用：

```text
[dedup alltoall] first layer on rank 0: ... rows sent instead of ... (...% removed)
[expert placement] ...: ... MoE layers, EP=..., layer ... rank 0 hosts experts ...
```

第二行仅在启用放置时出现。第一行仅统计 rank 0 首次调用的一层，包含本地行，不代表全模型的跨节点流量下降比例。


主要实现位于 [去重 dispatcher](../../hcu_megatron/core/transformer/moe/dedup_token_dispatcher.py)、[专家放置](../../hcu_megatron/core/transformer/moe/expert_placement.py)、[参数与 patch 注册](../../hcu_megatron/features_manager/moe/expert_placement_feature.py)、[路由采集](../../hcu_megatron/training/route_collector.py) 和 [离线拟合工具](../../tools/expert_placement/fit_placement.py)。

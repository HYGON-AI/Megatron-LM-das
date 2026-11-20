# TAOT

## 简介

TAOT 是基于拓扑感知规划和临时专家副本的 MoE 专家并行负载均衡方案。本项目参考 [LoongForge](https://github.com/baidu-baige/LoongForge)的相关实现，使用 `--moe-enable-echo` 启用，参数和代码中沿用 `echo expert` 命名。

TAOT 通过临时复制专家，让负载较低的 rank 分担热点 rank 的计算任务。它保留原有的 home 专家，并在每个 rank 上预留 echo 专家槽位。每次 forward 在路由确定后，根据各 rank 相对于平均负载的忙闲程度，选择需要复制的专家和目标槽位。权重复制完成后，部分 token 交给对应副本计算。token 选择的逻辑专家不变，只是执行位置发生变化。TAOT 在规划时同时考虑负载均衡收益和通信代价，优先利用同节点的空闲计算能力，必要时再跨节点分担负载，避免计算加速被权重搬运开销抵消。

反向传播时，副本产生的权重梯度通过通信回传并累加到 home 专家，最终由 home 参数统一参与优化器更新。


## 使用方式

### 1. 训练脚本增加参数

使用 All-to-All 权重搬运：

```bash
--moe-enable-echo
--moe-num-echo-experts 16
--moe-echo-expert-dispatch-type alltoall
--moe-echo-algorithm sinkhorn
--moe-echo-ranks-per-node 8
```

令原始专家数为 `E`、EP 大小为 `R`、echo 总数为 `S`，则：

```text
每 rank home 专家数 = E / R
每 rank echo 槽位数 K = S / R
每 rank 专家槽位总数 = E / R + K
EP 组内专家槽位总数 = E + S
```

例：`E=128, R=8, S=16`，每 rank 为 16 个 home 专家和 2 个 echo 槽位，共 18 个槽位；EP 组内共 144 个槽位。
### 2. 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--moe-enable-echo` | 关闭 | 启用 TAOT/Echo 规划、专家权重搬运和 token 重路由 |
| `--moe-num-echo-experts S` | 未设置 | 启用时必填；必须为正数且能被 EP 大小整除 |
| `--moe-echo-expert-dispatch-type` | `alltoall` | 专家权重搬运后端，可选 `alltoall`、`deepep` |
| `--moe-echo-algorithm` | `sinkhorn` | 可选 `sinkhorn`、`greedy`、`auction` |
| `--moe-echo-ranks-per-node` | `8` | planner 的拓扑提示，表示每物理节点包含的 EP ranks 数量 |
| `--moe-echo-expert-dispatch-overlap` | 关闭 | 将 FC1 / FC2 权重通信分别与对应的 home GEMM 重叠 |
| `--moe-echo-dump-dir` | 未设置 | 输出规划文件；当前代码每层、每 rank 仅保存首次规划 |

`greedy` 在每 rank 只有 1 个 echo 槽位时使用 `approx_bin_packing`，多个槽位时使用 `one_shot_greedy`。

### 3. DeepEP 专家权重搬运

将专家权重后端切换为：

```bash
--moe-echo-expert-dispatch-type deepep
```

如果 token 通信也使用 DeepEP，可同时配置：

```bash
--moe-token-dispatcher-type flex
--moe-flex-dispatcher-backend deepep
```


### 4. 开启通信计算重叠

```bash
--moe-echo-expert-dispatch-overlap
```
开启后，专家权重搬运与本地 home 专家的计算并行执行。系统先计算不依赖远端权重的 home 专家，在 echo 副本开始计算前等待所需权重就绪，从而利用计算时间隐藏部分通信开销。实际收益取决于权重搬运量和 home 专家的计算量。

## Forward / Backward 时序

初始化时，为原有 home 专家和新增 echo 副本预留计算槽位，同时保持 router 只对原有逻辑专家进行路由。echo 槽位不作为独立的优化器参数，其权重在 forward 时由选定的 All-to-All 或 DeepEP dispatcher 从 home 专家复制，反向产生的梯度再回传并累加到 home 参数，由 home 参数统一更新。

```
forward（公共部分）:
  route 一次 → 汇总各 EP rank 的 token 计数 → 生成卸载计划
  → 同步 EP rank 0 的专家映射和搬运计数 → 生成重路由 map / probs

forward（关闭 overlap）:
  FC1 权重搬运与安装 → FC2 权重搬运与安装
  → token dispatch → home / echo experts → token combine → postprocess

forward（开启 overlap）:
  token dispatch
  → FC1 权重通信 ∥ home FC1 GEMM → 等待 → echo FC1 GEMM
  → FC2 权重通信 ∥ home FC2 GEMM → 等待 → echo FC2 GEMM
  → 拼接 home / echo 输出 → token combine → postprocess

backward:
  副本专家计算梯度 → 权重通信的反向路径回传 → 累加到 home 专家
  overlap 的权重梯度通信按 FC2 → FC1 顺序执行
```


### 使用效果

| 配置    | 模型参数                      | 设备数     | echo-experts | 单步时长（ms）|加速  |
| ------- | ---------------------------- | --------- | ------------ | ------------ |------|
| baseline| seq-length 4096，EP 16，PP 1 | 16卡(双机) | 0            | 2205.7ms     |0     |
| sinkhorn| seq-length 4096，EP 16，PP 1 | 16卡(双机) | 32           | 1443.8ms     |34.54%|

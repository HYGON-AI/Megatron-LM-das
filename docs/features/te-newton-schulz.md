# Muon 的 Transformer Engine Newton–Schulz 实现

## 背景

Muon 在更新二维权重时，需要对梯度执行 Newton–Schulz（NS）正交化。张量并行（TP）训练中的大矩阵会被分片；当正交化计算成为优化器步骤的瓶颈时，需要一种能在 TP 分片上执行 NS 的实现。

本特性为 Muon 的 `distributed` TP 模式接入 Transformer Engine（TE）的分布式 NS，作为现有实现的可选替代。默认仍使用 `native` 实现（基于Emerging-Optimizers库）。

## 实现内容

- 新增 `--newton-schulz-tp-impl` 开关，只在选择 `transformer_engine` 时替换 Megatron 中的 `newton_schulz_tp` 调用。非 TP 路径和 `duplicated` 模式继续使用 Emerging-Optimizers。
- 在 TE 路径中，对 TP 分片先计算全局 Frobenius 范数并归一化；对按行分片的矩阵转置，以符合 TE 按列分片的输入要求，计算后再转回原布局。
- 复用调用方提供的 TP 进程组，并缓存对应的 TE `CusolverMpCtx`，避免每次优化器更新都重建上下文。TE 原地修改输入，因此适配层使用独立缓冲区，保持原有接口的输入输出语义。

## 使用方式

在 Muon 训练参数中加入：

```sh
--optimizer muon \
--muon-tp-mode distributed \
--newton-schulz-tp-impl transformer_engine
```

`--newton-schulz-tp-impl` 可选 `native`（默认）或 `transformer_engine`。选择 TE 时会自动启用优化器特性注册；需要 Transformer Engine 2.15.0 及以上版本，以及可用的分布式 NS 后端。该开关仅改变 TP `distributed` 模式下的 NS 实现，不改变 Muon 的其他更新逻辑。

## 适用场景与预期收益

适用于 **TP 大于 1、使用 Muon `distributed` 模式，且 NS 在优化器步骤中占比较高** 的训练任务。TE 的分布式实现有望降低这部分计算耗时；矩阵规模越大、NS 开销占比越高，越值得评估。收益应以相同模型、并行配置和训练参数下的优化器耗时及端到端迭代时间为准。

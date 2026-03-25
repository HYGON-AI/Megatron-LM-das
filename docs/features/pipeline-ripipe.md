
# 重计算流水线独立调度RiPipe（Recompute independent Pipelining）
## RiPipe功能是什么？

在目前的流水线调度中，重计算由反向计算触发，与反向计算绑定在一起调度，意味着重计算需要等待下一个stage返回梯度才可以开始计算。然而重计算并不需要用到反向计算的梯度，这导致bubble的增多和性能的下降。RiPipe是一个在标准的交错式流水线并行基础上优化的深度学习训练调度器。在未开启重计算的模式中，它通过流水线bubble中“偷跑”重计算任务，在极小的性能开销代价下带来减少显存节省。在开启重计算的模式中，它通过提前执行重计算任务，来减少流水线气泡，带来性能提升。


## RiPipe功能实现原理
为了将重计算和反向计算独立调度，需要将重计算的调度修改为由调度器主动触发，并修改调度器，将重计算作为一个调度单元加入到调度器中，这使我们获得了自由地插入或去除部分重计算的能力，进而可以在内存和性能方面做出优化。通过torch的saved_tensors_hooks实现一种新的重计算方法，在反向计算前合适的时机主动触发或者直接去除部分重计算，从而实现对内存或性能的优化。

RiPipe通过智能的重计算策略来减少内存使用和提高训练效率。该功能包含两种主要模式：

1. **recompute-in-advance**：提前重计算模式，通过提前执行重计算来减少流水线气泡
2. **recompute-in-bubble**：气泡重计算模式，利用流水线中的气泡时间进行重计算

## RiPipe功能兼容性限制

- `--recompute-in-bubble` 特性暂不兼容完全重计算uniform、完全重计算block、选择重计算、自适应选择重计算、swap-attention、no-align-grad-reduce和no-overlap-p2p-communication特性
- `--recompute-in-bubble` 不兼容moe场景下的--moe-adaptive-recompute-activation、--moe-layer-recompute特性
- `--recompute-in-advance` 特性暂不兼容完全重计算uniform、完全重计算block、选择重计算、自适应选择重计算、no-align-grad-reduce和no-overlap-p2p-communication特性
- `--recompute-in-bubble`和`--recompute-in-advance`两者不可同时开启


## RiPipe功能使用方法

### 1. --recompute-in-advance 配置示例

在训练脚本中添加以下参数：

```bash
# 提前重计算模式
--schedule-method=ripipe
--recompute-in-advance
--recompute-num-layers 1

# 必需的流水线配置
--pipeline-model-parallel-size 8
--virtual-pipeline-model-parallel-size 8  # 或其他值
--num-layers-per-virtual-pipeline-stage 1

# 必需的重计算配置
--recompute-granularity full
--recompute-method block
--recompute-modules mlp
```

### 2. --recompute-in-bubble 配置示例
```bash
# 气泡重计算模式
--schedule-method=ripipe
--recompute-in-bubble

# 必需的流水线配置
--pipeline-model-parallel-size 8
--virtual-pipeline-model-parallel-size 8  # 或其他值
--num-layers-per-virtual-pipeline-stage 1

# 不打开重计算
```

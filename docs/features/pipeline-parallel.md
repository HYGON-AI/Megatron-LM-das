## 流水线并行
hcu megatron对megatron已有流水线调度策略进行了优化，并提供一些额外的流水线方法。

### 背景
在interleaved 1f1b的稳态阶段，前/反向计算交替执行。针对moe模型，如果训练时开启ep并行，ep间的a2a通信耗时会在端到端时间中占据相当比重，对训练效率造成很大影响，如下图所示。
<figure style="text-align:center;">
  <img src=../source/images/interleaved_1f1b_origin_moe.png alt="示例图"/>
  <figcaption>
  图1. interleaved 1f1b中稳态阶段前/反向计算交替执行。针对moe模型，A2A通信在端到端时间中占据相当比重，对训练效率影响很大
  </figcaption>
</figure>

为了降低ep通信对训练效率影响，DeepSeek提出了DualPipe流水线并行算法，实现了两个micro batch forward和backward之间的 **ep a2a 通信与计算 overlap**，使得计算能全部或部分掩盖a2a通信，可极大提高训练效率。

<figure style="text-align:center;">
  <img src=../source/images/moe_a2a_overlap.png alt="示例图"/>
  <figcaption>
  图2. 不同micro batch间ep a2a通信与计算overlap
  </figcaption>
</figure>

小红书将ep a2a overlap引入其他流水线方法，提出了基于1F1B的ep a2a通信与计算的overlap。在原始的interleaved 1F1B 调度中，最后一个 pp stage 稳态阶段的 f 和 b 之间存在数据依赖，无法实现 f 和 b 之间的 EP A2A 通信与计算 overlap。为了解决这个问题，需要把最后一个 pp stage 的 warmup step 加 1。同时，考虑到 pp stage 间前反向的数据依赖，其他 pp stage 也需要调整 warmup step。

<figure style="text-align:center;">
  <img src=../source/images/interleaved_1f1b_adjust_warmup.png alt="示例图"/>
  <figcaption>
  图3. interleaved 1f1b的warmup阶段micro batch数加1
  </figcaption>
</figure>

### interleaved 1f1b支持moe a2a overlap
hcu megatron支持基于interleaved 1f1b的moe a2a overlap，如需使用该特性，训练时需要在脚本中增加以下参数：
```
--overlap-moe-expert-parallel-comm
```

### 分离参数梯度计算
如果前向计算比通信耗时短，反向计算比通信耗时多，前向计算不足以overlap反向的通信, 如下图所示。
<figure style="text-align:center;">
  <img src=../source/images/moe_a2a_overlap_split_bw1.png alt="示例图"/>
  <figcaption>
  图4. 前向计算无法完全掩盖ep通信
  </figcaption>
</figure>

此时可以拆分反向dw计算，用 dw 来 overlap 部分反向通信。拆分 dw 之后，计算流上 kernel 的排布更加紧密。
<figure style="text-align:center;">
  <img src=../source/images/moe_a2a_overlap_split_bw2.png alt="示例图"/>
  <figcaption>
  图5. 拆分dw计算
  </figcaption>
</figure>
如果开启dw拆分，需要在脚本中增加以下参数:

```
--delay-wgrad-compute
```
在开启--delay-wgrad-compute时，如果同时开启--overlap-grad-reduce，需要在启动脚本中增加以下环境变量：
```
export NVTE_OVERLAP_GRAD_REDUCE=1
```

### 拆分attn，缓解tp与ep竞争
针对图2中的调度排布，如果训练时同时开启tp和ep，ep会与attn部分的tp重叠，为了解决该问题，hcu megatron将attn部分拆分为三部分：（1）qkv计算，（2）core attn计算和（3）proj计算，并重新组织调度排布，如下图所示。
<figure style="text-align:center;">
  <img src=../source/images/moe_a2a_overlap_split_attn.png alt="示例图" style="width: 70%; height: auto;"/>
  <figcaption>
  图5. attn计算拆分为三部分
  </figcaption>
</figure>
hcu megatron默认使用该调度方案。如考虑使用图2中的调度方案(megatron v0.14及后续版本提供)，需要在训练脚本中增加以下参数:

```
--overlap-ep-comm-with-split-attn
```

### dualpipev流水线
hcu megatron提供dualpipev流水线调度，每个stage上有两个模型chunk，如下图所示。
<figure style="text-align:center;">
  <img src=../source/images/dualpipev.png alt="示例图"/>
  <figcaption>
  图6. dualpipev流水线
  </figcaption>
</figure>
如果使用dualpipev流水线，需要在脚本中增加以下参数：  

```
--schedule-method dualpipev
--delay-wgrad-compute  # dualpipev自动开启该参数，可去掉
```
dualpipev支持moe a2a overlap，开启overlap需要额外增加以下参数：  
```
--overlap-moe-expert-parallel-comm
--overlap-ep-comm-with-split-attn # 可选参数，如果使用该参数，attn被拆分为三部分
```
dualpipev可通过指定每个stage中transformer层数的方式对网络进行切分，使用该切分方式时需要增加如下参数
```
--num-layers-to-build  *** # 整数或者数组，如果为整数，表示每个chunk上的网络层数；如果为数组，数组长度为stage数的两倍，数组的元素值为对应chunk的transformer层数，顺序与前向计算一致
```
<figure style="text-align:center;">
  <img src=../source/images/dualpipev-2.png alt="示例图" style="width: 50%; height: auto;"/>
  <figcaption>
  图7. dualpipev切分，图中数字为每个chunk中的transformer层数。该情形下，可设置num-layers-to-build为[2,3,1,4,3,2,1,3]
  </figcaption>
</figure>

### ZB-H1流水线
hcu megatron支持ZB-H1流水线调度，通过拆分参数/激活值梯度计算，减少流水线气泡，具体见[Zero Bubble Pipeline Parallelism](https://github.com/sail-sg/zero-bubble-pipeline-parallelism/tree/zero-bubble-v0.1.0)。
ZB-H1流水线显存占用与1f1b流水线相同，在小batch情形下，训练性能有明显提升。使用该流水线调度时，需要在脚本中增加以下参数:
```
--schedule-method zb_h1
--delay-wgrad-compute # zb_h1自动开启该参数，可去掉
```
**注意**<br>
1、使用该流水线调度需要满足流水线stage数大于1;<br>
2、不支持开启vp。

### 1f1b流水线优化
hcu megatron支持对1f1b流水线cooldown阶段的参数/激活值梯度计算进行拆分，提升小batch情形下的训练性能。
开启该特性，需要在脚本中增加以下参数:
```
--delay-1f1b-cooldown-wgrad-compute
--delay-wgrad-compute # 使用delay-1f1b-cooldown-wgrad-compute时自动开启该参数，可去掉
```

<figure style="text-align:center;">
  <img src=../source/images/1f1b-delay-cooldown-stage-wgrad-compute.png alt="示例图" style="width: 70%; height: auto;"/>
  <figcaption>
  图8. 对1f1b流水线cooldown阶段的参数/激活值梯度计算进行拆分，提升小batch情形下的训练性能
  </figcaption>
</figure>

**注意**<br>
1、使用该特性需要满足流水线stage数大于1;<br>
2、不支持开启vp。


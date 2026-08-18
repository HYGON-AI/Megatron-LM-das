### 基于 Flux 的tp计算方法
[Flux](https://github.com/bytedance/flux)是字节提供的一个通算融合库，旨在通过计算掩藏GPU间的通信，提高模型训练/推理性能。  
本项目提供基于flux的tp并行计算方法，该方法使用flux相关kernel融合下图红框中的计算通信。

<figure style="text-align:center;">
  <img src=../source/images/tp-comp-comm-overlap.png alt="示例图" width="450"/>
  <figcaption>
  图1.
    <a href="https://mp.weixin.qq.com/s/ePANCDqQnvRuafsMLzC7IA" target="_blank" rel="noopener">tp-comp-comm-overlap</a>
  </figcaption>
</figure>

用户可以选择使用该方法替换megatron中tp计算方法，获得更好的训练和推理性能。使用该特性时需要启动脚本中加入如下参数：
```
--parallel-linear-impl flux
```
为了获得更好的训练性能，用户可选择保存前向all-gather数据，用于反向权重梯度计算，反向时无需再进行一次all-gather操作。

<figure style="text-align:center;">
  <img src=../source/images/tp-comp-comm-overlap-2.png alt="示例图" width="450"/>
  <figcaption>
  图2. 保存前向ag结果，用于反向计算
  </figcaption>
</figure>
如需使用该特性，需要在启动脚本中额外添加以下参数：

```
--save-flux-gather-input
```

对于图3红框内的计算通信，可以不使用flux kernel进行融合。此时反向计算时RS与QKW_WGRAD/FC1_WGRAD进行overlap；如果不使用save-flux-gather-input，QKV_DGRAD/FC1_DGRAD与AG进行overlap。

<figure style="text-align:center;">
  <img src=../source/images/tp-comp-comm-overlap-3.png alt="示例图" width="450"/>
  <figcaption>
  图3. 对于红框内计算通信，可以不替换为flux kernel
  </figcaption>
</figure>
如果不使用flux kernel进行融合图3红框内的计算通信，需要在启动脚本中额外添加以下参数：

```
--disable-bw-flux-gemmrs-op
```

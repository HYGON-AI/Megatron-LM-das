**参数副本复用**



在大模型训练中，需要保存一份FP32的参数，同时复制一份BF16的参数做训练，考虑到FP32+BF16的参数不会同时出现，通过将 FP32转换为BF16+残差，实现BF16复用，在反向更新参数时，通过BF16+残差恢复FP32，可以额外省出一份BF16参数的内存。

<figure style="text-align:center;">
  <img src=../source/images/param-reuse.png alt="示例图"/>
  <figcaption>
  模型训练流程
  </figcaption>
</figure>


**使用方式**

```
--use-optimizer-feature
--reuse-fp32-param
```




**异步激活offload**

为减少显存占用，通过将激活值拷贝到CPU上，在反向计算前再将激活值拷贝回HCU，从而降低峰值显存。对比重计算，通过将计算时间转换为异步H2D 和 异步D2H时间，可以以更少的性能损失，达到相同的显存降低效果。

<figure style="text-align:center;">
  <img src=../source/images/async-activation-offload.png alt="示例图"/>
  <figcaption>
  图1. 量化前all2all传输数据类型为bf16
  </figcaption>
</figure>



**参数选择**

```
必选:
--swap-attention
可选:
--swap-modules input_layernorm,self_attention,post_attention_norm
--specify-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```





**注意事项**

1. 在mcore模式下,要设置overlap_grad_reduce=True,te必须满足te>=2.5
2. 可以通过swap-modules控制做offload的模块,默认为self_attention,建议只开启self_attention
3. 可以通过specify-layers控制做offload的layer层

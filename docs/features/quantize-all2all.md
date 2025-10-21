### all2all量化通信
all2all量化通信，对all2all通信数据进行低精度表示，减少通信量。

如果不使用量化通信，all2all通信数据类型为bfloat16。
<figure style="text-align:center;">
  <img src=../source/images/quantize-all2all.png alt="示例图"/>
  <figcaption>
  图1. 量化前all2all传输数据类型为bf16
  </figcaption>
</figure>

本项目支持将数据量化为int8或int4类型，然后进行all2all通信。
<figure style="text-align:center;">
  <img src=../source/images/quantize-all2all-2.png alt="示例图"/>
  <figcaption>
  图2. 量化后all2all传输数据类型为int8或int4
  </figcaption>
</figure>

如需使用该特性，需要在启动脚本中额外添加以下参数：

**必需参数**

```
--use-quantize-comm
```

**可选参数**
```
--quant-comm-bits 4          # 量化精度, 可取4/8，分别将数据量化为int4/int8，缺省值为8；
--quant-group-size 32        # 数据被分成大小为quant-group-size的组，每组应用特定的量化策略，有助于提高量化效果或保持模型性能。quant-comm-bits为4时，quant-group-size可取16或32，默认为32。quant-comm-bits为8时，quant-group-size可取64或128，默认为128。
```

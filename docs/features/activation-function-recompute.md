# 激活函数重计算

gelu激活函数会产生大量的数据，但本身的计算量很小。此时进行激活函数的重计算可能在性能劣化极少的代价下，减少内存占用。但在现有重计算框架下，如果对gelu激活函数模块做重计算，并不能节省gelu函数的输出。这是因为在反向时，模块A所需要的gelu输出的激活值，会早于gelu激活函数模块的重计算流程，所以前向必须保留激活函数的输出，导致激活函数的输出并不能节省下来。



### 解决方法

#### 图一重计算与反向绑定

<figure style="text-align:center;">
  <img src=../source/images/sources_images_activation_function_a.png alt="示例图"/>
  <figcaption>
  图一
  </figcaption>
</figure>

#### 图二灵活插入重计算

<figure style="text-align:center;">
  <img src=../source/images/sources_images_activation_function_b.png alt="示例图"/>
  <figcaption>
  图二
  </figcaption>
</figure>

### 使用方法

```python
# 必选
--recompute-activation-function

# 可选, 指定激活函数重计算的层数
--recompute-activation-function-num-layers ${num}


```

##### 说明
激活函数重计算可以与全重计算同时开启:

1. 同时开启时, 仅支持 --recompute-method 为 block

2. 同时开启时, 会按照指定的全重计算和激活函数重计算的层数做各自类型的重计算, 即不会有一层既做全重计算又做激活函数重计算.

执行优先级是先计算全重计算层, 后计算激活函数重计算层. 在流水线并行未开启的情况下, 全重计算层数和激活函数重计算层数之和应该等于总层数.

暂不兼容自适应重计算特性.

### 使用效果

| 模型名称  | 模型参数                    | 设备数    | 内存收益 | 性能下降         |
| --------- | --------------------------- | --------- | -------- | ---------------- |
| llama2-7B | seq-length 4096，TP 1，PP 2 | 8卡(单机) | 2.6G(4%) | 210.5->206.5(2%) |


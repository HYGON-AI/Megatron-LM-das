# 融合线性交叉熵（HCU Linear CE）

在大模型训练中，输出层先将隐藏状态投影到词表空间，再计算交叉熵损失。常规实现会生成形状为 `[token 数量, 词表大小]` 的 logits 张量；当序列较长或词表较大时，这部分中间结果会占用较多显存。

HCU Linear CE 将输出层的线性投影与交叉熵计算融合，沿词表维度分块计算 logits，并通过在线 Softmax 累积损失计算所需的统计量。反向传播时，分块重算局部 logits 及其梯度，并计算隐藏状态梯度 `dHidden` 和输出权重梯度 `dWeight`，避免在全局显存中保存完整的 logits 和 dLogits 张量，从而降低输出层与损失计算的中间显存开销。

该特性在 HCU GPT 的 `gpt_model_postprocess` 中直接接入，复用官方交叉熵融合参数，无需修改训练循环。开启后，输出阶段直接将隐藏状态、输出权重、labels 和 loss_mask 传给融合算子，跳过单独的输出投影与交叉熵计算。具体显存收益和训练性能取决于模型规模、序列长度、词表大小及设备配置。

### 使用方法

使用前，需要在训练环境中安装与 HCU 架构及 PyTorch 版本匹配的 Linear CE 算子包。安装和动态库加载方式见 [HCU Linear CE 算子说明](../../hcu_megatron/core/fusions/linear_cross_entropy/README.md#安装和加载)。若需要指定动态库，可设置：

```bash
# 替换为实际的库文件路径；已安装匹配的算子包时无需设置。
export HCU_LINEAR_CE_EXTENSION_PATH=/path/to/libhcu_linear_ce_gfx936.so
```

在传给 `pretrain_gpt.py` 的训练参数中加入：

```bash
--cross-entropy-loss-fusion \
--cross-entropy-fusion-impl linear
```
注：若报错不支持kernel family，需要加上参数：
```bash
--make-vocab-size-divisible-by 256
```

当前实现要求使用 BF16，且张量并行度（TP）和上下文并行度（CP）均为 1。首次验证建议使用流水线并行度（PP）为 1 的配置：

```bash
--cross-entropy-loss-fusion \
--cross-entropy-fusion-impl linear \
--bf16 \
--tensor-model-parallel-size 1 \
--context-parallel-size 1 \
--pipeline-model-parallel-size 1
```

同时移除训练参数中的 `--sequence-parallel`。若使用 `examples/llama2/train_llama2_7B.sh`，应将融合开关加入 `TRAINING_ARGS` 数组，并调整并行配置；仅在 `run_llama.sh` 命令末尾追加该开关不会自动传递给训练入口。

可通过以下环境变量查看 native 算子的执行与选优日志：

```bash
export HCU_LINEAR_CE_LOG_LEVEL=1
```

### 注意事项
1. **数据类型与并行配置**：隐藏状态和输出权重必须使用 BF16；TP=1、CP=1，关闭序列并行（SP）。当前算子接口支持数据并行（DP）；流水线并行组合尚未完成验证，首次使用建议 PP=1，并根据显存容量调整模型、序列长度和 batch size。
2. **不兼容特性**：不支持 MTP、MuP、`--enable-vocab-parallel`、延迟 embedding 权重梯度计算及 packed sequence；不能与其他自定义 `output_processor` 同时使用。
3. **损失掩码**：训练时必须提供与 labels 形状一致的二值 `loss_mask`，其中 1 表示参与损失计算，0 表示忽略；不支持任意加权掩码。
4. **损失语义**：native 算子返回有效 token 的平均交叉熵。训练适配器将其缩放并展开为 `[batch, sequence]` 张量，以保持当前 `sum(loss * loss_mask)` 聚合下的总损失和梯度语义。展开后的值不是真实的逐token损失，不能直接用于逐token指标或其他损失聚合方式。
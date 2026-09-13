### MegaMoE

MegaMoE 使用 `Primus-Turbo` 提供的融合算子替换 Megatron 原生
`MoELayer`，将 expert all-to-all 与 grouped GEMM 融合。

启动参数：

```shell
--enable-mega-moe
--bf16
--tensor-model-parallel-size 1
--expert-model-parallel-size 8
```

`--use-turbo-mega-moe` 是 `--enable-mega-moe` 的兼容别名。MegaMoE 要求
TP=1、EP 为 2 到 8、BF16 参数、dropless SwiGLU expert 且 expert linear
不带 bias。当前不支持 MoE latent projection、sinkhorn 路由和 meta-device
初始化。

默认 expert 计算精度为 BF16。底层包支持 MXFP8 时，可增加：

```shell
--turbo-mega-moe-precision mxfp8
```

MXFP8 模式仍保存 BF16 参数和 checkpoint；适配层会在每个训练 step 后推进
权重量化缓存 generation，确保 optimizer 更新后的 expert 权重重新量化。

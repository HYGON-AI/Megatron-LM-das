# Dcu Megatron


本项目通过替换megatron的函数或类，引入新的特性或者实现更好的性能。替换的函数或类注册在dcu_megatron/adaptor/megatron_adaptor.py。

+ 支持函数替换  

```
from ..core.distributed.finalize_model_grads import _allreduce_word_embedding_grads
MegatronAdaptation.register('megatron.core.distributed.finalize_model_grads._allreduce_word_embedding_grads',
                            _allreduce_word_embedding_grads)
```
以上代码将megatron的_allreduce_word_embedding_grads替换为自定义的_allreduce_word_embedding_grads。

+ 支持类替换

```
from ..core.transformer.transformer_config import TransformerConfig, MLATransformerConfig

# Transformer config
MegatronAdaptation.register('megatron.core.transformer.transformer_config.TransformerConfig',
                            TransformerConfig)
MegatronAdaptation.register('megatron.core.transformer.transformer_config.MLATransformerConfig',
                            MLATransformerConfig)                   
```
以上代码将megatron的TransformerConfig和MLATransformerConfig替换为自定义类型。

+ 支持基类替换
```
from megatron.core.extensions.transformer_engine import TEGroupedLinear

if int(os.getenv("GROUPED_GEMM_BatchLinear", '0')):
    TEGroupedLinear.__bases__ = (te.pytorch.BatchLinear,)
```
以上代码将TEGroupedLinear的父类替换为te.pytorch.BatchLinear。

+ 支持增加修饰器
```
MegatronAdaptation.register('megatron.core.transformer.moe.moe_utils.permute',
                            torch.compile(mode='max-autotune-no-cudagraphs'),
                            apply_wrapper=True)
MegatronAdaptation.register('megatron.core.transformer.moe.moe_utils.unpermute',
                            torch.compile(mode='max-autotune-no-cudagraphs'),
                            apply_wrapper=True)
```
以上代码对permute和unpermute函数增加修饰器，效果如下:
```
@torch.compile(mode='max-autotune-no-cudagraphs')
def permute(
    tokens,
    routing_map,
    num_out_tokens: Optional[int] = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):

@torch.compile(mode='max-autotune-no-cudagraphs')
def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
    probs: torch.Tensor = None,
    routing_map: torch.Tensor = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
```

### 项目支持使用[flux kernel](http://10.6.10.68/dcutoolkit/deeplearing/flux)
在tp场景下，用户可以选择使用flux通算融合算子，获得更好的训练和推理性能。项目通过替换transformer engine方法集成flux，使用时需要设置环境变量USE_FLUX_OVERLAP=1，并设置transformer-impl为transformer_engine。


### 使用方式
在使用时，进入到examples目录下，有相关模型执行脚本，所用数据集请自行下载：https://r0ddbu55vzx.feishu.cn/drive/folder/ZxHHfCoX4lg75td2hTqcmiAin3g
```
examples
├── gpt3
│   ├── hostfile_gpt_567B
│   ├── README.md
│   ├── run_gpt_567B_1nodes.sh
│   ├── run_gpt_567B_multinodes.sh
│   ├── topo-input.xml
│   ├── train_gpt_567B_1nodes.sh
│   └── train_gpt_567B_multinodes.sh
└── mixtral
    ├── hostfile_mixtral_8x22B
    ├── hostfile_mixtral_8x7B
    ├── README.md
    ├── run_mixtral_8x22B_1nodes.sh
    ├── run_mixtral_8x22B_multinodes.sh
    ├── run_mixtral_8x7B_1nodes.sh
    ├── run_mixtral_8x7B_multinodes.sh
    ├── topo-input.xml
    ├── train_mixtral_8x22B_1nodes.sh
    ├── train_mixtral_8x22B_multinodes.sh
    ├── train_mixtral_8x7B_1nodes.sh
    └── train_mixtral_8x7B_multinodes.sh
```


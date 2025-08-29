# Dcu Megatron

## 项目介绍
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

### 项目支持内存缓存ckpt
+ 在大模型训练过程中如果需要使用内存缓存ckpt提升性能，需要在脚本中加入如下参数：
```
--use-ckpt-memory-cache
```
+ 注意事项:
1. 开启内存缓存ckpt功能后还需要一个python包和启动hyckptd进程，联系赵煜要
2. pip install hyckpt-1.0.1-py3-none-any.whl  安装到conda环境中
3. 启动hyckptd 进程 mpirun -pernode -hostfile 主机名文件 hyckptd可执行程序 --log 日志文件路径

### 项目支持使用[flux kernel](http://10.6.10.68/dcutoolkit/deeplearing/flux)
在tp场景下，用户可以选择使用flux通算融合算子，获得更好的训练和推理性能。项目通过替换transformer engine方法集成flux，使用该特性，需要启动脚本中加入如下参数:
```
--parallel-linear-impl flux
```

### 交错式1f1b流水线支持[moe a2a通信计算overlap](https://mp.weixin.qq.com/s?__biz=MzU2NzkyMzUxMw==&mid=2247550702&idx=2&sn=9f6bb8ea72475aa833bfd73718f03530&chksm=fdb928e884341e81762eeaffbc3d00a3023e4543001b5448f259977b8bf0e4603448db75360e&mpshare=1&scene=1&srcid=0306blxvLHplbcAOqnznmXiQ&sharer_shareinfo=962faa39bc50b5544c96cf846186f076&sharer_shareinfo_first=962faa39bc50b5544c96cf846186f076&version=4.1.20.70286&platform=mac#rd)
+ 项目支持moe a2a 通算overlap。如果需要使用该特性，需要启动脚本中加入如下两个参数:
```
--schedule-method interleaved_1f1b
--overlap-moe-expert-parallel-comm
```
+ 项目支持通过delay-wgrad-compute进行dw拆分，用于实现更好的overlap。当前从测试结果看，开启delay-wgrad-compute，收益甚微，待进一步优化。

### 项目支持dualpipev
+ 项目支持dualpipev。如果需要使用该特性，需要启动脚本中加入如下参数:
```
--schedule-method dualpipev
--delay-wgrad-compute
```
+ dualpipev支持moe a2a overlap，如果需要overlap，额外增加两个参数
```
--overlap-moe-expert-parallel-comm
```


### 项目支持量化通信
+ 项目支持量化通信，对all-to-all通信数据进行低精度表示，减少通信量。如果需要使用该特性，需要启动脚本中加入如下参数：
```
--use-quantize-comm
```

### 项目支持参数副本复用
+ 项目支持参数副本复用，主要在BF16的训练场景使用，前向计算开始前，将FP32的参数保存转换为BF16并保存Residual，优化器更新前基于BF16和Residual恢复FP32参数并进行更新。如果需要使用该特性，需要启动脚本中加入如下参数:
```
--use-optimizer-feature
--reuse-fp32-param
```

### 项目支持edgc
+ 项目支持PowerSGD低秩分解与误差反馈机制，能够根据训练阶段、系统环境及各流水线层的梯度熵变化，动态调整梯度压缩率。在显著降低通信开销的同时，有效保留关键梯度信息，兼顾训练效率与模型收敛精度。如果需要使用该特性，需要启动脚本中加入如下参数:
```
--enable-dynamic-grad-comp
--rank-adjust-window-size 1000
```

### 项目支持激活值offload
+ 在模型规格较大时，我们通常使用重计算降低显存占用，但是性能下降较严重，这里我们通过在前向计算时将激活值offload到CPU，在反向计算时，再将激活值copy到dcu来减少显存占用。需要在启动脚本中加入以下参数
```
必选:
--swap-attention
可选:
--swap-modules input_layernorm,self_attention,post_attention_norm
--specify-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```
+ 注意事项:
1. 在mcore模式下,要设置overlap_grad_reduce=True,te必须满足te>=2.5
2. 可以通过swap-modules控制做swap的模块,默认为self_attention,建议只开启self_attention
3. 可以通过specify-layers控制做swap的layer层

## 使用方式

### 项目下载

分为2种，git方式或离线方式

1、git方式下载

```shell
git clone -b core_v0.12.0 --recurse-submodules http://10.16.6.30/dcutoolkit/deeplearing/dcu_megatron.git 或
git clone -b core_v0.12.0 --recurse-submodules http://112.11.119.99//dcutoolkit/deeplearing/dcu_megatron.git
```

2、离线下载

2.1 离线下载该仓库的离线代码包

2.2 点击Megatron-LM@版本号, 下载对应版本的Megatron-LM离线代码包

2.3 将Megatron-LM离线代码包解压到dcu_megatron目录下的Megatron-LM目录



### 项目使用
在使用时，进入到examples目录下，有相关模型执行脚本，所用数据集请自行下载：https://r0ddbu55vzx.feishu.cn/drive/folder/ZxHHfCoX4lg75td2hTqcmiAin3g
```
examples/
├── deepseek_v3
├── gpt3
├── llama
├── mixtral
└── qwen
```

### 节点筛查(此次检查是基于GPT-MOE 567B模型单机参数)

```shell
1、到check_nodes目录下，将要筛查的节点写入clushnode文件
2、bash clush.sh，检查环境基本情况，如显存、内存等是否已释放
3、打开check_nodes.sh，将基本环境变量补齐或做相应修改
4、bash run_check.sh 1/4，进行单机或者四机的节点筛查 # 当前只支持单机和四机筛查
```

### 版本依赖
torch >= 2.6.0

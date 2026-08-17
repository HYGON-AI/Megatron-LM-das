# Hcu Megatron

## 项目介绍
本项目通过替换megatron的函数或类，引入新的特性或者实现更好的性能。替换的函数或类注册在hcu_megatron/adaptor/megatron_adaptor.py。

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

### 交错式1f1b流水线支持[moe a2a通信计算overlap](https://mp.weixin.qq.com/s?__biz=MzU2NzkyMzUxMw==&mid=2247550702&idx=2&sn=9f6bb8ea72475aa833bfd73718f03530&chksm=fdb928e884341e81762eeaffbc3d00a3023e4543001b5448f259977b8bf0e4603448db75360e&mpshare=1&scene=1&srcid=0306blxvLHplbcAOqnznmXiQ&sharer_shareinfo=962faa39bc50b5544c96cf846186f076&sharer_shareinfo_first=962faa39bc50b5544c96cf846186f076&version=4.1.20.70286&platform=mac#rd)
+ 项目支持moe a2a 通算overlap，实现计算掩盖全部或部分a2a通信。具体见[流水线并行](./docs/features/pipeline-parallel.md)

### 1f1b流水线支持拆分cooldown阶段梯度计算
+ 项目支持对1f1b流水线cooldown阶段的参数/激活值梯度计算进行拆分，提升小batch情形下的训练性能。具体使用说明见[流水线并行](./docs/features/pipeline-parallel.md)

### 项目支持dualpipev
+ 项目支持dualpipev。具体使用说明见[流水线并行](./docs/features/pipeline-parallel.md)

### 项目支持ZB-H1流水线
+ 项目支持ZB-H1流水线调度，可提升小batch情形下训练性能。具体使用说明见[流水线并行](./docs/features/pipeline-parallel.md)


### 项目支持量化通信
+ 项目支持量化通信，对all-to-all通信数据进行低精度表示，减少通信量。具体见[all2all量化通信](./docs/features/quantize-all2all.md)


### 项目支持参数副本复用
+ 项目支持参数副本复用，主要在BF16的训练场景使用，前向计算开始前，将FP32的参数保存转换为BF16并保存Residual，优化器更新前基于BF16和Residual恢复FP32参数并进行更新。具体见[参数副本复用](./docs/features/param-reuse.md)

### 项目支持edgc
+ 项目支持PowerSGD低秩分解与误差反馈机制，能够根据训练阶段、系统环境及各流水线层的梯度熵变化，动态调整梯度压缩率。在显著降低通信开销的同时，有效保留关键梯度信息，兼顾训练效率与模型收敛精度。具体见[edgc](./docs/features/edgc.md)介绍

### 项目支持激活值offload
+ 在模型规格较大时，我们通常使用重计算降低显存占用，但是性能下降较严重，这里我们通过在前向计算时将激活值offload到CPU，在反向计算时，再将激活值copy到hcu来减少显存占用。具体见[激活值offload](./docs/features/async-activation-offload.md)

### 项目支持指定重计算层
+ megatron支持对所有transformer/mtp层进行重计算，该情形下模型训练显存占用小，但是训练性能通常较差。为了在显存满足要求的同时，提高模型训练性能，hcu megatron支持对指定tranformer/mtp层进行重计算。使用该重计算方式，需要开启以下参数：
```
--recompute-granularity full
--recompute-layer-ids 0 4 8 12   # 对第0、4、8和12 transformer层进行重计算（从0开始对tranformer层进行编号）
--recompute-mtp-layer-ids 0    # 对第0 mtp层进行重计算（从0开始对mtp层进行编号）
```
+ 注意事项：
1. recompute-layer-ids的给定值范围为[0, N<sub>total_layers</sub>-1]，N<sub>total_layers</sub>为模型中transformer层数;
2. recompute-mtp-layer-ids的给定值范围为[0, N<sub>total_mtp_layers</sub>-1]，N<sub>total_mtp_layers</sub>为模型中mtp层数;
3. recompute-layer-ids/recompute-mtp-layer-ids允许同时设置，或只设置一个。如不设置，相应网络层不进行重计算；
4. 不允许设置recompute-method参数。




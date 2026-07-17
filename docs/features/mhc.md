# mhc

hcu_megatron 0.17.0增加了mhc实现，共提供了torch原生实现和tile_kernel实现两种方式，torch原生实现支持重计算，tile_kernel不支持重计算。



### 使用方式

```python
# 设置一, 走mhc的torch普通实现
--enable-hyper-connections

# 设置二, Tilekernel实现
--enable-hyper-connections
--mhc-use-tilekernels
--mhc-fuse-h-post-compute
--mhc-tau 1.0
--mhc-log-amax-per-step 20

--------------------------------------------------------------------------
使用Tilekernel实现前请完成以下操作:
1.获取hcu版本tile-kernels,并安装;或将代码放到指定目录手动设置
假设tile_kernels目录在Megatron-LM/megatron目录下，请设置
export PYTHONPATH=/mnt/hcu_megatron/Megatron-LM/megatron/:$PYTHONPATH

2.查找环境的python目录,设置z3/lib路径
export LD_LIBRARY_PATH="/usr/local/lib/python3.11/site-packages/z3/lib:$LD_LIBRARY_PATH"
---------------------------------------------------------------



# 可选,开启mhc重计算
--recompute-granularity selective
--recompute-modules layernorm core_attn mhc


# 其他可用参数
group.add_argument('--enable-hyper-connections', action='store_true', default=False,
                    help='use gathered input of AGKernel for wgrad computation')
group.add_argument('--mhc-use-tilekernels', action='store_true', default=False,
                    help='Whether to transpose weight when using flux kernel')
group.add_argument('--num-residual-streams', type=int, default=4,
                    help='Number of residual streams (n in paper).')
group.add_argument('--mhc-sinkhorn-iterations', type=int, default=1,
                    help='Number of Sinkhorn-Knopp iterations for doubly stochastic projection..')
group.add_argument('--mhc-recompute-layer-num', type=int, default=1,
                    help='Number of layers per MHC recompute block.'
                         'When set, every `mhc_recompute_layer_num` layers form a recompute block. The last layer'
                         'in each recompute block (i.e., layer_number % mhc_recompute_layer_num == 0 or the final'
                         'layer in the transformer block) will:'
                         '- NOT checkpoint its final MLP BDA'
                         '- Register the unified recompute hook on its MLP BDA output'
                         '- A new CheckpointManager is created for subsequent layers'                    
                         'If None, all layers in the transformer block share a single recompute block.'
                         'Must be a positive integer when set.')
group.add_argument('--mhc-init-gating-factor', type=float, default=0.01,
                    help='Initial value of Gating Factor (alpha in paper).')
group.add_argument('--mhc-expand-emb', action='store_true', default=False,
                    help='Whether to expand the embedding dimension for mHC.')
group.add_argument('--mhc-tau', type=float, default=0.05,
                    help='Number of residual streams (n in paper).')
group.add_argument('--mhc-lite', action='store_true', default=False,
                    help='Number of residual streams (n in paper).')
group.add_argument('--use-vwn', action='store_true', default=False,
                    help='If true, use vwn.')
group.add_argument('--mhc-hres-vwnstyle', action='store_true', default=False,
                    help='If true, use mhc-hres-vwnstyle.')
group.add_argument('--use-mhc-svd', action='store_true', default=False,
                    help='If true, use use-mhc-svd.')
group.add_argument('--mhc-fuse-h-post-compute', action='store_true', default=False,
                    help='If true, use mhc-fuse-h-post-compute.')
group.add_argument('--mhc-log-amax-per-step', type=int, default=1,
                    help='mhc-log-amax-per-step.')
group.add_argument('--mhc-fix-muons', action='store_true', default=False,
                    help='mhc-fix-muons.')
group.add_argument('--mhc-fuse-aggregate-compute', action='store_true', default=False,
                    help='mhc-fuse-aggregate-compute.')
group.add_argument('--mhc-init-hpre-use-module-layer', action='store_true', default=False,
                    help='If true, use module-level layer index (2*layer + is_mlp) for h_pre initialization.')
```



### 注意事项

1. torch原生实现支持重计算，tile_kernel不支持重计算。
2. mhc-sinkhorn-iterations参数设置小于等于10。


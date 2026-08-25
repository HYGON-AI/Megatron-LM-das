## 背景

[Megatron-Bridge](https://github.com/NVIDIA/Megatron-Bridge) 在 HuggingFace 和 Megatron 两种模型格式之间做双向转换。启用后不必为每个新模型手写 Megatron 侧的模型定义和权重映射：

- **加载**：直接读 HF 目录（`config.json` + safetensors），按 TP/PP/EP 切分后灌进 Megatron 模型，用 `--load-weights` 控制；
- **保存**：把训练中的 Megatron 模型导回 HF 格式，产物可直接被 `from_pretrained()` 加载，用 `--save-hf-weights` 控制；
- **模型覆盖面**：bridge 内置了 35 种架构的映射（Llama / Mistral / Qwen2 / Qwen3 / Qwen3-VL / Qwen2.5-VL / DeepSeek-V2/V3 / GLM4 / GLM-4V / Gemma2/3 / GPT-OSS 等），LLM 和 VLM 都支持。
  当前已注册的完整列表可以用下面的命令查看：

  ```sh
  grep -rn '@MegatronModelBridge.register_bridge' -A 5 \
      3rdparty/Megatron-Bridge/src/megatron/bridge/models/ | grep 'source='
  ```

模型形状（层数、hidden size、head 数、norm、激活函数等）以 HF 的 `config.json` 为准，命令行只负责分布式并行、batch、优化器等运行时参数。

## 用法

### 基本开关

```sh
--use-bridge                                # 启用 bridge
--bridge-hf-model ${TOKENIZER_MODEL_PATH}   # HF 模型目录（必填）
--load-weights                              # 从 HF 权重初始化，不加则随机初始化
```

`--bridge-hf-model` 指向的是标准 HF 模型目录（含 `config.json`、tokenizer、safetensors），
通常和 `--tokenizer-model` 用同一个路径。

### 导出 HF 格式权重

```sh
--save-hf-weights                # 保存 ckpt 时同步导出一份 HF 格式
--save-hf-weights-distributed    # 各 rank 分担写 safetensors 分片（大模型建议开）
```

导出时机跟随 `--save-interval`，产物落在 Megatron ckpt 的同一个 iteration 目录下：

```
${save}/
└── iter_0001000/
    ├── mp_rank_00/            # Megatron 原生 ckpt
    ├── mp_rank_01/
    └── hf/                    # HF 格式，可直接 from_pretrained()
        ├── config.json
        ├── tokenizer.json
        ├── model-00001-of-0000N.safetensors
        └── model.safetensors.index.json
```

约束（在参数校验阶段就会报错）：

- `--save-hf-weights` 必须搭配 `--use-bridge` 和 `--save`；
- `--save-hf-weights-distributed` 必须搭配 `--save-hf-weights`；
- 非持久化 ckpt（`--non-persistent-save-interval`）不会触发导出，那是本地快速重启用的临时状态。

导出是**集合操作**，所有 rank 都要参与：内部跨 TP/PP/EP 做 gather，默认只有 rank 0 落盘。
采用流式写入（逐张量 yield 且转到 CPU），不会把整个模型堆在显存里。

### VLM 相关

```sh
--bridge-language-model-only     # VLM 只构建语言模型，跳过 ViT / projector
--image-tokens-per-sample 1024   # 把视觉侧 FLOPs 计入吞吐统计
```

`--bridge-language-model-only` 对纯 LLM provider 是 no-op。开启后 provider 走
`provide_language_model` 路径，返回 `MCoreGPTModel`，同时会把 `mrope` 回退为标准 `rope`、
重新打开 SP scatter，以匹配纯文本的 2D position_ids。

### 完整示例

```sh
# LLM：Qwen3-8B，从 HF 权重续训并导出 HF 格式
python pretrain_gpt.py \
    --use-bridge \
    --bridge-hf-model /path/to/Qwen3-8B \
    --load-weights \
    --save ./ckpt --save-interval 1000 \
    --save-hf-weights \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    ...

# VLM：Qwen3-VL-8B，大模型开分布式导出
python pretrain_vlm.py \
    --use-bridge \
    --bridge-hf-model /path/to/Qwen3-VL-8B-Instruct \
    --load-weights \
    --save ./ckpt --save-interval 1000 \
    --save-hf-weights --save-hf-weights-distributed \
    ...
```

导出的权重直接加载：

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("./ckpt/iter_0001000/hf")
```

### 常见问题

**PP>1 导出报 pickle 相关错误**
导出 QKV 时需要跨 PP 广播模型 config，而 `broadcast_object_list` 内部使用 pickle。
hcu 在 `transformer_config.py` 里会动态生成 config 子类，必须保证它可跨进程序列化
（见 `_make_picklable_dataclass`）。PP=1 不发生广播，所以该类问题只在 PP>1 暴露。

**报 "architecture not supported"**
bridge 靠 `config.json` 的 `architectures` 字段做分发，要求名字以 `ForCausalLM` 或
`ForConditionalGeneration` 结尾（另有少量特例白名单）。若该架构尚未注册 bridge，
需要新增一个 `MegatronModelBridge` 子类。

**磁盘占用**
每个 save 点都会多存一份完整的 HF 权重（bf16 下约等于参数量×2 字节）。
配合 `--save-interval` 评估磁盘余量，必要时调大导出间隔。

## 调用链

bridge 的核心是**用 HF `config.json` 里的 `architectures` 字段做 dispatch**，
找到对应的 Bridge 实现，进而创建 Provider 和 Megatron 模型。所有模型走的都是同一条链路，
区别只在第 ② 步分发到哪个 Bridge。

### 加载（模型构建）

```
setup_model_and_optimizer()                                  [hcu_megatron/training/training.py]
  │
  ├─① AutoBridge.from_hf_pretrained(args.bridge_hf_model)    [auto_bridge.py]
  │     读 config.json → architectures: ["<Arch>"]
  │     校验后缀是否为 ForCausalLM / ForConditionalGeneration   [AutoBridge.supports()]
  │     返回 AutoBridge(hf_pretrained)
  │     注意：此处为惰性加载，只读 config，不把 HF 权重载入内存
  │
  ├─② bridge.to_megatron_provider(load_weights, hf_path)      [auto_bridge.py]
  │     │
  │     ├─ _model_bridge 属性
  │     │    _causal_lm_architecture → "<Arch>"
  │     │    get_model_bridge("<Arch>", hf_config) → 分发到已注册的 Bridge
  │     │
  │     │    注册表由装饰器构成，例如：
  │     │      LlamaForCausalLM                → GPTModel      [llama_bridge.py]
  │     │      Qwen3ForCausalLM                → GPTModel      [qwen3_bridge.py]
  │     │      Qwen3VLForConditionalGeneration → Qwen3VLModel  [qwen3_vl_bridge.py]
  │     │      DeepseekV3ForCausalLM           → GPTModel      [deepseek_v3_bridge.py]
  │     │      ...（共 35 种）
  │     │
  │     ├─ Bridge.provider_bridge(hf_pretrained)
  │     │    从 HF config 读 num_layers / hidden_size / num_query_groups 等
  │     │    → <Model>Provider(**kwargs)（TransformerConfig 子类）
  │     │    VLM 还会注入 vision_config / mrope / 特殊 token id
  │     │
  │     └─ load_weights=True 时注册 pre-wrap hook：
  │          load_weights_hf_to_megatron，在模型包裹前把 HF 权重按并行切分灌入
  │
  ├─③ 用 CLI args 覆盖 provider 的运行时参数，再 provider.finalize()
  │     只覆盖命令行显式指定的字段；模型形状类字段保持 HF 所有，
  │     否则会与 checkpoint 结构不一致导致加载失败
  │
  └─④ provider.provide_distributed_model(wrap_with_ddp, ddp_config)
        按 TP/PP/CP/EP 切分并包裹 DDP → Megatron 格式模型
```

### 保存（导出 HF）

```
save_checkpoint_and_time_wrapper()            [hcu_megatron/training/training.py]
  │
  ├─ 先完成 Megatron 原生 ckpt 保存
  │
  └─ save_hf_checkpoint(iteration, model)     [hcu_megatron/training/training.py]
        │  复用建模时缓存的 bridge 对象（get_bridge()），无需重新读 HF 模型
        │
        └─ bridge.save_hf_pretrained(model, path)      [auto_bridge.py]
             ├─ rank 0 写 config.json / tokenizer 等 artifacts
             └─ save_hf_weights → stream_weights_megatron_to_hf
                  跨 TP/PP/EP gather，逐张量转换为 HF 命名后流式写 safetensors
                  模型以 DDP 包裹的 chunk list 传入即可，内部会自行 unwrap
```

### 新增模型支持

若某架构尚未注册，实现一个 `MegatronModelBridge` 子类即可接入：

```python
@MegatronModelBridge.register_bridge(source=XxxForCausalLM, target=GPTModel, model_type="xxx")
class XxxBridge(MegatronModelBridge):
    def provider_bridge(self, hf_pretrained):
        """HF config → Megatron Provider"""

    def mapping_registry(self):
        """HF 权重名 ↔ Megatron 权重名的映射规则"""
```

`mapping_registry` 同时服务加载和导出两个方向，写一次即可双向使用。

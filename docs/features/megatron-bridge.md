## 背景
基于megatron-bridge可以通过megatron训练更多的模型

## 用法
使用下面的三个命令即可控制bridge使用
```sh
--use-bridge
--bridge-hf-model ${TOKENIZER_MODEL_PATH}
# --load-weights
```
## 完整调用链(qwen3.5vl)

setup_model_and_optimizer()                          [training.py:462]
  │
  ├─① AutoBridge.from_hf_pretrained(args.bridge_hf_model)  [auto_bridge.py:212]
  │     │  加载 HF 模型，读 config.json 拿到 architectures: ["Qwen3_5VLForConditionalGeneration"]
  │     │  校验 → supports() → 检查后缀是否为 "ForCausalLM" / "ForConditionalGeneration"
  │     │  返回 AutoBridge(hf_pretrained)
  │     │
  ├─② bridge.to_megatron_provider(load_weights=True, hf_path=...)  [auto_bridge.py:1118]
  │     │  provider_input = self._provider_bridge_input   → PreTrainedCausalLM
  │     │  provider = self._model_bridge.provider_bridge(provider_input)
  │     │      │
  │     │      └─ _model_bridge 属性 [auto_bridge.py:1299]
  │     │           self._causal_lm_architecture → "Qwen3_5ForConditionalGeneration"
  │     │           model_bridge.get_model_bridge("Qwen3_5ForConditionalGeneration", hf_config)
  │     │              │
  │     │              └─ 分发到已注册的 Bridge：          [qwen35_vl_bridge.py:434]
  │     │                   @MegatronModelBridge.register_bridge(
  │     │                       source="Qwen3_5ForConditionalGeneration",
  │     │                       target=Qwen3VLModel,
  │     │                       provider=Qwen35VLModelProvider,
  │     │                   )
  │     │                   class Qwen35VLBridge:
  │     │                       def provider_bridge(hf_pretrained):
  │     │                           ...从 HF config 读 num_layers/hidden_size 等
  │     │                           → Qwen35VLModelProvider(**kwargs)
  │     │                           → 注入 vision_config / mrope / token_ids
  │     │
  │     │  provider = Qwen35VLModelProvider(TransformerConfig 子类)
  │     │  return provider
  │     │
  │     └─ 回到 training.py:472, 用 CLI args 覆盖分布式并行参数
  │        provider.tensor_model_parallel_size = args.tensor_model_parallel_size  ...
  │        provider.finalize()
  │
  ├─③ ddp_config = DistributedDataParallelConfig(...)         [training.py:488-507]
  │
  └─④ model = provider.provide_distributed_model(wrap_with_ddp, ddp_config)
        │  [model_provider.py:107]
        │  根据 TP/PP/CP 参数切分并包裹模型
        │  → Qwen3VLModel (Megatron 格式模型)
        │
        └─ 返回给训练循环
关键桥接点：Qwen3_5ForConditionalGeneration 这个字符串是 HF config.json 里 architectures 字段的值，bridge 用它做 dispatch，找到 Qwen35VLBridge → 创建 Qwen35VLModelProvider → 最终构建 Qwen3VLModel。
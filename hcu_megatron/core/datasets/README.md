# `hcu_megatron/core/datasets`

VLM (Vision-Language Model) 训练数据加载模块。当前支持 Qwen2-VL / Qwen2.5-VL / Qwen3-VL / Gemma3-VL 四个家族。

本文档面向：**想要接入新 VLM 家族**、**修改现有数据流**、或**了解数据管线全貌**的人。

---

## 目录结构

```
core/datasets/
├── __init__.py
│
├── utils.py                     日志/迭代器/pad 通用小工具（print_rank_0, get_iterator, random_pad_list）
│
├── indexed_jsonl_dataset.py     JSONL 文件读取 + 按 domain 概率采样调度（底层）
├── mm_dataset.py                MultiModalDataset 基类：通用 __iter__ / 图片校验 / pad-shift
│
├── vlm_images.py                图片加载 + tar 缓存 + smart_resize + CP-rank pixel_values 重分布
├── vlm_tokenizer.py             Qwen / Gemma 家族的 tokenizer 属性注入
│
├── vlm_qwen.py                  QwenVLDataset（Qwen2/2.5/3-VL 三个 model_arch 共用）
├── vlm_gemma.py                 GemmaVLDataset
│
├── vlm_collator.py              DataCollatorForQwenVL / DataCollatorForGemmaVL
│
├── vlm_dataset.py               公开入口 (VLM_REGISTRY + build_train_valid_test_data_iter)
└── vlm_args.py                  CLI 参数注册 + JSON 数据配置解析
```

---

## 公开 API

`pretrain_vlm.py` 只依赖三个函数：

```python
from hcu_megatron.core.datasets.vlm_args import add_vlm_extra_args, parse_dataset_config
from hcu_megatron.core.datasets.vlm_dataset import build_train_valid_test_data_iter

parser = add_vlm_extra_args(parser)
args = parse_and_validate_args(...)
parse_dataset_config(args)                # 解析 --vlm-data-config-path JSON
train_iter, valid_iter, test_iter = build_train_valid_test_data_iter(args, tokenizer, ...)
```

不要绕过 `vlm_dataset.py` 直接 import 内部实现类，它们随时可能重构。

---

## 数据管线总览

```
JSONL 文件（磁盘）
    │
    │  按 domain 采样概率交错
    ▼
MultimodalIndexedJsonlDataset (indexed_jsonl_dataset.py)
    │  yield {"json_data": {...}, "domain_id": ..., ...}
    ▼
MultiModalDataset.__iter__ (mm_dataset.py)     ← 通用循环
    ① fetch_images (vlm_images.py) + 校验
    ② convert_conversations (<image> → 结构化 content)
    ③ _pre_convert_hook (家族钩子，如 Qwen3VL resize)
    ④ convert_example (家族实现)
    │
    ├─ Qwen 家族 (vlm_qwen.py)
    │    process_vision → apply_chat_template → padding_vision_token
    │    → tokenize → gen_label_mask → _pad_truncate_shift
    │
    └─ Gemma 家族 (vlm_gemma.py)
         process_vision → 手工拼接对话字符串 → tokenize
         → gen_label_mask → _pad_truncate_shift
    │
    ▼  yield 单样本 tensor dict
    │
    ▼
DataCollator{QwenVL,GemmaVL} (vlm_collator.py)
    │  pixel_values 拼接 / CP-rank 重分布 / loss_mask
    │  position_ids 不在此计算 —— 由 Bridge 侧的模型 forward 自己算
    │  （Qwen VL 每个 PP stage 都会重算 3D mRoPE，见"关于 position_ids"）
    │
    ▼  yield batch dict
    │
    ▼
pretrain_vlm.get_batch → forward_step → model(...)
```

### 关于 position_ids

hcu_megatron 的 collator **不生产 position_ids**。原因：

- **Qwen2.5-VL / Qwen3-VL / Qwen3.5-VL**（走 Bridge 路径）：模型 forward 内部会
  重新调用 `get_rope_index` 计算 3D mRoPE，每个 PP stage 都算一次。参见
  `3rparty/Megatron-Bridge/src/megatron/bridge/models/qwen_vl/`：
    * `modeling_qwen25_vl.py`：forward 里 unconditional 调 `self.get_rope_index(...)`
    * `modelling_qwen3_vl/model.py`：forward 里 `if position_ids is None: get_rope_index(...)`
    * `qwen3_vl_step.py`：主动 `forward_args["position_ids"] = None`
  也就是说 dataset 侧算出来的 position_ids 会被 Bridge 直接丢弃。
- **Gemma3-VL**：模型内部自算 RoPE，dataset 也不需要提供。

如果未来接入不使用 Bridge、也不在模型内自算 position_ids 的家族，可以在 collator
中重新加入 position_ids 计算，或者要求 Bridge/model 忽略 dataset 传入的值。

---

## 关键抽象

### 1) `BaseIndexedJsonlDataset` (indexed_jsonl_dataset.py)

按 `domain_probabilities` 生成一个长度为 `global_batch_size` 的调度表：

```
gbs=16, probs=[0.5, 0.3, 0.2]
→ [0,0,0,0,0,0,0,0, 1,1,1,1,1, 2,2,2]
```

每步消费一个 slot，长期分布与 probability 一致；然后按 `dp_rank` 切片。

domain 数据耗尽时自动进入下一个 epoch（in-session 计数，不持久化）。

**子类需要实现 `__iter__`。目前只有 `MultimodalIndexedJsonlDataset` 一个实现。**

### 2) `MultiModalDataset` (mm_dataset.py)

家族数据集的**基类**。承载所有家族共享的逻辑：

- 通用 `__iter__` 循环（取样 → 加载图片 → 校验 → hook → convert_example）
- `_pre_convert_hook(imgs, json_data)` — 家族钩子（默认 no-op）
- `_pad_truncate_shift(...)` — 自回归 pad/truncate/shift，参数化 Qwen vs Gemma 的差异
- 默认 `gen_label_mask` — Qwen 走 chat_template；Gemma 覆盖它

**子类必须实现：** `process_vision` / `convert_example`

### 3) `VLM_REGISTRY` (vlm_dataset.py)

家族分发的注册表。每个 model_arch 对应一个 `VLMFamily` 条目：

```python
@dataclass(frozen=True)
class VLMFamily:
    dataset_cls:       MultiModalDataset 子类
    tokenizer_setup:   Callable(tokenizer, args)
    build_collator:    Callable(args, tokenizer) -> collator
    build_processor:   Callable(args) -> AutoProcessor
    verify_hf_config:  Optional callable (Qwen3VL 用)
```

`build_train_valid_test_data_iter(args, ...)` 只做一件事：**根据 `args.model_arch` 从 `VLM_REGISTRY` 拿条目，按条目组装管线**。所有 `if model_arch == ...` 分支收敛到这一层。

---

## 输入数据格式

### JSONL 样本

每行一个 JSON 对象：

```json
{
    "conversations": [
        {"role": "user",      "content": "挂在交通灯杆上的是什么？<image>"},
        {"role": "assistant", "content": "一个绿色的街牌挂在交通灯杆上。"}
    ],
    "images": [
        {"image_path": "7_0.png"}
    ]
}
```

约束：
- `conversations` 至少 2 轮（`len(...) > 1`），最后一轮的 role 用于 loss target
- 可以没有图片；如果有图片则每张都必须能正常解码
- `images[i]` 可以只写 `image_path`（绝对路径），也可以指向 tar：
  ```json
  {"image_path": "img.png", "tar_name": "batch.tar", "offset": 12345, "size": 6789}
  ```

### `--vlm-data-config-path` JSON

```json
{
    "train_data_infos": {
        "math":  {"path": "/data/math_dir",  "probability": 0.5},
        "chat":  {"path": "/data/chat_dir",  "probability": 0.5}
    },
    "eval_data_infos": {
        "math_eval": {"path": "/data/math_eval_dir"}
    }
}
```

- `path` 可以是目录（读目录下所有 `*.jsonl`）或单个 `.jsonl` 文件
- train 部分 `probability` 必填；eval 部分不需要
- domain 会按 `path` 字典序排序以稳定 `domain_id`（保证多次运行 domain 编号一致）

---

## CLI 参数

```
--model-arch                 qwen2vl | qwen2.5vl | qwen3vl | gemma3vl
--processor-path             HF processor 路径
--tarfile-path               tar 根目录（默认 "/"）
--min-pixels-num             image processor 最小像素
--max-pixels-num             image processor 最大像素
--spatial-merge-size         vision spatial merge（默认 2）
--mask-history               多轮对话只保留最后一轮作为 loss target
--freeze-language-model      冻结 LLM 权重
--freeze-vision-model        冻结 ViT 权重
--freeze-vision-projection   冻结投影权重
--vlm-data-config-path       上述 JSON 路径
--vlm-dataloader-prefetch-factor
--vlm-top-domains-to-cut     调度差额分配到最大 N 个 domain 上
```

---

## 如何扩展一个新的 VLM 家族

假设要接入 `MyNewVL`（类似 Qwen 但有独特的 vision token 或 chat_template）。

**Step 1：新增 dataset 类** `vlm_mynew.py`

```python
from hcu_megatron.core.datasets.mm_dataset import MultiModalDataset

class MyNewVLDataset(MultiModalDataset):
    def __init__(self, ..., use_for_hf, mask_history, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_for_hf = use_for_hf
        self.mask_history = mask_history

    def process_vision(self, images, videos=None):
        # PIL images → pixel_values (+ 可选 image_grid_thw)
        ...

    def convert_example(self, example, conversations, imgs, domain_states, tools=None):
        # 完整单样本管线，参考 vlm_qwen.py / vlm_gemma.py
        # 末尾必须调用 self._pad_truncate_shift(...) 保证 shift 行为一致
        ...

    # 如需 Qwen3VL 那样的额外预处理：
    def _pre_convert_hook(self, imgs, json_data):
        return maybe_transformed(imgs)
```

**Step 2：新增 collator** `vlm_collator.py`

如果和 Qwen 家族的 collator 兼容，可以直接复用 `DataCollatorForQwenVL`；否则新写一个（参考 `DataCollatorForGemmaVL` 的简单版）。

**Step 3：新增 tokenizer setup** `vlm_tokenizer.py`

```python
def setup_mynew_tokenizer(tokenizer, args):
    hf = _unwrap_hf_tokenizer(tokenizer)
    # 注入必要的 image_token / image_token_id / ...
    ...
    _inject_common_ids(tokenizer, hf)
```

**Step 4：在 `vlm_dataset.py` 注册**

```python
from hcu_megatron.core.datasets.vlm_mynew import MyNewVLDataset
from hcu_megatron.core.datasets.vlm_tokenizer import setup_mynew_tokenizer

def _build_mynew_processor(args): ...
def _build_mynew_collator(args, tokenizer): ...

VLM_REGISTRY = {
    ...
    "mynewvl": VLMFamily(
        dataset_cls=MyNewVLDataset,
        tokenizer_setup=setup_mynew_tokenizer,
        build_collator=_build_mynew_collator,
        build_processor=_build_mynew_processor,
    ),
}
```

**Step 5：在 `vlm_args.py` 的 `--model-arch` choices 里加上 `mynewvl`**

**Step 6：不需要改任何其它文件。**

`build_train_valid_test_data_iter` 会自动通过 `_get_family(args)` 找到新条目。

---

## 常见修改场景

### 添加一个共享的图片预处理步骤

改 `vlm_images.py`。所有家族共享。

### 修改 pad/truncate/shift 行为

改 `MultiModalDataset._pad_truncate_shift`。所有家族共享（通过 `truncate_before_shift` 参数区分 Qwen 和 Gemma 风格）。

### Qwen 家族独有的 3D mRoPE 逻辑

由 Bridge 侧模型的 forward 计算，不在此模块。参考
`3rparty/Megatron-Bridge/src/megatron/bridge/models/qwen_vl/`。

### 新增一个 CLI 参数

改 `vlm_args.py::add_vlm_extra_args`。

### 修改 JSONL / 数据配置 JSON 的解析规则

改 `vlm_args.py::parse_dataset_config` + `_parse_split`。

### 新增 domain 采样策略（比如加权 loss 而不是加权采样）

改 `indexed_jsonl_dataset.py::generate_global_batch_domain_id`。

---

## 未接线的能力

以下功能**目前未接线**，如需启用要额外工作：

- **断点续训精确恢复**：过去有 `ConsumedByThisRank` / `update_consumed` 追踪每个 rank 每个 worker 的行号，但从未接入 checkpoint save/load，已删除。如需恢复，需要同时改 `hcu_megatron/training/training.py` 的 checkpoint 路径。
- **HuggingFace shuffle_buffer**：过去 CLI 有 `--vlm-shuffle-buffer-size`，实现里只是存了没用，已删除。
- **视频输入**：`vlm_collator.py` 有视频路径的字段骨架，但样本管线里
  `second_per_grid_ts` 恒为 `None`，实际未 exercise 过。

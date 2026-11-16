## BigMac 流水线调度

### 简介

BigMac 是面向多模态大模型训练的流水线并行优化方案，目标是兼顾计算效率与激活显存占用。其核心是依赖安全的嵌套流水线：保留语言模型主干的计算顺序和前后向依赖，将模态编码器、生成器的计算按时机嵌入调度，减少不同模块计算开销不均衡造成的流水线气泡。

在显存管理上，BigMac 将模态计算按流水线并行规模组织为微批单元，以有限的前向窗口控制尚未完成反向的单元数量，并在梯度就绪后尽早执行反向、释放激活。这样可以缩短模态激活的存活时间，避免为整个训练批次持续保留全部模态激活，在保持语言模型流水线运行的同时控制显存开销。

### 主要功能

当前 DAS 接入面向 Qwen3-VL 图文训练，使用视觉编码器与语言模型的嵌套调度；模型主体、loss 和优化器沿用 DAS 的 Megatron-Bridge 入口。主要功能如下：

- **视觉微批分担**：同一 PP 组中，微批 `i` 由 PP rank `i % PP` 编码。各 PP 阶段有完整视觉塔结构，TP 沿用模型配置；输出和 DeepStack 特征交给原生语言模型。
- **视觉窗口控制**：`VE_FORWARD_LIMIT` 默认 3，限制视觉前向超前窗口。窗口减小可能增加等待，增大可能增加激活显存。
- **变长通信**：固定长度 `int64` header 集中交换动态形状，payload 按需补齐并恢复真实形状，兼容无图像微批；反向复用前向形状，异步通信完成前保留缓冲。
- **按归属准备设备数据**：未归属本阶段的微批只推进 CPU 迭代器，跳过对应 H2D 和 TP 广播。CPU 数据流仍保持完整推进。
- **mRoPE 与 grid**：视觉负责 rank 计算位置结果，语言阶段在当前 forward 内复用；视觉几何元数据在 CPU 处理，注意力所需边界张量回到设备。
- **P2P 通信**：Qwen 默认选择执行器局部的非批量收发配置，保留必要 work 等待，不改写原生共享配置。
- **副本一致性**：优化器初始化前广播视觉参数与 buffers；反向结束先做视觉梯度 PP 求和，再进入原生 DP 收尾。目标桶为 64 MiB，最多两个未完成 work；副本 shared 标记避免梯度范数重复计数。
- **通信阶段对齐**：按接收方所在视觉通信阶段推迟语言 P2P 发送，防止大微批数下语言梯度发送与视觉 collective 形成等待环路，保持原计算顺序和视觉窗口。

视觉 PP 求和必须先于原生 DP 归约，因此 BigMac 视觉模型会关闭 DDP 梯度归约重叠。

### 代码结构

实现集中在 `hcu_megatron/core/pipeline_parallel/bigmac`。目录内文件如下：

| 文件                              | 职责                                   |
| --------------------------------- | -------------------------------------- |
| `__init__.py`                     | 按需加载公共接口                       |
| `config.py`                       | 特性开关与 HF 模型结构配置读取         |
| `data.py`                         | 本示例的数据路径、缺图与零 worker 兼容 |
| `schedules.py`                    | 调度器与通信计划                       |
| `executor.py` / `executor_vpp.py` | PP 执行器、元数据与迭代器 / VPP 执行器 |
| `communication.py`                | P2P、动态形状通信与视觉梯度分桶求和    |
| `qwen.py`                         | Qwen3-VL 桥接、mRoPE 与视觉 grid       |
| `integration.py`                  | 运行时安装与调度入口接入               |
| `trace.py`                        | 执行与通信 Trace                       |
| `simulator.py` / `visualize.py`   | 依赖与耗时模拟 / 绘图                  |
| `LICENSE`                         | 原始 MIT 许可证                        |

参数注册、校验与补丁接入位于 `hcu_megatron/features_manager/pipeline_parallel/bigmac_feature.py`，由 DAS feature manager 加载，无需手工调用安装函数。关闭特性时使用原生调度，视觉副本与 BigMac 专用 DDP 设置不生效。

如需直接使用规划接口：

```python
from hcu_megatron.core.pipeline_parallel.bigmac import BigMac, GPipe, PipelineSimulator
```

公共接口按需导入对应模块，绘图时才加载 matplotlib。

### 启动

在已有 DAS/HCU 训练环境中，从仓库根目录执行。请用实际模型和数据集路径替换以下示例。

```bash
HF_MODEL_PATH=./models/Qwen3-VL-8B-Instruct \
DATASET_PATH=./data/vlm-config.json \
USE_PP_EXECUTOR=1 USE_BIGMAC_STYLE=1 \
GPUS_PER_NODE=8 TP=2 PP=4 CP=1 MBS=1 GBS=8 \
bash examples/qwen3/train_qwen3vl_bigmac.sh
```

入口为单文件 [train_qwen3vl_bigmac.sh](../../examples/qwen3/train_qwen3vl_bigmac.sh)，通过所选 Python 的 `torch.distributed.run` 启动原生 `pretrain_vlm.py`。`PYTHON` 可指定解释器。

同参数原生对照使用参数 `USE_PP_EXECUTOR=0 USE_BIGMAC_STYLE=0`。底层 `MEGATRON_USE_PP_EXECUTOR`、`MEGATRON_PP_EXECUTOR_WITH_VE` 分别优先于两个 `USE_*` 变量；切换时应清除旧覆盖值：

```bash
unset MEGATRON_USE_PP_EXECUTOR MEGATRON_PP_EXECUTOR_WITH_VE
```

| 参数                                             | 默认值或用途                                        |
| ------------------------------------------------ | --------------------------------------------------- |
| `HF_MODEL_PATH` / `DATASET_PATH`                 | 模型目录 / DAS VLM JSON 配置文件          |
| `TP` / `PP` / `CP`                               | 2 / 4 / 1                                           |
| `MBS` / `GBS`                                    | 1 / 8                                               |
| `SEQ_LEN` / `MAX_POSITION_EMBEDDINGS`            | 2048 / 262144                                       |
| `SP` / `RECOMPUTE_GRANULARITY`                   | 0 / selective；selective 指定 `core_attn`           |
| `TRAIN_ITERS` / `LR_WARMUP_ITERS` / `EVAL_ITERS` | 50 / 0 / 0                                          |
| `NUM_WORKERS`                                    | 0                                                   |
| `LR` / `MIN_LR` / `WEIGHT_DECAY`                 | 1e-5 / 1e-6 / 0.01                                  |
| `MASTER_ADDR` / `MASTER_PORT` | 127.0.0.1 / 6202 |
| `OUTPUT_BASEPATH` / `TENSORBOARD_PATH`           | 输出根目录 / 默认根目录下的 tensorboard             |

微批数 `N = GBS / (MBS × DP)`，视觉模型要求 `N % PP == 0`。

查看命令而不启动训练：

```bash
DRY_RUN=1 bash examples/qwen3/train_qwen3vl_bigmac.sh
```

没有有效 HF 配置时保留 `<HF_MODEL_ARGS_FROM_CONFIG>` 占位；提供有效配置后会执行标准库配置解析并打印实际结构参数。DRY_RUN 不证明数据和训练依赖可用。

### 模型配置

结构参数由 `hcu_megatron/core/pipeline_parallel/bigmac/config.py` 读取所选 HF `config.json`，读取内容包括层数、hidden/FFN、注意力头、KV heads、head dimension 和 embedding 共享设置，并校验视觉输出宽度与语言 hidden 一致。

切换 Qwen3-VL 模型时，修改 `HF_MODEL_PATH` 即可读取对应结构，避免语言模型参数与视觉编码器输出维度不匹配。默认配置要求语言层数能被 `PP` 整除。

### 数据配置

`DATASET_PATH` 须指向 JSON 配置，不能直接指向 WebDataset 目录。配置内的 `path` 可为 JSONL 文件或包含 JSONL 的目录：

```json
{
  "path_base": "config",
  "train_data_infos": {
    "train": {"path": "train.jsonl", "probability": 1.0}
  },
  "eval_data_infos": {
    "val": {"path": "val.jsonl"}
  }
}
```

每行 JSONL 通过图片路径引用媒体文件：

```json
{"conversations":[{"role":"user","content":"<image>描述图片"},{"role":"assistant","content":"示例回答"}],"images":[{"image_path":"images/sample.jpg"}],"image_path_base":"jsonl"}
```

`path_base=config` 按配置目录解析相对 domain 路径；`image_path_base=jsonl` 按 JSONL 目录解析普通相对图片路径。没有标记时保留原语义。改配置中的 JSONL 位置不会重写样本中的旧绝对图片路径。

缺文件或权限错误会立即抛出带上下文的异常，其他解码异常仍保留原跳过行为。`NUM_WORKERS=0` 时预取因子为 `None`。数据兼容处理集中在 `bigmac/data.py`，由本示例显式启用；同一脚本的原生和 BigMac 对照采用相同处理，其他原生入口保持原有行为。

### 约束

- 视觉模式要求原生 Qwen3-VL Bridge、CP=1、非 VPP、非 packed 图文或纯文本批次；未接入音频、视频训练。
- 只与 `schedule_method=vanilla` 组合，不与 DualPipeV、RiPipe、Seq1F1B、ZB-H1 等替代调度叠加。
- 不支持词表并行、延迟 wgrad、专家通信重叠、FSDP、full-iteration CUDA graph、细粒度 activation offload；视觉模式不支持参数汇聚重叠。
- 视觉模式不支持 `--bridge-language-model-only`，也不支持会触发强制梯度归约的 `--save-wgrads-interval`。
- 验证及 PP=1 调度回到原生路径；视觉副本不会因验证回退而从已构建的模型中删除。



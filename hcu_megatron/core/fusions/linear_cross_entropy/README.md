# HCU Linear CE for Megatron

本目录只负责 Megatron 框架接入。汇编 Kernel、在线选优和 workspace 都属于 flash-train 生成的架构专属 `.so`，不放入 Megatron 仓库。

本实现基于 NVIDIA Megatron-LM 的 Apache-2.0 代码库及其 Linear CE 接口设计，由 Hygon 增加 HCU 平台检测、架构 `.so` 加载、DP 接入、测试与操作文档；上游版权和仓库 `LICENSE` 保持不变。

## 文件

```text
hcu_megatron/core/fusions/
├── fused_linear_cross_entropy.py   # 公共 Autograd API
└── linear_cross_entropy/
    ├── __init__.py                 # 兼容导出
    ├── entry.py                    # 张量和 DP 语义检查
    ├── extension.py                # 按架构加载 .so 并调用 Torch op
    ├── platform.py                 # HCU 与 gfxNNN 架构检测
    └── README.md
```

本目录没有 `kernels/`、`native/`、manifest、registry 或 shape plan。

## 当前接口范围

- hidden、weight：BF16、连续、同一 HCU device
- labels：int64、连续、同一 HCU device
- reduction：`mean`
- ignore index：`-100`
- 并行：DP only，`tp_group=None`、`sequence_parallel=False`
- hidden 支持 2D 或 3D，native 调用前展平成 `(N,D)`，backward 恢复原 shape
- 当前已在 gfx936 验证两个问题规模，框架代码本身不写死 shape

## 安装和加载

安装当前架构对应的 artifact wheel：

```bash
pip install hcu_linear_ce_artifacts_gfx936-*.whl
```

loader 从 HCU device properties 读取 `gfxNNN`，再请求：

```text
hcu_linear_ce_artifacts.paths.extension_path(arch)
```

最终加载 `libhcu_linear_ce_<arch>.so`。例如 gfx936 加载 `libhcu_linear_ce_gfx936.so`，未来 gfx938 加载 `libhcu_linear_ce_gfx938.so`，无需修改 Megatron Python 接口。

现场调试可覆盖路径：

```bash
export HCU_LINEAR_CE_EXTENSION_PATH=/path/to/libhcu_linear_ce_gfx936.so
```

## 使用

```python
from hcu_megatron.core.fusions.fused_linear_cross_entropy import linear_cross_entropy

loss = linear_cross_entropy(
    hidden,
    output_layer_weight,
    labels,
    tp_group=None,
    reduction="mean",
    ignore_index=-100,
    sequence_parallel=False,
)
```

公共 `LinearCrossEntropy` Autograd Function 在 forward 保存 native backward 需要的 hidden、weight、labels、saved LSE 和 valid count。调用方只接收 loss。

## 运行流程

1. `entry.py` 检查 dtype、shape、连续性、device 和 DP 语义。
2. `platform.py` 检测 HCU 和当前 `gfxNNN` 架构。
3. `extension.py` 加载对应架构 `.so`，取得 `torch.ops.hcu_linear_ce`。
4. forward/backward 张量直接传给 native op。
5. `.so` 先查当前进程 Runtime Plan；多候选时可恢复 Persistent Tuning Cache，miss/invalid 才在线选优。

Megatron 不选择候选，也不读取 `.co`、manifest 或 shape capability 文件。

日志由 native runtime 控制：

```bash
export HCU_LINEAR_CE_LOG_LEVEL=1
export HCU_LINEAR_CE_LOG_LEVEL=2
```

也可以使用兼容开关请求 level 1 摘要：

```bash
export HCU_LINEAR_CE_LOG_KERNEL=1
```

预期摘要包括：

```text
FORWARD WINNER: ...
BACKWARD WINNERS: dH=... | dW=... | combined=...
PLAN_READY ...
RUNTIME_PLAN_HIT role=...
```

## 测试

本地无卡契约测试：

```bash
python -m unittest tests.unit_tests.fusions.test_hcu_linear_cross_entropy -v
```

测试覆盖 gfx936/gfx938 架构选择、`.so` loader、entry 参数透传、3D hidden shape 恢复、公共 API 签名和仓库无旧 Kernel/manifest 内容。

native 构建与 HCU 正确性验证在 flash-train 的 `linear_cross_entropy/` 中执行。

## 训练接入测试

以下测试用于验证 `--cross-entropy-loss-fusion --cross-entropy-fusion-impl linear` 的 GPT 接入逻辑和训练 loss 适配，不需要 HCU 设备或 native `.so`。两个参数同时生效时，`gpt_model_postprocess` 直接调用融合训练损失接口。请在仓库根目录执行命令。

### GPT 适配契约测试

[test_hcu_linear_cross_entropy_adaptor.py](../../../../tests/unit_tests/fusions/test_hcu_linear_cross_entropy_adaptor.py) 执行生产代码中的参数处理函数和 HCU GPT `_postprocess` 函数体，并替换其外部依赖，检查：

- 官方参数的默认值、`linear` 选项扩展，以及 BF16、TP/CP、SP、MTP 等配置校验。
- 两个参数共同控制融合路径，`native/te` 保留原路径；训练和带 labels 验证调用均使用融合路径。
- 非输出阶段、无 labels 和活动推理模式下的原路径回退。
- 不支持的静态配置在 Feature 参数校验阶段被拒绝；缺失的 `loss_mask`、packed sequence、`mtp_in_postprocess` 和已有 `output_processor` 冲突在 GPT 输出计算前被拒绝。
- Feature 仅扩展已有参数选项（兼容 list/tuple、重复调用和早期解析器缺少该参数的情况），不再注册额外补丁，以及共享权重和输出层权重的选择。
- 早期适配参数不含官方开关时，完整参数校验仍生效，并在上游校验后恢复 `linear` 配置。

```bash
python -m unittest discover -s tests/unit_tests/fusions \
  -p 'test_hcu_linear_cross_entropy_adaptor.py' -v
```

### 训练损失与梯度测试

[test_hcu_linear_cross_entropy_training.py](../../../../tests/unit_tests/fusions/test_hcu_linear_cross_entropy_training.py) 需要安装 PyTorch，使用 CPU 上的普通线性投影和交叉熵替代 native 调用，检查：

- `linear_cross_entropy_for_training` 返回值经 `sum(loss * loss_mask)` 聚合后的总损失，与普通逐 token CE 参考结果一致。
- hidden 和输出权重梯度与参考结果一致，连续两次反向传播的梯度累积正确。
- 部分 mask、`ignore_index=-100` 和全部 mask 场景；全部 mask 时总损失和梯度为零。
- 空 batch 和非二值加权 mask 被明确拒绝。

```bash
python -m unittest discover -s tests/unit_tests/fusions \
  -p 'test_hcu_linear_cross_entropy_training.py' -v
```

未安装 PyTorch 时，该文件的数值测试会标记为 `skipped`，不能视为已完成数值验证。

### 一并运行 Linear CE 单元测试

以下命令同时运行原有 loader/entry 契约测试和上述两组训练接入测试：

```bash
mkdir -p .test-tmp
TMPDIR="$PWD/.test-tmp" PYTHONDONTWRITEBYTECODE=1 \
python -m unittest discover -s tests/unit_tests/fusions \
  -p 'test_hcu_linear_cross_entropy*.py' -v
```

这些测试不构建完整 Megatron 模型，也不验证 native HCU kernel 的数值精度。设备端 forward/backward smoke 测试见下文“从源码到 Megatron 的完整操作”；完整模型训练、多卡通信及优化器更新仍需在目标训练配置下单独验证。

## 环境准备与安装

Megatron 运行在 HCU 节点时，先加载与 native 构建一致的运行时环境：

```bash
source /opt/dtk/env.sh
source /workspace/env.sh
```

然后安装与设备架构匹配的 artifact wheel。wheel 文件由 flash-train 的 `linear_cross_entropy/scripts/build_artifact_wheel.sh` 生成：

```bash
python -m pip install hcu_linear_ce_artifacts_gfx936-*.whl
```

wheel 名称和库文件名都带架构标识；gfx936 和 gfx938 应分别构建、分别安装和分别验证。Megatron Python 接口不随架构复制。

loader 默认根据当前 HCU 的 `gfxNNN` 架构发现 wheel 中的库。实际搜索顺序是：`HCU_LINEAR_CE_EXTENSION_PATH`、已安装的 `hcu_linear_ce_artifacts`、Megatron package-local `linear_cross_entropy/lib/libhcu_linear_ce_<arch>.so`。现场调试可用环境变量覆盖默认发现路径：

```bash
export HCU_LINEAR_CE_EXTENSION_PATH=/path/to/libhcu_linear_ce_gfx936.so
```
覆盖路径只应指向已经完成 native 正确性验证的架构专属 `.so`。首次进程调用可能包含动态库加载、Torch op 注册和在线选优，这些启动成本不代表稳态调用。

加载链路是：`entry.py` 做输入和并行语义检查，`platform.py` 检测 HCU 与架构，`extension.py` 调用 `torch.ops.load_library()` 加载 `.so`，然后通过 `torch.ops.hcu_linear_ce.forward/backward` 调用 native 实现。Megatron 不读取汇编、`.co`、manifest 或 shape 表。

## 从源码到 Megatron 的完整操作

完整流程应在 HCU 节点执行。可以直接加载 flash-train 构建的 `.so`：

```bash
cd /path/to/Megatron-LM-das
export PYTHONPATH=$PWD
export HIP_VISIBLE_DEVICES=0
export HCU_LINEAR_CE_LOG_LEVEL=1
export HCU_LINEAR_CE_EXTENSION_PATH=/path/to/flash-train/linear_cross_entropy/dist/gfx936/lib/libhcu_linear_ce_gfx936.so
python tests/unit_tests/fusions/validate_hcu_linear_cross_entropy.py \
  --library "$HCU_LINEAR_CE_EXTENSION_PATH" \
  --n 2048 --d 4096 --v 32000
```

也可以安装架构 wheel，让 loader 自动发现 `.so`：

```bash
python -m pip install --force-reinstall \
  /path/to/flash-train/linear_cross_entropy/dist/artifact_wheel/gfx936/wheelhouse/hcu_linear_ce_artifacts_gfx936-*.whl
unset HCU_LINEAR_CE_EXTENSION_PATH
python tests/unit_tests/fusions/validate_hcu_linear_cross_entropy.py \
  --n 2048 --d 4096 --v 32000
```

all-ignore 边界用例：

```bash
python tests/unit_tests/fusions/validate_hcu_linear_cross_entropy.py \
  --n 2048 --d 4096 --v 32000 --all-ignore
```

成功时输出 `MEGATRON_LINEAR_CE_SMOKE_PASS`。脚本通过 Megatron 公共 Autograd API 连续执行两次 forward/backward，检查有限值、重复一致性与 all-ignore 语义。无卡单元测试仍可单独执行：

```bash
python -m unittest tests.unit_tests.fusions.test_hcu_linear_cross_entropy -v
```

真实 HCU 验证使用 `flash-train/linear_cross_entropy/tests/validate_direct_link_runtime.py`，先验证 loss、dHidden 和 dWeight，再从 Megatron 公共 `linear_cross_entropy` API 验证端到端 backward。Windows 本地只能执行契约和 loader 单元测试，不能验证设备 Kernel。

## 在线选优日志

选优 cache 由 `.so` 管理，Megatron 不读取、解析或写入 cache。默认目录是 `${XDG_CACHE_HOME:-~/.cache}/hcu_linear_ce/`，可设置：

```bash
export HCU_LINEAR_CE_CACHE_DIR=/data/hcu_linear_ce_cache
export HCU_LINEAR_CE_TUNING_MODE=auto
export HCU_LINEAR_CE_TUNING_MODE=retune
```

新进程首次遇到问题签名时，多个兼容候选会优先恢复持久化 winner；记录缺失或失效才在线兜底。只有一个兼容候选时直接运行，不 benchmark、不写伪造性能记录。首次 `.so`、rocBLAS 和 device 初始化仍属于冷启动，不代表稳态封装开销。

设置 `HCU_LINEAR_CE_LOG_LEVEL=1` 查看 winner、plan ready 和 cache hit；设置为 `2` 查看每个候选的每次 sample 以及 median/min/max。forward、backward dH 和 backward dW 分别选优，backward 还会打印选中 dH+dW 的 combined 结果。

当前每个角色只有一个已注册 family，因此正常构建会输出 `SINGLE_CANDIDATE_SELECTED` 并直接建立 Runtime Plan，不执行 warmup/benchmark，也不写持久化性能 `.cache` 记录。增加并注册至少两个兼容 family 后，才会发生真实在线选优与跨进程 winner 恢复。

## 常见问题

- `HCU Linear CE backend unavailable`：确认 PyTorch 能看到 HCU、已加载 DTK 环境，并且设备名或架构信息能识别为 HCU。
- `extension is not installed or loaded`：安装与当前 `gfxNNN` 匹配的 wheel，或设置 `HCU_LINEAR_CE_EXTENSION_PATH`。
- `loaded ... does not register both ... forward and backward`：检查 `.so` 是否来自完整 native 构建，且没有混用不同架构或不同构建目录的对象。
- 输入校验失败：当前实现要求 BF16 hidden/weight、int64 labels、连续张量、`mean`、`ignore_index=-100`、DP only；hidden 可为 2D 或 3D。
- 选优日志缺少候选 sample：将 `HCU_LINEAR_CE_LOG_LEVEL` 设为 `2`，并在首次使用该问题签名的新进程中运行。

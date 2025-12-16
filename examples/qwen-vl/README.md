
# 使用 dcu_megatron 训练 Qwen2.5-VL模型

> qwen-vl预训练实现参考[Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch/tree/main/examples/qwen2_5_vl)框架

## 安装Pai-Megatron-Patch
```shell
cd /workspace
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
git checkout 13485f9fbe
```

## 数据集和模型下载
```shell
# 下载模型权重
cd /workspace
mkdir Qwen2.5-VL-3B
cd Qwen2.5-VL-3B
modelscope download --model Qwen/Qwen2.5-VL-3B-Instruct --local_dir ./
cd ..

# 下载预训练数据集
mkdir llava-datasets
cd llava-datasets
git clone https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain
cd LLaVA-Pretrain
unzip images.zip

# 安装 Megatron-Energon （假设clone的dcu_megatron源码路径为 /workspace/dcu_megatron）
cd /workspace/dcu_megatron/Megatron-Energon
pip install -e .

# 将数据集处理成webdataset格式
cd /workspace/Pai-Megatron-Patch/toolkits/pretrain_data_preprocessing
python convert_llava_pretrain_to_wds.py /workspace/llava-datasets/LLaVA-Pretrain/

# 数据集处理
cd /workspace/llava-datasets/LLaVA-Pretrain/wds
energon prepare ./

# 依次键入如下题目后的选项
> Please enter a desired train/val/test split like "0.5, 0.2, 0.3" or "8,1,1": 9,1,0
> Do you want to create a dataset.yaml interactively? [Y/n]: Y
> Please enter a number to choose a class: 9 (VQASample)
> Do you want to set a simple field_map[Y] (or write your own sample_loader [n])? [Y/n]: Y
> Please enter a webdataset field name for 'image' (<class 'torch.Tensor'>): jpg
> Please enter a webdataset field name for 'context' (<class 'str'>): json[0][value]
> Please enter a webdataset field name for 'answers' (typing.Optional[typing.List[str]], default: None): json[1][value]
> Please enter a webdataset field name for 'answer_weights' (typing.Optional[torch.Tensor], default: None):

```

## Megatron-Core模型训练流程

### Megatron-Core模型格式转换
注意: **请使用4.52.0以下版本的transformers**进行权重转换，否则会导致权重key错误，建议`pip install transformers==4.51.3`

传参列表如下：
```shell
MODEL_SIZE=$1               # 模型大小，3B, 7B, 32B, 72B
LOAD_DIR=$2                 # 源权重路径
SAVE_DIR=$3                 # 目标权重路径
MG2HF=$4                    # 转换方向 可选: true, false
USE_CUDA=$5                 # 是否使用GPU转换 建议: true
PR=$6                       # 转换精度 可选: fp32 bf16 fp16
HF_DIR=$7                   # HF权重路径(mcore2hf时必须提供)
```

使用下述脚本将checkpoint转换到MCore格式
```shell
cd /workspace/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor
bash scripts/qwen2_5_vl/run_8xH20.sh \
3B \
/workspace/Qwen2.5-VL-3B/Qwen2.5-VL-3B-Instruct \
/workspace/Qwen2.5-VL-3B/Qwen2.5-VL-3B-Instruct-to-mcore  \
false \
true \
bf16
```

执行预训练命令
```shell
# 填写好 run_qwen2.5vl_3B.sh 和 hostfile_qwen2.5vl_3B 对应的参数后
bash run_qwen2.5vl_3B.sh
```
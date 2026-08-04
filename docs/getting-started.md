# 快速开始

> 在 HCU 上跑起第一个 AI 模型训练。

## 前置条件

- 已安装 HCU 驱动和 DTK（参考 [environment-setup.md](environment-setup.md)）
- Python 3.10+ 环境
- 至少8张 HCU（64GB 显存推荐）

## 大语言模型 (LLM)

这里给出预训练 和 sft 的示例, 前面几步都是相同的

1. 拉取镜像

    `docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:pytorch2.9.0-ubuntu22.04-dtk26.04-py3.10_te2.10`
    
    python3.10需要用 `from typing_extensions import override` 替换掉 `from typing import override`

2. 启动容器
    ```bash
    docker run -it \
        --shm-size=64G \
        --device=/dev/kfd \
        --device=/dev/mkfd \
        --device=/dev/dri \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --ulimit memlock=-1:-1 \
        --ipc=host \
        --network=host \
        --group-add video \
        --privileged \
        --name CONTAINER_NAME \
        -v /opt/hyhal:/opt/hyhal:ro \
        -v /root/.ssh:/root/.ssh:ro \
        -v /path/to/workspace:/path/to/workspace \
        harbor.sourcefind.cn:5443/dcu/admin/base/custom:pytorch2.9.0-ubuntu22.04-dtk26.04-py3.10_te2.10 \
        /bin/bash
    ```
3. 拉取hcu-megatron源码
    ```bash
    git clone http://42.228.13.241:10068/dcutoolkit/deeplearing/dcu_megatron.git
    cd dcu_megatron
    git checkout origin/core_v0.18.2
    git submodule update --init --recursive
    ```
4. 下载模型配置

    `modelscope download --model Qwen/Qwen3-8B --local_dir ./Qwen3-8B`


### 使用 hcu-megatron 做全参预训练

5. 下载数据集

    `modelscope download --dataset kanliu/oscar-en-10k-megatron oscar-en-10k.jsonl --local_dir ./oscar-en-10k`

6. 处理数据集
    ```bash
    python Megatron-LM/tools/preprocess_data.py \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model ./Qwen3-8B \
        --input ./oscar-en-10k/oscar-en-10k.jsonl \
        --output-prefix ./oscar-en-10k/oscar-en-10k-qwen3 \
        --append-eod \
        --workers 16
    ```
    处理完成后会分别保存为.bin和.idx文件

7. 配置启动脚本
    ```bash
    cd examples/qwen3
    # 设置节点列表, 单节点使用localhost即可
    echo localhost > hostfile

    # 设置rccl环境变量
    vim ../../requirements/env.sh
        # 设置socket等信息
        export GLOO_SOCKET_IFNAME=实际的网卡

    # 完善启动脚本中的配置信息
    vim run_qwen.sh
        # 下面为配置参考设置
        DTK_ENV="/opt/dtk/env.sh" 
        DATA_PATH=${MEGATRON_PATH}/oscar-en-10k/oscar-en-10k-qwen3_text_document 
        TOKENIZER_MODEL_PATH=${MEGATRON_PATH}/Qwen3-8B
        LAUNCHER="torchrun"
        CHECKPOINT_PATH="./ckpt"
        NCCL_ENV=${MEGATRON_PATH}/requirements/env.sh
        LAUNCH_WITH_BINDING=${MEGATRON_PATH}/requirements/launch_with_binding.sh

    ```
8. 启动单节点训练

    从 hostfile 中使用 1 个节点训练

    `bash run_qwen.sh hostfile 1`

9. 启动多节点训练

    ```bash
    # 在hostfile中写入多个节点的实际ip
    vim hostfile
        #例如
        192.169.0.1
        192.168.0.2

    # 设置run_qwen.sh中的免密端口
    vim run_qwen.sh
        # 例如配置了免密端口 /usr/sbin/sshd -p 11451
        --mca plm_rsh_args '-p 11451'

    bash run_qwen.sh hostfile 2
    ```

### 使用 hcu-megatron 做sft全参微调

5. 下载数据集

    `modelscope download --dataset iic/ms_bench README.md ms_agent_bench_v1_sft_sample.jsonl --local_dir ./ms_bench`

6. 处理数据集

    ```bash
    python examples/tools/preprocess_sftdata.py \
        --src ./ms_bench/ms_agent_bench_v1_sft_sample.jsonl \
        --out-dir ./ms_bench/sft \
        --valid-ratio 0.1 \
        --seed 42
    ```
    处理完成后会分别生成train.jsonl和val.jsonl

7. 配置启动脚本

    ```bash
    cd examples/qwen3
    # 设置节点列表, 单节点使用localhost即可
    echo localhost > hostfile

    # 设置rccl环境变量
    vim ../../requirements/env.sh
        # 设置socket等信息
        export GLOO_SOCKET_IFNAME=实际的网卡

    # 完善启动脚本中的配置信息
    vim run_qwen.sh
        # 下面为配置参考设置
        DTK_ENV="/opt/dtk/env.sh" 
        DATA_PATH=${MEGATRON_PATH}/ms_bench/sft
        TOKENIZER_MODEL_PATH=${MEGATRON_PATH}/Qwen3-8B
        LAUNCHER="torchrun"
        CHECKPOINT_PATH="./ckpt"
        NCCL_ENV=${MEGATRON_PATH}/requirements/env.sh
        LAUNCH_WITH_BINDING=${MEGATRON_PATH}/requirements/launch_with_binding.sh

        # 修改train_qwen3_8b.sh中的数据集设置(默认为全参pretrain)
        DATA_ARGS=(
            # --tokenizer-type HuggingFaceTokenizer
            # --tokenizer-model ${TOKENIZER_MODEL_PATH}
            # --data-path ${DATA_PATH} 
            # --split 949,50,1
            --sft
            --sft-tokenizer-prompt-format default
            --tokenizer-type SFTTokenizer
            --tokenizer-model ${TOKENIZER_MODEL_PATH}
            --no-create-attention-mask-in-dataloader
            --train-data-path ${DATA_PATH}/train.jsonl
            --valid-data-path ${DATA_PATH}/valid.jsonl
        )
    ```

8. 启动单节点训练

    从 hostfile 中使用 1 个节点训练

    `bash run_qwen.sh hostfile 1`

9. 启动多节点训练

    ```bash
    # 在hostfile中写入多个节点的实际ip
    vim hostfile
        #例如
        192.169.0.1
        192.168.0.2

    # 设置run_qwen.sh中的免密端口
    vim run_qwen.sh
        # 例如配置了免密端口 /usr/sbin/sshd -p 11451
        --mca plm_rsh_args '-p 11451'

    bash run_qwen.sh hostfile 2
    ```



## 多模态模型 (VLM)

### 使用 hcu-megatron 做图像理解sft

1. 拉取镜像

    `docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:pytorch2.9.0-ubuntu22.04-dtk26.04-py3.10_te2.10`

2. 启动容器
    ```bash
    docker run -it \
        --shm-size=64G \
        --device=/dev/kfd \
        --device=/dev/mkfd \
        --device=/dev/dri \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --ulimit memlock=-1:-1 \
        --ipc=host \
        --network=host \
        --group-add video \
        --privileged \
        --name CONTAINER_NAME \
        -v /opt/hyhal:/opt/hyhal:ro \
        -v /root/.ssh:/root/.ssh:ro \
        -v /path/to/workspace:/path/to/workspace \
        harbor.sourcefind.cn:5443/dcu/admin/base/custom:pytorch2.9.0-ubuntu22.04-dtk26.04-py3.10_te2.10 \
        /bin/bash
    ```
3. 拉取hcu-megatron源码
    ```bash
    git clone http://42.228.13.241:10068/dcutoolkit/deeplearing/dcu_megatron.git
    cd dcu_megatron
    git checkout origin/core_v0.18.2
    git submodule update --init --recursive
    ```
4. 下载模型配置

    `modelscope download --model Qwen/Qwen3-VL-8B-Instruct README.md --local_dir ./Qwen3-VL-8B-Instruct`

5. 下载并处理数据集

    ```bash
    python examples/tools/preprocess_msftdata.py \
        --data-dir ./multi_sft \
        --splits train valid \
        --max-worker 20 \
        --retries 3 \
        --train-limit 1000 \
        --valid-limit 100
    ```
    会自动下载并处理数据集到指定目录

7. 配置启动脚本

    ```bash
    cd examples/qwen3vl
    # 设置节点列表, 单节点使用localhost即可
    echo localhost > hostfile

    # 设置rccl环境变量
    vim ../../requirements/env.sh
        # 设置socket等信息
        export GLOO_SOCKET_IFNAME=实际的网卡

    # 完善vlm-config.json中的配置信息, 将path修改为实际的train.jsonl路径
    vim vlm-config.json

    # 完善启动脚本中的配置信息
    vim run_qwen.sh
        # 下面为配置参考设置
        DTK_ENV="/opt/dtk/env.sh" 
        DATA_PATH="./vlm-config.json"
        TOKENIZER_MODEL_PATH=${MEGATRON_PATH}/Qwen3-VL-8B-Instruct
        LAUNCHER="torchrun"
        CHECKPOINT_PATH="./ckpt"
        NCCL_ENV=${MEGATRON_PATH}/requirements/env.sh
        LAUNCH_WITH_BINDING=${MEGATRON_PATH}/requirements/launch_with_binding.sh

        #将下面CMD中的脚本改成train_qwen3vl_8B.sh
        bash train_qwen3vl_8B.sh
    ```

8. 启动单节点训练

    从 hostfile 中使用 1 个节点训练

    `bash run_qwen.sh hostfile 1`

9. 启动多节点训练

    ```bash
    # 在hostfile中写入多个节点的实际ip
    vim hostfile
        #例如
        192.169.0.1
        192.168.0.2

    # 设置run_qwen.sh中的免密端口
    vim run_qwen.sh
        # 例如配置了免密端口 /usr/sbin/sshd -p 11451
        --mca plm_rsh_args '-p 11451'

    bash run_qwen.sh hostfile 2
    ```

## 下一步

- [模型部署文档](model-deployment/)
- [多模态方案概览](multimodal/overview.md)
- [性能调优指南](optimization/performance-tuning.md)

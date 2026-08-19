# 基础镜像已内置 DTK 26.04 / MPI5.0 / gcc11.4 / cmake3.29 / py3.10
FROM harbor.sourcefind.cn:5443/dcu/admin/base/dtk:26.04-ubuntu22.04-mpi5.0-gcc11.4-cmake3.29-py3.10

SHELL ["/bin/bash", "-c"]

# pip 配置 + das-install
RUN mkdir -p ~/.pip && \
    printf '%s\n' \
    '[global]' \
    'index-url = http://10.16.1.201:9929/nightly/dtk2604/+simple/' \
    'trusted-host = 10.16.1.201' \
    > ~/.pip/pip.conf && \
    pip install --ignore-installed --no-cache-dir blinker && \
    printf '%s\n' \
    '#!/bin/bash' \
    '# usage: das-install <pkg>[==<ver>] <torch_tag>' \
    'pkg=$1 tag=$2 prefix=' \
    'if [[ "$pkg" == *==* ]]; then prefix="${pkg#*==}"; pkg="${pkg%%==*}"; fi' \
    'ver=$(pip index versions "$pkg" 2>/dev/null | grep -oP "${prefix}[^ ()]*${tag}[^ ()]*" | head -1)' \
    '[ -z "$ver" ] && echo "Error: No version for $pkg matching $tag" && exit 1' \
    'echo "Installing $pkg==$ver"' \
    'pip install --no-cache-dir --extra-index-url https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com "$pkg==$ver"' \
    > /usr/local/bin/das-install && chmod +x /usr/local/bin/das-install

# AICC 编译器 (nightly 最新版地址由 CI 通过 build-arg 传入, 在镜像内下载安装)
ARG AICC_URL=http://10.16.1.201:8000/ai_cc/nightly/dtk_llvm_a33f28_202608121638.run
ARG AICC_VERSION=a33f28_202608121638
ENV AICC_VERSION=${AICC_VERSION}

RUN wget -q "${AICC_URL}" -O /tmp/aicc.run && \
    chmod +x /tmp/aicc.run && \
    yes | /tmp/aicc.run --dtk_dir /opt/dtk && \
    rm -f /tmp/aicc.run

# ---layer---
RUN TORCH_VERSION=2.10.0 \
    && TORCH_TAG="torch${TORCH_VERSION//./}" \
    && pip install --no-cache-dir torch==${TORCH_VERSION} torchvision \
    && das-install flash_attn ${TORCH_TAG} \
    && das-install triton==3.6.0 ${TORCH_TAG} \
    && das-install apex ${TORCH_TAG} \
    && das-install deepspeed ${TORCH_TAG} \
    && das-install transformer-engine ${TORCH_TAG} \
    && pip cache purge \
    && rm -f /usr/local/bin/das-install

# Megatron-LM-das 源码 (构建上下文即仓库根目录, 3rdparty 子模块由 CI checkout)
ENV PYTHONPATH=/workspace/Megatron-LM
COPY . /workspace/Megatron-LM

RUN rm -rf ~/.pip/pip.conf

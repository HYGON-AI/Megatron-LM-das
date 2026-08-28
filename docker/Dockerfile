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
    && das-install deep-ep ${TORCH_TAG} \
    && pip cache purge \
    && rm -f /usr/local/bin/das-install

# Megatron-LM-das 源码 (构建上下文即仓库根目录, 3rdparty 子模块由 CI checkout)
ENV PYTHONPATH=/workspace/Megatron-LM
COPY . /workspace/Megatron-LM

# rccl-tests (带 MPI 编译, 供 multi/rccltest/test_rccl.sh 做单机/多机通信单测)
# 注意:
#   1. 不能用 rocminfo 探测 gfx 架构 —— 构建期无 GPU 设备, rocminfo 会因缺
#      libhsa-runtime64.so.1 直接退出 127。这里改为显式列出 GPU_TARGETS。
#   2. 基础镜像里 /opt/dtk/lib/libhsa-runtime64.so 是指向 /opt/hyhal 的悬空软链
#      (/opt/hyhal 是宿主机驱动目录, 只在 run 时挂载)，链接期找不到该库。
#      因此先造一个同名 SONAME 的空 stub 供链接, 运行时由真实的 /opt/hyhal 提供。
ARG RCCL_TEST
ARG RCCL_TESTS_COMMIT=155f4748e6de7ce3ab54a0bf09931d9316262260
ARG RCCL_GPU_TARGETS="gfx936 gfx938"
RUN : "${RCCL_TEST:?RCCL_TEST build argument is required}" && \
    mkdir -p /tmp/hsastub && cd /tmp/hsastub && : > empty.c && \
    gcc -shared -fPIC -Wl,-soname,libhsa-runtime64.so.1 -o libhsa-runtime64.so.1 empty.c && \
    ln -sf libhsa-runtime64.so.1 libhsa-runtime64.so && \
    git clone "${RCCL_TEST}" /opt/rccl-test && \
    cd /opt/rccl-test && git checkout ${RCCL_TESTS_COMMIT} && \
    LIBRARY_PATH=/tmp/hsastub:${LIBRARY_PATH} \
    make HIP_HOME=/opt/dtk \
         NCCL_HOME=/opt/dtk/rccl \
         CUSTOM_RCCL_LIB=/opt/dtk/rccl/lib/librccl.so \
         MPI_HOME=/opt/mpi \
         CFLAGS=-I/opt/mpi/include \
         MPI=1 \
         GPU_TARGETS="${RCCL_GPU_TARGETS}" \
         -j $(nproc) && \
    rm -rf /tmp/hsastub /opt/rccl-test/build/*.o /opt/rccl-test/.git

RUN rm -rf ~/.pip/pip.conf

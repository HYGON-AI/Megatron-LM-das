# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

# ============================================================================
# 构建参数 (全部提到文件头统一维护, 不提供默认值, 必须由 --build-arg / CI 显式传入)
# 备注: 一律不写示例值, 具体配置见 CI / 仓库环境变量
# ============================================================================
# --- 基础镜像 ---
ARG BASE_IMAGE      # dtk 基础镜像完整引用 (必传, 如 …/hcu/base/dtk:26.04-ubuntu22.04-py3.10)
                    # 基础镜像里固化了 ENV DTK_VERSION (token 格式, 如 dtk2604), FROM 后可直接用
# --- pip 源 ---
ARG PIP_INDEX_URL       # 内网 pip 完整 index-url; CI 按 DTK 版本拼好传入 (...,/nightly/dtk2604/+simple/)
                        # trusted-host 由 Dockerfile 从链接自动提取; token 不再单独传/推导
# --- AICC 编译器 ---
ARG RESOURCE_SERVER_URL # 资源服务器地址
ARG AICC_VERSION        # AICC 完整版本号, 可选; 缺省时按基础镜像 DTK_VERSION 派生大版本
# --- 镜像信息 ---
ARG BUILD_TIME          # 构建时间戳 (与产物 tag 的时间段一致, 如 20261009-1113); 缺省用当时时间
# --- 框架依赖 ---
ARG TORCH_VERSION       # torch 大版本; das-install 的 torch_tag 由此推导
# --- rccl-tests ---
ARG RCCL_TEST           # rccl-tests 仓库地址 (必传, 构建期强校验)
ARG RCCL_TESTS_COMMIT   # rccl-tests 锁定 commit
ARG RCCL_GPU_TARGETS    # 编译目标 GPU 架构列表, 空格分隔
# --- run_nhc ---
ARG RUN_NHC     # run_nhc 源码仓库地址 (必传, 构建期强校验)
ARG RUN_NHC_COMMIT   # run_nhc 锁定 commit (必传)

# ============================================================================
# run_nhc 二进制构建阶段
# 源码只存在于本阶段; 最终镜像不含任何 node_check 源码
# ============================================================================
FROM ${BASE_IMAGE} AS run_nhc_builder
SHELL ["/bin/bash", "-c"]
ARG PIP_INDEX_URL
ARG RUN_NHC
ARG RUN_NHC_COMMIT

RUN : "${PIP_INDEX_URL:?PIP_INDEX_URL build argument is required}" && \
    : "${RUN_NHC:?RUN_NHC build argument is required}" && \
    mkdir -p ~/.pip && \
    printf '%s\n' \
      '[global]' \
      "index-url = ${PIP_INDEX_URL}" \
      "trusted-host = $(echo "${PIP_INDEX_URL}" | sed 's#^http\?://##; s#:.*##')" \
      > ~/.pip/pip.conf && \
    python3 -m pip install --no-cache-dir \
      --extra-index-url https://mirrors.aliyun.com/pypi/simple/ \
      --trusted-host mirrors.aliyun.com \
      pyinstaller requests && \
    git clone "${RUN_NHC}" /src/run_nhc && \
    if [ -n "${RUN_NHC_COMMIT}" ]; then \
        cd /src/run_nhc && git checkout "${RUN_NHC_COMMIT}"; \
    fi && \
    cd /src/run_nhc/node_check && \
    mkdir -p /src/pyi /src/pyi-work && \
    python3 -m PyInstaller --clean --noconfirm --onedir \
        --paths /src/run_nhc \
        --distpath /src/pyi --workpath /src/pyi-work --specpath /src/pyi-work \
        --name checkstatus_bin \
        --add-data "${PWD}/etc:etc" \
        --add-data "${PWD}/tools:tools" \
        --add-data "${PWD}/version:." \
        checkstatus.py && \
    python3 -m PyInstaller --clean --noconfirm --onedir \
        --paths /src/run_nhc \
        --distpath /src/pyi --workpath /src/pyi-work --specpath /src/pyi-work \
        --name generate_template_bin \
        --add-data "${PWD}/etc:etc" \
        --add-data "${PWD}/tools:tools" \
        --add-data "${PWD}/version:." \
        generate_template.py && \
    mkdir -p /out/node_check/bin /out/node_check/log && \
    cp -a /src/pyi/checkstatus_bin/checkstatus_bin /out/node_check/bin/checkstatus && \
    cp -a /src/pyi/checkstatus_bin/_internal /out/node_check/bin/_internal && \
    cp -a /src/pyi/generate_template_bin/_internal/. /out/node_check/bin/_internal/ && \
    cp -a /src/pyi/generate_template_bin/generate_template_bin /out/node_check/bin/generate_template && \
    find /out/node_check/bin/_internal -type f -name 'readline*.so' -delete && \
    cp -a /src/run_nhc/node_check/etc /out/node_check/etc

# setuid wrapper
RUN printf '%s\n' \
      '#include <errno.h>' \
      '#include <stdio.h>' \
      '#include <stdlib.h>' \
      '#include <string.h>' \
      '#include <sys/stat.h>' \
      '#include <unistd.h>' \
      '' \
      'int main(int argc, char *argv[]) {' \
      '    if (setgid(0) != 0 || setuid(0) != 0) {' \
      '        perror("failed to acquire root privileges");' \
      '        return 126;' \
      '    }' \
      '    umask(022);' \
      '    setenv("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", 1);' \
      '    unsetenv("IFS"); unsetenv("LD_PRELOAD"); unsetenv("LD_LIBRARY_PATH");' \
      '    unsetenv("PYTHONPATH"); unsetenv("PYTHONHOME");' \
      '    char **args = calloc((size_t)argc + 1, sizeof(char *));' \
      '    if (args == NULL) { perror("failed to allocate argument vector"); return 125; }' \
      '    args[0] = "/opt/run_nhc/node_check/bin/checkstatus";' \
      '    for (int i = 1; i < argc; ++i) { args[i] = argv[i]; }' \
      '    execv(args[0], args);' \
      '    fprintf(stderr, "failed to execute %s: %s\n", args[0], strerror(errno));' \
      '    return errno == ENOENT ? 127 : 126;' \
      '}' \
      > /tmp/run_nhc_wrapper.c && \
    gcc -O2 -Wall -Wextra -Werror /tmp/run_nhc_wrapper.c -o /out/run_nhc && \
    rm -f /tmp/run_nhc_wrapper.c

FROM ${BASE_IMAGE}

# FROM 之后使用的参数需在此重新声明 (继承上方全局传入值, 不另设默认)
# 注意: ARG 这种写法拿不到 DTK_VERSION 的"版本号"值 —— 基础镜像里 ENV DTK_VERSION
# =dtk2604 (token 格式) 会遮蔽 build-arg 传入的值 (如 26.04), 且遮蔽优先于 ARG 默认值。
# 但这反而可利用: 下面 dtk 相关步骤直接读它, 值恰好就是想要的 token。
# (RUN 之间的 shell 变量不跨层, 每个用到处都自带一行兜底, 不依赖前一层)
SHELL ["/bin/bash", "-c"]
ARG PIP_INDEX_URL
ARG RESOURCE_SERVER_URL
ARG AICC_VERSION
ARG BUILD_TIME
ARG TORCH_VERSION
ARG RCCL_TEST
ARG RCCL_TESTS_COMMIT
ARG RCCL_GPU_TARGETS

# pip 配置 + das-install (trusted-host 自动从 PIP_INDEX_URL 提取 host)
RUN : "${PIP_INDEX_URL:?PIP_INDEX_URL build argument is required (如 .../nightly/dtk2604/+simple/)}" && \
    mkdir -p ~/.pip && \
    printf '%s\n' \
    '[global]' \
    "index-url = ${PIP_INDEX_URL}" \
    "trusted-host = $(echo "${PIP_INDEX_URL}" | sed 's#^http\?://##; s#:.*##')" \
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

# AICC 编译器 (安装脚本从资源服务器拉取 Nightly 下对应大版本的最新包, 安装成功后
# 会在 $DTK_DIR/hcu_llvm.info 写入本次完整版本/commit 等记录; 口径同 vllm-plugin)
# AICC 大版本按基础镜像的 DTK_VERSION (token 格式) 派生, 显式传入 AICC_VERSION 时优先使用
ENV AICC_VERSION=${AICC_VERSION}

RUN TOKEN="${AICC_VERSION:-${DTK_VERSION}}" && \
    case "${TOKEN}" in \
        dtk2604) MAJOR=1.0.0 ;; \
        dtk2610) MAJOR=2.0.0 ;; \
        *) echo "Error: no AICC major mapping for DTK token ${TOKEN}" && exit 1 ;; \
    esac && \
    wget -q "${RESOURCE_SERVER_URL}/ai_cc/Nightly/hcu_llvm_installer.sh" -O /tmp/hcu_llvm_installer.sh && \
    chmod +x /tmp/hcu_llvm_installer.sh && \
    bash /tmp/hcu_llvm_installer.sh --major "${MAJOR}" && \
    rm -f /tmp/hcu_llvm_installer.sh

# ---layer---
RUN TORCH_TAG="torch${TORCH_VERSION//./}" \
    && pip install torch==${TORCH_VERSION} \
    && pip install nvidia-resiliency-ext torch==${TORCH_VERSION} \
    && das-install torchvision ${TORCH_TAG} \
    && das-install flash_attn ${TORCH_TAG} \
    && das-install triton==3.6.0 ${TORCH_TAG} \
    && das-install apex ${TORCH_TAG} \
    && das-install deepspeed ${TORCH_TAG} \
    && das-install transformer-engine ${TORCH_TAG} \
    && das-install deep-ep ${TORCH_TAG} \
    && das-install fast-hadamard-transform-cuda ${TORCH_TAG} \
    && das-install hip-dsa ${TORCH_TAG} \
    && das-install mhc-fused-kernel ${TORCH_TAG} \
    && das-install flash-mla ${TORCH_TAG} \
    && das-install tilelang ${TORCH_TAG} \
    && das-install byte-flux ${TORCH_TAG} \
    && das-install causal-conv1d ${TORCH_TAG} \
    && pip cache purge \
    && rm -f /usr/local/bin/das-install

COPY requirements/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir ms-swift==4.6.0 \
    && pip install --no-cache-dir mcore-bridge==1.6.5 \
    && pip cache purge \
    && rm -f /tmp/requirements.txt

# Megatron-LM-das 源码 (构建上下文即仓库根目录, 3rdparty 子模块由 CI checkout)
ENV PYTHONPATH=/workspace/Megatron-LM
COPY . /workspace/Megatron-LM

# run_nhc 二进制安装 (源码只在 run_nhc_builder 阶段, 这里只拷冻结产物)
COPY --from=run_nhc_builder /out/node_check /opt/run_nhc/node_check
COPY --from=run_nhc_builder /out/run_nhc /usr/local/bin/run_nhc
RUN chmod 4755 /usr/local/bin/run_nhc && \
    mkdir -p /opt/run_nhc/node_check/log && \
    /opt/run_nhc/node_check/bin/checkstatus --version

# rccl-tests (带 MPI 编译, 供 multi/rccltest/test_rccl.sh 做单机/多机通信单测)
# 注意:
#   1. 不能用 rocminfo 探测 gfx 架构 —— 构建期无 GPU 设备, rocminfo 会因缺
#      libhsa-runtime64.so.1 直接退出 127。这里改为显式列出 GPU_TARGETS。
#   2. 基础镜像里 /opt/dtk/lib/libhsa-runtime64.so 是指向 /opt/hyhal 的悬空软链
#      (/opt/hyhal 是宿主机驱动目录, 只在 run 时挂载)，链接期找不到该库。
#      因此先造一个同名 SONAME 的空 stub 供链接, 运行时由真实的 /opt/hyhal 提供。
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

# /etc/image.info (口径同基础镜像 dtk 段: ASCII 字 + 名称/构建时间, 容器启动时由
# ENTRYPOINT /usr/local/bin/hcu_image_info 打印); figlet 字体与基础镜像同源
# (资源服务器 pkg/slant.flf), 构建后重新生成, 覆盖基础镜像的 dtk 记录
RUN cd /tmp && \
    wget -q "${RESOURCE_SERVER_URL}/pkg/slant.flf" -O /tmp/slant.flf && \
    { figlet -f slant -d /tmp -w 80 megaton; \
      echo "dtk    : ${DTK_VERSION}"; \
      echo "built  : ${BUILD_TIME:-$(date +%Y%m%d-%H%M)}"; } > /etc/image.info && \
    rm -f /tmp/slant.flf
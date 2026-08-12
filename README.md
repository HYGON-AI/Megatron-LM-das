# <div align="center"><strong>hcu-megatron</strong></div>
## 简介
本项目通过替换megatron的函数或类，引入新的特性或者实现更好的性能。

## 环境依赖

> 版本依赖：dtk >= 25.04，transformer-engine >= 2.4.0，torch >= 2.6.0

## 下载方式
1. 在线 git clone 方式：
  ```bash
  git clone -b 最新分支 --recurse-submodules https://github.com/HYGON-AI/Megatron-LM-das.git
  ```
2. 离线下载方式：
  ```bash
  1. 下载仓库离线代码包
  2. 点击 `Megatron-LM@版本号`，下载对应版本的 Megatron-LM 离线代码包
  3. 将 Megatron-LM 离线代码包解压到 `hcu_megatron/Megatron-LM` 目录下
  ```


## 使用方式

支持LLM pretrain, sft 和 vlm 的sft

使用方式详见[docs/getting-started.md](./docs/getting-started.md)。

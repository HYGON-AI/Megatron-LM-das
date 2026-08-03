## <div align="center"><strong>hcu-megatron</strong></div>
### 简介
本项目通过替换megatron的函数或类，引入新的特性或者实现更好的性能。

## 安装

> 版本依赖：dtk >= 25.04，transformer-engine >= 2.4.0，torch >= 2.6.0

### 方式一：pip 安装 whl 包

直接下载已编译好的 whl 包安装：

```bash
pip install hcu_megatron*.whl
```

### 方式二：源码下载与编译

**1. 下载源码(git或者离线)**

git 方式：

```bash
git clone -b core_v0.18.0 --recurse-submodules http://42.228.13.241:10068/hcutoolkit/deeplearing/hcu_megatron.git
```

离线方式：

1. 下载仓库离线代码包
2. 点击 `Megatron-LM@版本号`，下载对应版本的 Megatron-LM 离线代码包
3. 将 Megatron-LM 离线代码包解压到 `hcu_megatron/Megatron-LM` 目录下

**2. 编译并安装**

```bash
cd hcu_megatron
python3 setup.py -v bdist_wheel
pip install dist/hcu_megatron*.whl
```

## 使用方式

### 修改pretrain_gpt.py
1、如果通过下载源码方式使用hcu megatron, 不需要改动；
2、如果通过安装hcu_megatron*.whl方式使用hcu megatron，由于词汇层并行等特性需要修改`pretrain_gpt.py`，建议使用项目下的`pretrain_gpt.py`覆盖Megatron-LM目录中的`pretrain_gpt.py`文件。

### 运行训练

进入 `examples` 目录，选择对应模型的执行脚本：

```
examples/
├── deepseek_v3
├── gpt3
├── llama
├── mixtral
└── qwen
```

以 DeepSeek V3 671B 为例：

```bash
cd examples/deepseek_v3
# num_nodes 为运行的节点数 默认8卡/机
bash run_deepseekv3_671B.sh hostfile_deepseekv3_671B num_nodes
```

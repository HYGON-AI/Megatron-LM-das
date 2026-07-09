## <div align="center"><strong>hcu-megatron</strong></div>
### 简介
本项目通过替换megatron的函数或类，引入新的特性或者实现更好的性能。

### 安装
+ dtk版本大于25.04 && transformer-engine版本大于2.4.0

#### 1、使用pip方式安装
+ 下载hcu-megatron whl包，并安装
```
pip install hcu_megatron*  # 下载的hcu-megatron whl包
```

#### 2、使用源码编译方式安装
```
git clone http://10.16.6.30/dcutoolkit/deeplearing/hcu_megatron.git  # 根据需要切换分支
python3 setup.py -v bdist_wheel
pip install dist/hcu_megatron*
```

### 注意事项
+ 使用hcu-megatron时,需要使用对应版本的megatron


### 使用方式
+ 获取 Megatron-LM并指定分支
```
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_r0.12.0    # 根据hcu-megatron版本，选择对应的Megatron-LM版本
```
+ 修改Megatron-LM目录下的pretrain_gpt.py文件，增加一行引用
```
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)

from hcu_megatron import megatron_adaptor     # 新增一行代码
```
+ 特性介绍见[features](./features.md)文件
+ 运行模型训练模型，可参考[deepseek训练脚本](http://112.11.119.99:10068/dcutoolkit/deeplearing/hcu_megatron/-/blob/core_v0.15.0/examples/deepseek_v3/run_deepseekv3_671B.sh)。
+ 运行命令

```shell
# num_nodes是需要运行的节点数
bash run_deepseekv3_671B.sh hostfile_deepseekv3_671B num_nodes
```

处理多模态数据集coco

1. 数据集下载和处理
链接: https://modelscope.cn/datasets/modelscope/coco_2014_caption

更新之后的数据集是自动构建的, 如果不清楚怎么自动获取的, 也可以手动下载
```python
# 1. 参考dataset_infos.json文件中的链接: 手动下载zip文件
"download_checksums": {
    "https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/coco_2014_caption/train2014.csv.zip": {
        "num_bytes": 536720,
        "checksum": "ec2c53c38fcec918035c3b02c8e7469ca5120a2672098617c91fe041ab4c4968"
    },
    "https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/coco_2014_caption/val2014.csv.zip": {
        "num_bytes": 963442,
        "checksum": "541949e1c46bd8cf087b7bee505dc2c32a5fb860cdcf04308aa9d8b60559f230"
    }
},
wget https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/coco_2014_caption/train2014.csv.zip
# 2. 解压压缩包
unzip train2014.csv.zip
# 3. 根据解压出来的csv文件下载图片, 可用提供的脚本download_coco.py多线程下载
python download_coco.py
# 4. 将text和images进行配对, 组成一个json文件, 可用提供的脚本csv2json.py处理
python csv2json.py
```
输入的数据格式如下：
```json
# sft:
{
    "conversations": [
        {
            "role": "user",
            "content": "挂在交通灯杆上的是什么？<image>"
        },
        {
            "role": "assistant",
            "content": "一个绿色的街牌挂在交通灯杆上。"
        }
    ],
    "images": [
        {
            "image_path": "7_0.png",
        }
    ]
}
```

2. 按照下面的格式构建数据集config
```json
{
    "train_data_infos": {
        "coco_caption": {
            "path": "/path/to/filename.jsonl",
            "probability": 1.0,
            "sample_rate": 1.0
        }
    }
}
```
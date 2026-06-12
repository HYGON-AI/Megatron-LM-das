import csv
import json
import os
from urllib.parse import urlparse

def csv_to_megatron_json(csv_file_path, output_json_path, image_dir):
    """
    将CSV文件转换为Megatron多模态训练所需的JSON格式。

    Args:
        csv_file_path (str): 输入CSV文件路径
        output_json_path (str): 输出JSON文件路径
        image_dir (str): 图片文件目录路径
    """
    processed_count = 0
    error_count = 0

    with open(csv_file_path, 'r', encoding='utf-8') as csv_file, \
         open(output_json_path, 'w', encoding='utf-8') as json_file:

        reader = csv.reader(csv_file)
        # 跳过标题行（如果存在）
        first_row = next(reader)
        if first_row[0] == 'uniq_id':
            print("检测到标题行，已自动跳过")
        else:
            # 如果第一行不是标题，也处理它
            process_row(first_row, json_file, image_dir)
            processed_count += 1

        # 处理剩余行
        for row_num, row in enumerate(reader, start=2):
            try:
                if len(row) >= 4:
                    process_row(row, json_file, image_dir)
                    processed_count += 1

                    # 每处理1000条打印一次进度
                    if processed_count % 1000 == 0:
                        print(f"已处理 {processed_count} 条数据...")
                else:
                    print(f"警告：第{row_num}行数据不完整，已跳过")
                    error_count += 1

            except Exception as e:
                print(f"错误：处理第{row_num}行时出现异常 - {e}")
                error_count += 1

    return processed_count, error_count


def process_row(row, json_file, image_dir):
    """
    处理单行数据并写入JSON。

    Args:
        row (list): CSV行数据
        json_file (file): 打开的JSON文件对象
        image_field_name (str): 图片字段名称
    """
    uniq_id, image_id, caption, image_url = row[0], row[1], row[2], row[3]

    # 从URL提取图片文件名
    image_path = image_dir + '/' + uniq_id + '_' + image_id + '_' + os.path.basename(urlparse(image_url).path)

    # 清理caption文本（去除首尾空格）
    caption = caption.strip()

    # 构建JSON对象
    json_obj = {
        "conversations": [
            {
                "role": "user",
                "content": "介绍一下图片内容<image>"
            },
            {
                "role": "assistant",
                "content": caption
            }
        ],
        "images": [
            {
                "image_path": image_path,
            }
        ]
    }

    # 写入JSON行
    json_file.write(json.dumps(json_obj, ensure_ascii=False) + '\n')


def validate_json(json_path, num_samples=5):
    """
    验证生成的JSON文件，打印前几个样本。

    Args:
        json_path (str): JSON文件路径
        num_samples (int): 打印的样本数量
    """
    print(f"\n验证生成的JSON文件：{json_path}")
    print("-" * 50)

    with open(json_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data = json.loads(line)
            print(f"样本{i+1}: {data}")

    # 统计总行数
    with open(json_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    print(f"\n总共生成 {total_lines} 条JSON数据")


# 主程序
if __name__ == "__main__":
    # 配置参数
    CSV_INPUT = "train2014.csv"        # 你的CSV文件路径
    JSON_OUTPUT = "train2014.json"  # 输出JSON文件路径
    IMAGE_DIR = "/path/to/coco_2014_caption/coco_images"           # 图片文件目录路径

    print("开始处理CSV文件...")
    print(f"输入文件: {CSV_INPUT}")
    print(f"输出文件: {JSON_OUTPUT}")
    print(f"图片目录: {IMAGE_DIR}")
    print("-" * 50)

    # 检查输入文件是否存在
    if not os.path.exists(CSV_INPUT):
        print(f"错误：找不到输入文件 {CSV_INPUT}")
        exit(1)

    # 执行转换
    success, errors = csv_to_megatron_json(CSV_INPUT, JSON_OUTPUT, image_dir=IMAGE_DIR)

    # 打印处理结果
    print("\n处理完成！")
    print(f"成功处理: {success} 条")
    print(f"错误/跳过: {errors} 条")

    # 验证输出
    if success > 0:
        validate_json(JSON_OUTPUT)

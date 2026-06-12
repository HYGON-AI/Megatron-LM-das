import csv
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from tqdm import tqdm  # 如果没有安装，先执行 pip install tqdm

# --- 配置 ---
csv_file_path = 'train2014.csv'  # 替换为你的CSV文件路径
download_dir = 'coco_images'     # 下载的图片存放目录
max_workers = 20                 # 并发线程数，可根据网络调整，一般10-30
request_timeout = 15             # 单个请求超时秒数
max_retries = 3                  # 单个链接失败后的最大重试次数
# --- 配置结束 ---

# 创建下载目录
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

def download_single_image(task):
    """
    下载单张图片的worker函数。
    包含重试逻辑和错误返回。
    """
    uniq_id, image_id, image_url = task

    # 准备保存的文件名
    parsed_url = urlparse(image_url)
    original_filename = os.path.basename(parsed_url.path)
    save_path = os.path.join(download_dir, f"{uniq_id}_{image_id}_{original_filename}")

    # 如果文件已存在，则跳过（断点续传）
    if os.path.exists(save_path):
        return True, uniq_id, "Skipped (already exists)"

    # 带重试的下载循环
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(image_url, stream=True, timeout=request_timeout)
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True, uniq_id, "Success"

        except requests.exceptions.RequestException as e:
            if attempt == max_retries:
                return False, uniq_id, f"Failed after {max_retries} attempts: {e}"
            # 非最后一次失败，短暂等待后重试
            time.sleep(0.5)

    return False, uniq_id, "Unknown error"

# 1. 从CSV文件读取并解析所有任务
tasks = []
with open(csv_file_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    # 如果你的文件有表头，取消下面这行的注释来跳过表头
    next(reader, None)
    for row in reader:
        if len(row) >= 4:
            uniq_id, image_id, caption, url = row[0], row[1], row[2], row[3]
            tasks.append((uniq_id, image_id, url))

print(f"共发现 {len(tasks)} 个下载任务。")

# 2. 创建线程池并执行高并发下载
failed_list = []
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # 提交所有任务
    future_to_task = {executor.submit(download_single_image, task): task for task in tasks}

    # 使用tqdm显示进度条
    progress_bar = tqdm(total=len(tasks), desc="下载进度", unit="张")

    for future in as_completed(future_to_task):
        success, uid, message = future.result()
        if not success:
            failed_list.append((uid, message))
        progress_bar.update(1)

    progress_bar.close()

# 3. 输出最终结果
print(f"\n下载任务结束。成功: {len(tasks) - len(failed_list)}, 失败: {len(failed_list)}")
if failed_list:
    print("失败链接详情：")
    for uid, msg in failed_list:
        print(f"  ID: {uid} - {msg}")

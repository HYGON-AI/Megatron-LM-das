# copyright (c) 2024 tencent inc. all rights reserved.
# guanyouhe@tencent.com
from copy import deepcopy
import math
from typing import Dict, Sequence, Optional, Tuple
from types import SimpleNamespace

from PIL import Image
from torch.utils.data.dataloader import default_collate
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers import AutoProcessor
try:
    from transformers import Qwen3VLProcessor
except:
    Qwen3VLProcessor = None

from dcu_megatron.core.datasets.utils import print_rank_0, get_iterator, random_pad_list
from dcu_megatron.core.datasets.mm_dataset import (
    MultiModalDataset,
    fetch_images,
    convert_conversations,
)

# copy from: https://github.com/QwenLM/Qwen2-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
# 目前只保存读image的
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def resize_image(
    ele: dict[str, str],
    image: Image.Image,
    default_min_pixels: int = None,
    default_max_pixels: int = None,
    size_factor: int = IMAGE_FACTOR,
) -> Image.Image:
    # resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        default_min_pixels = default_min_pixels or MIN_PIXELS
        default_max_pixels = default_max_pixels or MAX_PIXELS
        width, height = image.size
        min_pixels = ele.get("min_pixels", default_min_pixels)
        max_pixels = ele.get("max_pixels", default_max_pixels)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image

def pad_and_split_pixel_values(
    cp_size: int,
    hw_factor: int,
    pixel_values: list[torch.Tensor],
    image_grid_thws: list[torch.Tensor],
):
    """将 pixel_values 按 image_grid_thw 逐图拆分，分配到各 CP rank，并对齐到 hw_factor。

    hw_factor 由 context_parallel_size * (seq_parallel 时再 * tp_size) * 4 计算得到。
    qwen2vl/qwen2.5vl/qwen3vl 的 merge_size 均为 2，因此 pixel_values 天然是 4 对齐的。

    返回:
        new_pixel_values: 重新排序并可能补零后的 pixel_values 列表
        new_image_grid_thws: 对应的 grid_thw 列表（含 padding 占位项）
        cp_img_num: 每个 CP rank 的图像 token 数量
        images_padded: 每个 CP rank 是否需要 padding
    """
    assert len(pixel_values) == len(image_grid_thws)

    # 第一步：将多图聚合的 pixel_values 拆成单图维度
    split_pixel_values = []
    split_image_grid_thws = []
    for pixel_value, image_grid_thw in zip(pixel_values, image_grid_thws):
        split_image_grid_thw = list(torch.split(image_grid_thw, 1, dim=0))
        split_image_grid_thws.extend(split_image_grid_thw)
        slice_begin = 0
        for ele in split_image_grid_thw:
            slice_end = slice_begin + ele.prod().item()
            split_pixel_values.append(pixel_value[slice_begin:slice_end].clone())
            slice_begin = slice_end

    pixel_values = split_pixel_values
    image_grid_thws = split_image_grid_thws
    img_num = len(image_grid_thws)

    # 第二步：将图像均匀分配到各 CP rank
    img_num_per_rank = img_num // cp_size
    img_num_remain = img_num % cp_size
    cp_img_num = []
    for i in range(cp_size):
        cp_img_num.append(img_num_per_rank)
        if i < img_num_remain:
            cp_img_num[i] += 1

    # 第三步：按 rank 聚合 pixel_values，必要时补零对齐
    img_idx = 0
    new_pixel_values = []
    new_image_grid_thws = []
    images_padded = []
    for i in range(cp_size):
        seq_len = 0
        img_begin_idx = img_idx
        img_end_idx = img_begin_idx + cp_img_num[i]
        img_idx += cp_img_num[i]

        for j in range(img_begin_idx, img_end_idx):
            seq_len += pixel_values[j].size(0)
            new_pixel_values.append(pixel_values[j])
            new_image_grid_thws.append(image_grid_thws[j])

        image_padded = 0 != seq_len % hw_factor
        if image_padded:
            padded_seqlen = (seq_len + hw_factor - 1) // hw_factor * hw_factor - seq_len
            # hw_factor 已包含 *4，因此 padded_seqlen 必然是 4 的倍数
            assert padded_seqlen > 0 and padded_seqlen % 4 == 0
            # 补零 pixel_values，并插入一个占位 grid：
            #   t=1, h=2, w=padded_seqlen//2 → 总 token 数 = 1*2*(padded_seqlen//2) = padded_seqlen
            # 下游通过 images_padded 标记跳过该占位项
            new_pixel_values.append(
                torch.zeros(
                    [padded_seqlen, pixel_values[0].size(-1)],
                    dtype=pixel_values[0].dtype,
                    device=pixel_values[0].device,
                )
            )
            new_image_grid_thws.append(
                torch.tensor(
                    [[1, 2, padded_seqlen // 2]],
                    dtype=image_grid_thws[0].dtype,
                    device=image_grid_thws[0].device,
                )
            )
            cp_img_num[i] += 1
        images_padded.append(int(image_padded))

    return new_pixel_values, new_image_grid_thws, cp_img_num, images_padded


# MegatronLegacyTokenizer / _HuggingFaceTokenizer / Qwen2VLTokenizer 已删除。
# 原 monkey-patch 方式不走 __init__，无法正确加载 HF tokenizer。
# 现改为在 build_train_valid_test_data_iter 中直接从 HF tokenizer 注入属性。


class Qwen2VlDataset(MultiModalDataset):
    """Qwen2-VL / Qwen2.5-VL / Qwen3-VL 多模态对话数据集。

    数据流全貌
    ==========
    JSONL 原始数据
        │
        ▼
    MegaIndexedJsonlDatasetMM.__iter__()
        读取 JSONL 行 → yield {"json_data": {...}, "domain_line": 1, ...}
        │
        ▼
    Qwen2VlDataset.__iter__()                     ← 当前类
        ① 从 json_data 加载图片 (fetch_images)
        ② 解析对话结构 (convert_conversations: <image> → {"type":"image","image":"0"})
        ③ convert_example() 处理为训练 tensor
        │
        ▼
    DataCollatorForQwen2Vl.__call__()
        拼接 batch、pad pixel_values、计算 RoPE position_ids、生成 loss_mask

    核心方法
    ========
    process_vision(images)     — 图片 → pixel_values + image_grid_thw
    padding_vision_token(text) — 将 chat_template 中的 <|image_pad|> 按 grid 展开
    convert_example()          — 单条样本的完整处理管线
    gen_label_mask()           — 计算哪些 token 属于 label（定义在父类 MultiModalDataset）
    """

    def __init__(
        self,
        min_pixels_num,
        max_pixels_num,
        use_for_hf,
        mask_history,
        *args,
        **kwargs,
    ):
        """
        Args:
            min_pixels_num: 图片最小像素数（传给 image_processor）
            max_pixels_num: 图片最大像素数
            use_for_hf:      HuggingFace 兼容模式（label shift 方向不同）
            mask_history:    多轮对话只保留最后一轮为 label
            *args, **kwargs: 透传给 MultiModalDataset → BaseIndexedJsonlDataset
        """
        super().__init__(*args, **kwargs)
        self.min_pixels_num = min_pixels_num
        self.max_pixels_num = max_pixels_num
        self.use_for_hf = use_for_hf
        self.mask_history = mask_history

    # =========================================================================
    # 图片 → pixel_values + image_grid_thw
    # =========================================================================

    def process_vision(self, images, videos=None):
        """调用 Qwen2VLImageProcessor 将 PIL Image 列表转为模型输入。

        Qwen2-VL 的 image_processor 不同于标准 ViT：
          - 不将图片切为固定 patch，而是根据图片尺寸动态调整
          - 输出 pixel_values 是一维序列（而非 [C, H, W]），配合 image_grid_thw 还原空间结构

        Returns:
            BatchFeature {
                "pixel_values":   shape [total_pixels, C]   — 拼接后的像素值
                "image_grid_thw": shape [num_images, 3]      — 每张图的 (t, h, w)
            }

        示例：一张 560×560 图片，merge_size=2，patch_size=14
          - spatial_merge_size=2 → 每 2×2 patch 合并 → 有效尺寸变为 20×20
          - t=1 (图片固定), h=20, w=20 → image_grid_thw = [[1, 20, 20]]
          - pixel_values 总 token 数 = 1*20*20 = 400
        """
        if images is not None:
            image_inputs = self.image_processor(images=images, return_tensors="pt")
        else:
            image_inputs = {}

        if videos is not None:
            videos_inputs = self.image_processor(images=None, videos=videos, return_tensors="pt")
        else:
            videos_inputs = {}

        return BatchFeature(data={**image_inputs, **videos_inputs})

    # =========================================================================
    # 图像 token 占位符展开
    # =========================================================================

    def padding_vision_token(self, text: str, image_grid_thw, video_grid_thw=None):
        """将 chat_template 输出的 <|image_pad|> 占位符按实际图像 token 数展开。

        背景
        ----
        Qwen2-VL 的 chat_template 对每张图片只输出一个 <|image_pad|> 占位符。
        但模型实际需要 N 个 image token 来编码这张图，N = (t * h * w) / merge_size^2。

        此函数将字符串中的每个 <|image_pad|> 替换为 N 个 <|image_pad|>，
        使得 tokenizer 编码后 input_ids 长度 = 文本 token 数 + 图像 token 数。

        参数
        ----
        text: apply_chat_template 后的文本
        image_grid_thw: shape [num_images, 3]，每张图处理后的 (t, h, w)

        示例
        ----
        text = "...<|vision_start|><|image_pad|><|vision_end|>..."
        image_grid_thw = [[1, 20, 20]]  → 共 400 个 image token (1*20*20/1)
        输出: "...<|vision_start|><|image_pad|><|image_pad|>...(×400)<|vision_end|>..."
        """
        merge_length = self.image_processor.merge_size**2
        if image_grid_thw is not None:
            index = 0
            while self.tokenizer.image_token in text:
                text = text.replace(
                    self.tokenizer.image_token,
                    "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length), 1
                )
                index += 1
            text = text.replace("<|placeholder|>", self.tokenizer.image_token)

        if video_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            while self.tokenizer.video_token in text:
                text = text.replace(
                    self.tokenizer.video_token,
                    "<|placeholder|>" * (video_grid_thw[index].prod() // merge_length), 1
                )
                index += 1
            text = text.replace("<|placeholder|>", self.tokenizer.video_token)

        return text

    # =========================================================================
    # 图像 token 计数
    # =========================================================================

    def get_image_token_cnt(self, image_grid_thw, video_grid_thw=None):
        """计算 image_grid_thw 对应的视觉 token 总数。

        用于校验：实际 input_ids 中 <|image_pad|> 的出现次数应该等于此值。
        token 数 = sum(t_i * h_i * w_i) / merge_size^2
        """
        merge_length = self.image_processor.merge_size**2
        total_cnt = torch.tensor(0, dtype=torch.long)
        if image_grid_thw is not None:
            for i in range(image_grid_thw.shape[0]):
                total_cnt += image_grid_thw[i].prod() // merge_length

        if video_grid_thw is not None:
            for i in range(video_grid_thw.shape[0]):
                total_cnt += video_grid_thw.prod() // merge_length

        return total_cnt.item()

    # =========================================================================
    # 单条样本 → 训练 tensor（核心管线）
    # =========================================================================

    def convert_example(
        self,
        example,
        conversations,
        imgs,
        domain_states,
        tools=None,
    ):
        """将一条对话样本转为训练所需的 tensor 字段。

        完整流程
        --------
        step 1. process_vision(imgs)
                PIL Images → pixel_values + image_grid_thw

        step 2. apply_chat_template(conversations)
                对话列表 → 文本字符串
                例如: "<|im_start|>user\n...<image>\n<|im_end|>\n<|im_start|>assistant\n..."

        step 3. padding_vision_token(text, image_grid_thw)
                将单个 <|image_pad|> 占位符展开为 N 个

        step 4. tokenizer(text)
                文本 → input_ids (含 image token)

        step 5. gen_label_mask(conversations, image_grid_thw)
                确定哪些 token 属于 assistant 回复 → 只有这些参与 loss 计算
                非 assistant 的 token 以及 mask_history 被 mask 的 token → labels = -100

        step 6. pad / truncate
                补齐到 max_seq_len + 1（+1 是为了 shift 做准备）
                过长则右侧截断（保留最新内容）

        step 7. shift
                input_ids = input_ids[:-1]   — 模型输入，去掉最后一个 token
                labels    = labels[1:]        — 预测目标，去掉第一个 token
                （HF 模式下 labels = labels[:-1]）

        step 8. 校验
                image token 数是否匹配、是否整条都 ignore

        产生的字段
        ----------
        input_ids:        [seq_len]          — 模型输入 token
        labels:           [seq_len]          — 训练目标（prompt 部分为 -100）
        attention_mask:   [seq_len]          — 1=有效, 0=padding
        pixel_values:     [total_pixels, C]  — vit 输入（可能为 None）
        image_grid_thw:   [num_images, 3]    — 每张图的 (t, h, w)
        image_input_mask: [seq_len]          — True 的位置是 image token
        prompt_len:       标量               — prompt 部分的 token 长度
        domain_line:      标量               — 累计消费样本数

        Returns
        -------
        dict 或 None（图片损坏/尺寸异常/全 ignore 时返回 None 跳过该样本）
        """
        # ── step 1: 图片处理 ──
        media_info = self.process_vision(imgs)
        image_grid_thw = media_info.get("image_grid_thw", None)

        # ── step 2: 对话模板 → 文本 ──
        # add_generation_prompt=False 表示不在末尾追加 assistant 前缀
        # add_vision_id=False 表示不在 <|vision_start|> 后插入图片编号
        all_text = self.processor.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False,
            add_vision_id=False,
        )
        # ── step 3: 展开 image token 占位符 ──
        all_text = self.padding_vision_token(all_text, image_grid_thw)
        # ── step 4: tokenize ──
        all_text_tokenizer = self.tokenizer._tokenizer(all_text, padding=False)
        input_ids = all_text_tokenizer.input_ids
        attention_mask = all_text_tokenizer.attention_mask

        # ── step 5: label mask ──
        labels = torch.tensor(input_ids, dtype=torch.int64)
        label_mask = self.gen_label_mask(conversations, imgs, tools, image_grid_thw, rm_bos=False)
        for mask in label_mask:
            labels[mask[0]:mask[1]] = -100
        prompt_len = label_mask[-1][-1]  # prompt 结束位置 = 最后一个 mask 区间的末尾
        labels = labels.tolist()

        effective_max_len = self.max_seq_len

        # ── step 6: pad / truncate ──
        # 自回归训练需要 max_len + 1 个 token：shift 后得到 max_len 对 (input, label)
        target_len = effective_max_len + 1
        if len(input_ids) < target_len:
            pad_len = target_len - len(input_ids)
            if self.moe_pad_with_random_token:
                # MoE 场景：用随机 token 填充（避免 pad_token 导致负载不均）
                ban_token_ids = [
                    self.tokenizer.image_token_id, self.tokenizer.video_token_id,
                    self.tokenizer.vision_start_token_id, self.tokenizer.vision_end_token_id
                ]
                input_ids = random_pad_list(input_ids, pad_len, ban_token_ids)
            else:
                input_ids += [self.tokenizer._tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len
            attention_mask += [0] * pad_len

        # ── step 7: shift ──
        # 标准 AR：input 去掉最后一个 token，label 去掉第一个 token
        # 例如 input=[1,2,3], label=[1,2,3] → input=[1,2], label=[2,3]
        # HF 模式：input 和 label 对齐去掉最后一个
        input_ids = input_ids[:-1]
        attention_mask = attention_mask[:-1]
        if self.use_for_hf:
            labels = labels[:-1]
        else:
            labels = labels[1:]

        # 过长样本：右侧截断
        if len(input_ids) > effective_max_len:
            input_ids = input_ids[-effective_max_len:]
            labels = labels[-effective_max_len:]
            attention_mask = attention_mask[-effective_max_len:]

        # ── 打包输出 ──
        example["input_ids"] = torch.tensor(input_ids, dtype=torch.int64)
        example["labels"] = torch.tensor(labels, dtype=torch.int64)
        example["attention_mask"] = torch.tensor(attention_mask, dtype=torch.bool)
        example["pixel_values"] = media_info.get("pixel_values", None)
        example["image_grid_thw"] = image_grid_thw
        if self.image_token_id is not None:
            assert self.tokenizer.image_token_id == self.image_token_id
        example["image_input_mask"] = example["input_ids"] == self.tokenizer.image_token_id

        domain_states.domain_lines += example["domain_line"]

        # ── step 8: 校验 ──
        # image token 数必须匹配 image_grid_thw 计算出的总数
        sum_image_token = example["image_input_mask"].sum().cpu().item()
        total_image_token = self.get_image_token_cnt(image_grid_thw)
        all_ignore = torch.all(example["labels"] == -100).item()
        assert total_image_token >= sum_image_token, \
            f"image token mismatch: in_ids={sum_image_token}, grid={total_image_token}"
        # total > sum：有部分 image token 在截断时被丢弃了 → 跳过该样本
        # all_ignore：整条样本没有可学习的 label → 跳过
        if total_image_token > sum_image_token or all_ignore:
            print(f"Abort Sample at dp-rank:{self.underlying.dp_rank}")
            return None

        example["domain_line"] = torch.tensor(domain_states.domain_lines, dtype=torch.int64)
        example["prompt_len"] = torch.tensor(prompt_len, dtype=torch.int64)
        domain_states.domain_lines = 0
        return example

    # =========================================================================
    # 迭代器：JSONL 行 → 训练就绪的 tensor dict
    # =========================================================================

    def __iter__(self):
        """逐条处理多模态样本。

        每轮迭代
        --------
        1. 从底层 MegaIndexedJsonlDatasetMM 取出一条 json_data
        2. 加载图片 (fetch_images)：支持本地路径 / tar 包 / LMDB
        3. 图片校验：尺寸 >= IMAGE_FACTOR、宽高比 <= MAX_RATIO
        4. 转换对话格式 (convert_conversations)：<image> → {"type":"image",...}
        5. Qwen3VL 额外 resize（Qwen3VLProcessor 不自动 resize）
        6. convert_example() → 训练 tensor
        7. yield（无效样本则 continue 跳过）
        """
        # domain_states 是跨样本的可变状态容器，用 SimpleNamespace 避免创建新对象
        domain_states = SimpleNamespace(domain_lines=0)
        for example in self.underlying:
            json_data = example["json_data"]

            # ── 加载图片 ──
            imgs = None
            if 'images' in json_data and len(json_data['images']) > 0:
                imgs = fetch_images(json_data['images'], self.tar_dir, self.lmdb_port)
                # 图片有效性校验：None / 过小 / 宽高比过大 → 跳过
                imgs_valid = True
                for img in imgs:
                    if img is None:
                        imgs_valid = False
                        break
                    width, height = img.size
                    if width < IMAGE_FACTOR or height < IMAGE_FACTOR:
                        imgs_valid = False
                        break
                    if max(height, width) / min(height, width) > MAX_RATIO:
                        imgs_valid = False
                        break
                if not imgs_valid:
                    domain_states.domain_lines += example["domain_line"]
                    print(f"Abort Sample at dp-rank:{self.underlying.dp_rank}[invalid image]")
                    continue

            # ── 解析对话 ──
            # 将 content 中的 <image>/<video> 转为 processor 接受的格式
            # 例如 "看图<image>回答" → [{"type":"text","text":"看图"}, {"type":"image","image":"0"}, ...]
            conversations = convert_conversations(json_data['conversations'])
            tools = json_data.get('tools', None)
            assert len(conversations) > 1, "SFT 至少需要一轮对话 (user + assistant)"
            del example["json_data"]

            # ── Qwen3VL 额外 resize ──
            # Qwen3VL 使用 Qwen2VLImageProcessorFast，不会在 process_vision 中做 resize，
            # 需要在此处提前 resize；Qwen2VL/Qwen2.5VL 的 processor 会在内部处理
            if Qwen3VLProcessor is not None and isinstance(self.processor, Qwen3VLProcessor) \
            and imgs is not None:
                imgs = [
                    resize_image(ele, img, self.min_pixels_num, self.max_pixels_num)
                    for ele, img in zip(json_data['images'], imgs)
                ]

            # ── 转换样本 ──
            example_copy = deepcopy(example)
            example_copy = self.convert_example(
                example_copy, conversations, imgs, domain_states, tools
            )
            if example_copy is None:
                continue

            yield example_copy


# =============================================================================
# 统一的 3D RoPE position ID 计算
#
# 三个模型的差异仅在于 temporal position 的计算方式：
#   qwen2vl:    顺序递增 0, 1, 2, ...
#   qwen2.5vl:  按 second_per_grid_t * tokens_per_second 间隔递增
#   qwen3_vl*:  恒为 0（temporal 信息由 timestamp token 编码，且 video_grid_thw 被预处理为每帧 t=1）
# =============================================================================

def _compute_t_index_qwen2vl(llm_grid_t, llm_grid_h, llm_grid_w, second_per_grid_t):
    """qwen2vl / qwen3_vl: temporal pos 顺序递增 0, 1, 2, ...
       对于 qwen3_vl，video_grid_thw 已被预处理为每帧 t=1，llm_grid_t 恒为 1，t_index 恒为 [0]"""
    return torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()


def _compute_t_index_qwen25vl(llm_grid_t, llm_grid_h, llm_grid_w, second_per_grid_t):
    """qwen2.5vl: temporal pos 按 second_per_grid_t * 2 间隔递增"""
    tokens_per_second = 2  # 所有 qwen2.5vl 写死
    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
    time_tensor = expanded_range * second_per_grid_t * tokens_per_second
    return time_tensor.long().flatten()


def _compute_hw_indices(llm_grid_t, llm_grid_h, llm_grid_w):
    """height / width 索引：三个模型完全相同"""
    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
    return h_index, w_index


def _compute_text_only_position_ids(input_ids, attention_mask):
    """纯文本 fallback：三个模型完全相同"""
    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
    else:
        position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device)
            .view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        )
        mrope_position_deltas = torch.zeros(
            [input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype,
        )
    return position_ids, mrope_position_deltas


def get_rope_index(
    input_ids: torch.LongTensor,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    tokenizer=None,
    spatial_merge_size: int = 2,
    model_arch: str = "qwen2vl",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """统一的 3D RoPE position ID 计算，支持 qwen2vl / qwen2.5vl / qwen3_vl[._moe].

    三个模型的差异仅在于 temporal position 的计算方式：
      qwen2vl:    顺序 0, 1, 2, ...
      qwen2.5vl:  按 second_per_grid_t * tokens_per_second 间隔递增
      qwen3_vl*:  恒为 0（temporal 信息由 timestamp token 编码）
    """
    image_token_id = tokenizer.image_token_id
    video_token_id = tokenizer.video_token_id
    vision_start_token_id = tokenizer.vision_start_token_id

    # ---- qwen3vl 专属预处理：按 timestamp 拆分 video_grid_thw，确保每帧 t=1 ----
    if model_arch in ("qwen3_vl_moe", "qwen3_vl") and video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    # ---- 纯文本路径：三个模型完全相同 ----
    if input_ids is None or (image_grid_thw is None and video_grid_thw is None):
        return _compute_text_only_position_ids(input_ids, attention_mask)

    # ---- 多模态路径：以下逻辑三个模型共享 ----
    total_input_ids = input_ids
    if attention_mask is None:
        attention_mask = torch.ones_like(total_input_ids)
    attention_mask = attention_mask.to(total_input_ids.device)

    position_ids = torch.ones(
        3, input_ids.shape[0], input_ids.shape[1],
        dtype=input_ids.dtype, device=input_ids.device,
    )
    image_index, video_index = 0, 0
    mrope_position_deltas = []

    for i, input_ids in enumerate(total_input_ids):
        input_ids = input_ids[attention_mask[i] == 1]
        vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
        vision_tokens = input_ids[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        input_tokens = input_ids.tolist()
        llm_pos_ids_list: list = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums

        for _ in range(image_nums + video_nums):
            # 定位下一个 image / video token
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1

            if ed_image < ed_video:
                t, h, w = (image_grid_thw[image_index][0],
                           image_grid_thw[image_index][1],
                           image_grid_thw[image_index][2])
                second_per_grid_t = 0.0
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (video_grid_thw[video_index][0],
                           video_grid_thw[video_index][1],
                           video_grid_thw[video_index][2])
                second_per_grid_t = (second_per_grid_ts[video_index].item()
                                     if second_per_grid_ts is not None else 1.0)
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t.item(), h.item() // spatial_merge_size, w.item() // spatial_merge_size,
            )
            text_len = ed - st

            # 文本段 position ID（三个模型相同）
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            # 视觉段 position ID：仅 temporal 计算方式随模型不同
            if model_arch in ("qwen2.5vl",):
                t_index = _compute_t_index_qwen25vl(llm_grid_t, llm_grid_h, llm_grid_w, second_per_grid_t)
            else:
                # qwen2vl: 顺序 0,1,2,...   qwen3vl: llm_grid_t 恒为 1，t_index 恒为 [0]
                t_index = _compute_t_index_qwen2vl(llm_grid_t, llm_grid_h, llm_grid_w, second_per_grid_t)
            h_index, w_index = _compute_hw_indices(llm_grid_t, llm_grid_h, llm_grid_w)
            llm_pos_ids_list.append(
                torch.stack([t_index, h_index, w_index]) + text_len + st_idx
            )
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        # 末尾文本段
        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
        mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

    mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    return position_ids, mrope_position_deltas


def get_ltor_masks_and_position_ids(
    input_ids,
    image_thw_grids,
    video_thw_grids,
    second_per_grid_ts,
    target,
    pad_token,
    ignore_index=None,
    model_arch="qwen2vl",
    tokenizer=None,
    spatial_merge_size=None,
    attention_mask=None,
    hf_config=None,
):
    """Build masks and position id for left to right model."""
    position_ids, _ = get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_thw_grids,
        video_grid_thw=video_thw_grids,
        second_per_grid_ts=second_per_grid_ts,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        spatial_merge_size=spatial_merge_size,
        model_arch=model_arch,
    )
    # Loss mask.
    loss_mask = torch.ones(target.size(), dtype=torch.float, device=input_ids.device)
    loss_mask[target == pad_token] = 0.0  # mask paddings
    if ignore_index is not None:
        loss_mask[target == ignore_index] = 0.0  # mask prompts

    return loss_mask, position_ids


class DataCollatorForQwen2Vl(object):
    """将 Qwen2VlDataset 产出的单样本 dict 拼装为 batch。

    负责：pixel_values 的 pad/split、position_ids 计算、loss_mask 生成。
    """

    def __init__(
        self,
        hw_factor: int = 1,
        model_arch="qwen2vl",
        tokenizer=None,
        spatial_merge_size=None,
        cp_size=1,
        hf_config=None,
    ):
        super().__init__()
        # hw_factor = context_parallel_size * (seq_parallel ? tp_size : 1) * 4
        # qwen2vl/qwen2.5vl/qwen3vl 的 merge_size 均为 2，pixel_values 天然 4 对齐
        self.hw_factor = hw_factor * 4
        self.model_arch = model_arch
        self.tokenizer = tokenizer
        self.spatial_merge_size = spatial_merge_size
        self.cp_size = cp_size
        self.hf_config = hf_config

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        new_instances = []
        pixel_values = []
        image_grid_thws = []
        for instance in instances:
            if instance["pixel_values"] is not None:
                pixel_values.append(instance["pixel_values"])
                image_grid_thws.append(instance["image_grid_thw"])
            del instance["pixel_values"]
            del instance["image_grid_thw"]
            new_instances.append(instance)

        nopad_image_grid_thw = None
        if len(image_grid_thws) > 0:
            nopad_image_grid_thw = torch.cat(image_grid_thws, dim=0)
        res = default_collate(new_instances)
        if len(pixel_values) > 0:
            pixel_values, image_grid_thws, cp_img_num, images_padded = pad_and_split_pixel_values(
                self.cp_size,
                self.hw_factor,
                pixel_values,
                image_grid_thws,
            )
            if self.model_arch in ["qwen3_vl_moe", 'qwen3_vl']:
                for image_padded in images_padded:
                    assert not image_padded, "not support image padded now"
            elif any(images_padded):
                # qwen2vl / qwen2.5vl 模型 forward 无 images_padded 参数，
                # padding 条目留在 pixel_values/image_grid_thw 中会导致视觉特征数与
                # input_ids 中 image token 数不匹配。直接截掉末尾的 padding 条。
                strip_count = sum(images_padded)
                pixel_values = pixel_values[:-strip_count]
                image_grid_thws = image_grid_thws[:-strip_count]
                images_padded = [False] * len(images_padded)
            res["pixel_values"] = torch.cat(pixel_values, dim=0)
            res["image_grid_thw"] = torch.cat(image_grid_thws, dim=0)
            res["has_image"] = torch.tensor([True], dtype=torch.bool)
            res["images_padded"] = torch.tensor(images_padded, dtype=torch.int64)
            res["cp_img_num"] = torch.tensor(cp_img_num, dtype=torch.int64)
        else:
            res["has_image"] = torch.tensor([False], dtype=torch.bool)

        second_per_grid_ts = None  # 视频暂未支持
        loss_mask, position_ids = get_ltor_masks_and_position_ids(
            res["input_ids"],
            nopad_image_grid_thw,
            None,
            second_per_grid_ts,
            res["labels"],
            self.tokenizer.pad_token_id,
            ignore_index=-100,
            model_arch=self.model_arch,
            tokenizer=self.tokenizer,
            spatial_merge_size=self.spatial_merge_size,
            hf_config=self.hf_config,
        )
        res["loss_mask"] = loss_mask
        if len(pixel_values) > 0:
            res["position_ids"] = position_ids
        else:
            res["position_ids"] = position_ids.clone()
        return res


def get_processor(args):
    processor_path = args.processor_path
    if args.model_arch in ["qwen3_vl_moe", 'qwen3_vl']:
        min_pixels = args.min_pixels_num
        max_pixels = args.max_pixels_num
    else:
        min_pixels = args.min_pixels_num if args.min_pixels_num else MIN_PIXELS
        max_pixels = args.max_pixels_num if args.max_pixels_num else MAX_PIXELS
    init_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "token": None,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "use_fast": True,
    }
    processor = AutoProcessor.from_pretrained(processor_path, **init_kwargs)
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    return processor


def build_train_valid_test_datasets(
    args,
    tokenizer,
    rank=0,
    dp_rank=0,
    dp_size=1,
    use_for_hf=False,
):
    """构建 Qwen2VlDataset 训练/验证/测试数据集。

    从 args 中读取数据路径、domain 配比、processor 配置等，创建 Qwen2VlDataset 实例。
    """
    train_path_likes = args.data_path
    eval_path_likes = args.px_eval_data_path
    domain_probabilities = args.px_domain_probabilities
    retention_rates_per_domains = getattr(args, "px_train_apply_pareto", None)
    domain_names = args.px_train_data_domain_names
    enable_pareto = getattr(args, "px_train_apply_pareto", [])
    pareto_alpha = getattr(args, "px_train_pareto_alpha", [])
    pareto_scale = getattr(args, "px_train_pareto_scale", [])
    pareto_score_scale = getattr(args, "train_pareto_score_scale", [])
    processor = get_processor(args)
    mask_history = args.mask_history
    max_seq_length = args.seq_length

    print_rank_0(
        f'build_train_valid_datasets train_data_consuming_progresses {getattr(args, "train_data_consuming_progresses", None)}'
    )
    train_ds = Qwen2VlDataset(
        args.min_pixels_num,
        args.max_pixels_num,
        use_for_hf,
        mask_history,
        tokenizer,
        max_seq_length,
        train_path_likes,
        domain_probabilities,
        domain_names,
        args.global_batch_size,
        train_data_consuming_progresses=getattr(args, 'train_data_consuming_progresses', None),
        rank=rank,
        dp_rank=dp_rank,
        dp_size=dp_size,
        num_workers=args.num_workers,
        shuffle_buffer_size=args.px_shuffle_buffer_size,
        seed=args.seed,
        train=True,
        retention_rates_per_domains=retention_rates_per_domains,
        enable_pareto=enable_pareto,
        pareto_alphas=pareto_alpha,
        pareto_scales=pareto_scale,
        pareto_score_scales=pareto_score_scale,
        top_domains_to_cut=args.px_top_domains_to_cut,
        processor=processor,
        tar_dir=args.tarfile_path,
        lmdb_port=args.lmdb_port,
        image_token_id=getattr(args, 'image_token_id', None),
        moe_pad_with_random_token=getattr(args, 'moe_pad_with_random_token', False),
    )

    eval_ds = None
    if eval_path_likes is not None:
        eval_ds = Qwen2VlDataset(
            args.min_pixels_num,
            args.max_pixels_num,
            use_for_hf,
            mask_history,
            tokenizer,
            max_seq_length,
            eval_path_likes,
            [1.0],
            args.px_eval_data_domain_names,
            args.global_batch_size,
            train_data_consuming_progresses=None,
            rank=rank,
            dp_rank=dp_rank,
            dp_size=dp_size,
            num_workers=args.num_workers,
            shuffle_buffer_size=args.px_shuffle_buffer_size,
            seed=args.seed,
            train=False,
            retention_rates_per_domains=retention_rates_per_domains,
            enable_pareto=enable_pareto,
            pareto_alphas=pareto_alpha,
            pareto_scales=pareto_scale,
            pareto_score_scales=pareto_score_scale,
            top_domains_to_cut=args.px_top_domains_to_cut,
            processor=processor,
            tar_dir=args.tarfile_path,
            lmdb_port=args.lmdb_port,
            image_token_id=getattr(args, 'image_token_id', None),
            moe_pad_with_random_token=getattr(args, 'moe_pad_with_random_token', False),
        )
        assert args.px_reset_dataloader_at_start_of_eval, "需要--px-reset-dataloader-at-start-of-eval来保保证每次eval的数据是一样的"
    test_ds = None

    return train_ds, eval_ds, test_ds


def build_train_valid_test_data_iter(
    args,
    tokenizer,
    rank=0,
    dp_rank=0,
    dp_size=1,
    use_for_hf=False,
):
    """构建 Qwen2VlDataset 的 DataLoader 迭代器。

    负责 tokenizer 视觉属性注入、hw_factor 计算、DataCollator 创建、DataLoader 组装。
    """

    # ── 注入 Qwen2-VL 视觉 token 属性 ──
    # tokenizer 类型: DefaultTokenizerText → MegatronTokenizerText
    #   tokenizer._tokenizer         = Megatron-LM HuggingFaceTokenizer (库实现)
    #   tokenizer._tokenizer.tokenizer = HF AutoTokenizer (真正的 Qwen2-VL tokenizer)
    hf = tokenizer._tokenizer.tokenizer

    # 将 _tokenizer 替换为 HF AutoTokenizer，dataset 代码通过 _tokenizer(text, ...) 调用
    tokenizer._tokenizer = hf

    # Qwen2-VL 视觉 token 常量及 ID（HF tokenizer 不暴露为命名属性，用 convert_tokens_to_ids 查）
    tokenizer.image_token = '<|image_pad|>'
    tokenizer.image_token_id = hf.convert_tokens_to_ids('<|image_pad|>')
    tokenizer.video_token = '<|video_pad|>'
    tokenizer.video_token_id = hf.convert_tokens_to_ids('<|video_pad|>')
    tokenizer.vision_start_token = '<|vision_start|>'
    tokenizer.vision_start_token_id = hf.convert_tokens_to_ids('<|vision_start|>')
    tokenizer.vision_end_token = '<|vision_end|>'
    tokenizer.vision_end_token_id = hf.convert_tokens_to_ids('<|vision_end|>')

    # DefaultTokenizerText 缺少的常用属性，从 HF tokenizer 补充
    tokenizer.pad_token_id = hf.pad_token_id
    tokenizer.eos_token_id = hf.eos_token_id
    tokenizer.bos_token_id = hf.bos_token_id
    train_ds, eval_ds, test_ds = build_train_valid_test_datasets(
        args,
        tokenizer,
        rank,
        dp_rank,
        dp_size,
        use_for_hf=use_for_hf,
    )

    hw_factor = args.context_parallel_size
    if args.sequence_parallel:
        hw_factor *= args.tensor_model_parallel_size
    # qwen3vl 的 pixel_values 需要不同的对齐策略，跳过额外 padding
    if args.model_arch in ["qwen3_vl_moe", 'qwen3_vl']:
        hw_factor = 1

    hf_config = None
    if args.model_arch in ["qwen3_vl_moe", 'qwen3_vl']:
        from transformers import Qwen3VLConfig
        hf_config = Qwen3VLConfig.from_pretrained(args.processor_path)
        assert hf_config.image_token_id == tokenizer.image_token_id
        assert hf_config.video_token_id == tokenizer.video_token_id
        assert hf_config.vision_start_token_id == tokenizer.vision_start_token_id
        assert hf_config.vision_end_token_id == tokenizer.vision_end_token_id

    collate_func = DataCollatorForQwen2Vl(
        hw_factor=hw_factor,
        model_arch=args.model_arch,
        tokenizer=tokenizer,
        spatial_merge_size=args.spatial_merge_size,
        cp_size=args.context_parallel_size,
        hf_config=hf_config,
    )

    batch_size = args.micro_batch_size
    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_func,
        prefetch_factor=args.px_dataloader_prefetch_factor,
    )

    eval_dataloader = None
    if eval_ds is not None:
        eval_dataloader = torch.utils.data.DataLoader(
            eval_ds,
            batch_size=batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_func,
            prefetch_factor=args.px_dataloader_prefetch_factor,
        )
    test_dataloader = None
    if test_ds is not None:
        test_dataloader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_func,
            prefetch_factor=args.px_dataloader_prefetch_factor,
        )
    if use_for_hf:
        return train_dataloader, eval_dataloader, test_dataloader
    return get_iterator(train_dataloader), get_iterator(eval_dataloader
                                                       ), get_iterator(test_dataloader)

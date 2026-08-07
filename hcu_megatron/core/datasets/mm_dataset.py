import os
import io
import re
import copy
from collections import defaultdict

from PIL import Image
import torch
from torch.utils.data import IterableDataset as TorchIterableDataset

from hcu_megatron.core.datasets.mega_indexed_jsonl_dataset import MegaIndexedJsonlDatasetMM


tar_file_cache = {}  # tar_filepath: (tar_context, access_cnt)
tar_file_cache_size = 25
tar_access_cnt = 0


def get_image_from_tar(tar_filepath, offset, size) -> Image.Image:
    global tar_file_cache
    global tar_file_cache_size
    global tar_access_cnt
    tar_access_cnt += 1
    if tar_filepath in tar_file_cache:
        tar_context = tar_file_cache[tar_filepath][0]
        tar_file_cache[tar_filepath][1] = tar_access_cnt
    else:
        with open(tar_filepath, "rb") as tar_fd:
            tar_context = tar_fd.read()
        if len(tar_file_cache) > tar_file_cache_size:
            # get the min access_cnt file
            min_key = ""
            min_cnt = tar_access_cnt * 10
            for k, v in tar_file_cache.items():
                if v[1] < min_cnt:
                    min_key = k
                    min_cnt = v[1]
            tar_file_cache.pop(min_key)
        tar_file_cache[tar_filepath] = [tar_context, tar_access_cnt]

    image = Image.open(io.BytesIO(tar_context[offset:offset + size]))
    return image


def fetch_image(
    ele: dict[str, str],
    tar_dir,
) -> Image.Image:
    image_path = ele["image_path"]
    # 目前只支持本地路径及从tar包中读取
    if "tar_name" in ele:
        image_obj = get_image_from_tar(
            os.path.join(tar_dir, ele["tar_name"]), ele["offset"], ele["size"]
        )
        image = image_obj.convert("RGB")
    else:
        image_obj = Image.open(image_path)
        image = image_obj.convert("RGB")

    return image


def fetch_images(images: list[dict], tar_dir: str) -> list[Image.Image]:
    return [fetch_image(ele, tar_dir) for ele in images]


def convert_pattern(
    user_input: str, image_pattern: str = '<image>', video_pattern: str = '<video>'
):
    """
        Split user input into format tokenizer accepts.
    """
    pattern = r"({image}|{video})".format(image=image_pattern, video=video_pattern)
    contents = []
    cur = 0
    mm_idx = defaultdict(int)
    for matched in re.finditer(pattern, user_input):
        start, end = matched.span()
        if start > cur:
            contents.append({"type": "text", "text": user_input[cur:start]})

        contents.append(
            {
                "type": matched.string[start:end][1:-1],
                matched.string[start:end][1:-1]: str(mm_idx[matched.string[start:end][1:-1]])
            }
        )

        cur = end
        mm_idx[matched.string[start:end][1:-1]] += 1

    if cur < len(user_input):
        contents.append({"type": "text", "text": user_input[cur:len(user_input)]})

    return contents


def convert_conversations(conversations):
    res = []
    for conversation in conversations:
        new_conversation = copy.deepcopy(conversation)
        new_conversation['content'] = convert_pattern(conversation['content'])
        res.append(new_conversation)

    return res


def remove_bos(text):
    # 这里用了两次process，<bos>会多一个
    bos = '<bos>'
    assert text.startswith(bos)
    return text[len(bos):]


'''
输入的数据格式如下（SFT）：
```json
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
要求：
1. 可以没有图片
2. images 也可以只写 image_path 绝对路径
3. 每一条样本都要是有效样本，图片必须能正常读出来
4. conversations[-1] 默认为 label
'''


class MultiModalDataset(TorchIterableDataset):
    def __init__(
        self,
        tokenizer,
        max_seq_len,
        path_likes,
        domain_probabilities,
        domain_names,
        global_batch_size,
        train_data_consuming_progresses=None,
        rank=0,
        dp_rank=0,
        dp_size=1,
        num_workers=1,
        shuffle_buffer_size=1000,
        seed=0,
        train=False,
        top_domains_to_cut=1,
        processor=None,
        tar_dir="/",
        image_token_id=None,
        moe_pad_with_random_token=False,
    ):
        self.underlying = MegaIndexedJsonlDatasetMM(
            path_likes=path_likes,
            domain_probabilities=domain_probabilities,
            domain_names=domain_names,
            global_batch_size=global_batch_size,
            train_data_consuming_progresses=train_data_consuming_progresses,
            rank=rank,
            dp_rank=dp_rank,
            dp_size=dp_size,
            num_workers=num_workers,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            train=train,
            top_domains_to_cut=top_domains_to_cut,
        )
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_processor = processor.image_processor if processor is not None else None
        self.max_seq_len = max_seq_len
        self.tar_dir = tar_dir
        self.image_token_id = image_token_id
        self.moe_pad_with_random_token = moe_pad_with_random_token

    def gen_label_mask(self, conversations, imgs, tools, image_grid_thw=None,
                       label_role=["assistant"], rm_bos=True):
        pre_len = 0
        mask_indexs = []
        for i in range(len(conversations)):
            if conversations[i]['role'] in ['system']:
                continue
            add_generation_prompt = (conversations[i]['role'] in ['user'])
            text = self.processor.apply_chat_template(
                conversations[:i + 1],
                tools=tools,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            # Use image token count to slice the right subset of image_grid_thw
            if image_grid_thw is not None:
                n_imgs = text.count(self.tokenizer.image_token)
                thw_prefix = image_grid_thw[:n_imgs] if n_imgs > 0 else None
                text = self.padding_vision_token(text, thw_prefix)
            if rm_bos:
                text = remove_bos(text)
            cur_len = len(self.tokenizer._tokenizer(text, padding=False).input_ids)
            if conversations[i]['role'] not in label_role:
                mask_indexs.append([pre_len, cur_len])
            pre_len = cur_len
        if self.mask_history:
            mask_indexs = [[mask_indexs[0][0], mask_indexs[-1][-1]]]
        return mask_indexs

    def __iter__(self):
        raise NotImplementedError

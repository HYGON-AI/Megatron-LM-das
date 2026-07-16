from datetime import datetime
from types import SimpleNamespace
import glob
import itertools
import json
import copy
import os

import torch
from torch.utils.data import IterableDataset as TorchIterableDataset

from hcu_megatron.core.datasets.utils import print_rank_0


class ConsumedByThisRank:
    '''
    train_data_consuming_progresses: {
        rank_0: {
            domain_0: {
                wk_0: (epoch, line),
                wk_1: (epoch, line),
                ...
            },
            ...
        },
        ...
    }
    consumed = train_data_consuming_progresses[rank][domain_id][worker_id]
    '''
    def __init__(self, num_domains, num_workers):
        self.num_domains = num_domains
        self.num_workers = num_workers
        self.domain_cand_off = -1
        self.each_domain = {}

    def __str__(self):
        import pprint
        to_disp = dict(
            num_domains=self.num_domains,
            num_workers=self.num_workers,
            domain_cand_off=self.domain_cand_off,
            each_domain=self.each_domain,
        )
        return pprint.pformat(to_disp)

    def __repr__(self):
        return str(self)

    def merge(self, rhs):
        for domain_id, consumed_in_this_domain in self.each_domain.items():
            if domain_id not in rhs.each_domain:
                continue
            for worker_id, consumed in consumed_in_this_domain.items():
                if worker_id not in rhs.each_domain[domain_id]:
                    continue
                rhs_consumed = rhs.each_domain[domain_id][worker_id]
                if consumed.epoch < rhs_consumed.epoch:
                    consumed.epoch = rhs_consumed.epoch
                    consumed.line = rhs_consumed.line
                if consumed.epoch == rhs_consumed.epoch and consumed.line < rhs_consumed.line:
                    consumed.line = rhs_consumed.line


def get_consumed_by_this_rank(consuming_progresses, rank, num_domains, num_workers):
    empty = ConsumedByThisRank(num_domains, num_workers)
    if consuming_progresses is None:
        # eval only
        return empty
    cbt = consuming_progresses.setdefault(rank, empty)
    assert cbt is not None
    assert cbt.num_domains == num_domains
    return cbt


def get_consumed_in_this_domain(consumed_by_this_rank, domain_id):
    consumed_in_this_domain = consumed_by_this_rank.each_domain.setdefault(domain_id, {})
    return consumed_in_this_domain


def get_consumed_by_this_worker(consumed_in_this_domain, worker_id):
    consumed_by_this_wk = consumed_in_this_domain.setdefault(
        worker_id, SimpleNamespace(epoch=0, line=0)
    )
    return consumed_by_this_wk


def update_consumed(consuming_progresses, rank, data):
    if data is None:
        return
    if not data['train'][0].item():
        return

    consumed_by_this_rank = consuming_progresses[rank]
    consumed_by_this_rank.domain_cand_off = data['domain_cand_off'][-1].item()

    for domain_id in range(consumed_by_this_rank.num_domains):
        for worker_id in range(consumed_by_this_rank.num_workers):
            consumed = get_consumed_by_this_worker(
                get_consumed_in_this_domain(consumed_by_this_rank, domain_id), worker_id
            )
            prev_epoch = consumed.epoch
            mask = data["domain_id"] == domain_id
            if data["domain_epoch"][mask].numel() == 0:
                continue

            max_epoch = data["domain_epoch"][mask].max().item()
            if max_epoch != prev_epoch:
                consumed.epoch = max_epoch
                consumed.line = 0

            max_epoch_mask = (data['domain_id']
                              == domain_id) & (data['worker_id']
                                               == worker_id) & (data['domain_epoch'] == max_epoch)
            n_packed_sum = data['domain_line'][max_epoch_mask].sum().item()
            consumed.line += n_packed_sum


def adjust_domain_id_list_for_dp(gbs, dp_rank, dp_world_size, domain_id_list):
    fake_ring_domain_id_list = domain_id_list + domain_id_list
    offset = gbs // dp_world_size
    start_index = offset * dp_rank
    end_index = start_index + gbs
    return fake_ring_domain_id_list[start_index:end_index]


def generate_global_batch_domain_id(
    gbs, domains_probs, dp_rank, dp_world_size, top_domains_to_cut=1
):
    assert gbs >= dp_world_size and gbs % dp_world_size == 0
    domains_num = len(domains_probs)
    domain_samples = [max(1, int(p * gbs)) for p in domains_probs]
    total_samples = sum(domain_samples)

    # 如果总样本数大于/小于全局批次大小，从样本数最大 cut_domain 个域中减去/加上对应数量的样本额度
    if total_samples != gbs:
        top_domain_indices = sorted(
            range(len(domain_samples)), key=lambda i: domain_samples[i], reverse=True
        )[:top_domains_to_cut]
        differences = abs(total_samples - gbs)
        assert differences > 0
        for index in top_domain_indices:
            cut_nums = differences // top_domains_to_cut + int(
                index < differences % top_domains_to_cut
            )
            assert cut_nums < domain_samples[index], f"somethong wrong"
            # 上面 assert 会触发的情况是 domain多，gbs 小，比如domains_num = 20, gbs = 10, 分都不够分，
            # 那么需要认为调整，或者完全没有必要用此 dataset
            if total_samples < gbs:
                domain_samples[index] += cut_nums
            else:
                domain_samples[index] -= cut_nums

    domain_id_list = []
    for domain_id, num_samples in enumerate(domain_samples):
        domain_id_list.extend([domain_id] * num_samples)
    assert len(domain_id_list) == gbs, f"samples num mismatch"
    return adjust_domain_id_list_for_dp(gbs, dp_rank, dp_world_size, domain_id_list=domain_id_list)


def tokenize_text(tokenizer, text):
    y = tokenizer(text, add_special_tokens=False)
    input_ids = y.input_ids
    attention_mask = y.attention_mask
    return input_ids, attention_mask


class BaseIndexedJsonlDataset(TorchIterableDataset):
    """Indexed JSONL 数据集基类 —— 提供 domain 调度 + 文件读取 + 消费追踪。

    设计目标
    ========
    将"数据从哪来、按什么顺序取、读到哪了"这些通用逻辑统一封装，
    子类只需实现 __iter__() 来定义"取出一条数据后怎么处理"。

    三层消费追踪体系
    ================
    训练可能随时中断，热启动需要精确恢复每个 rank 上的每个 worker 的数据消费进度。
    因此设计了三级映射：

        train_data_consuming_progresses: dict
            rank → ConsumedByThisRank
                    └── each_domain: dict
                            domain_id → dict
                                    worker_id → SimpleNamespace(epoch, line)

    - rank 级别: 记录 domain_cand_off（当前在 global_batch_domain_id 的哪个位置）
    - domain 级别: 一个 rank 上有多个 domain 的数据在混合消费
    - worker 级别: DataLoader 多 worker 时每个 worker 的进度独立追踪

    domain 调度机制
    ==============
    问题是：每个 global_batch 里各个 domain 的数据各占多少条？

    解法：generate_global_batch_domain_id() 根据 domain_probabilities 生成一个
    长度为 global_batch_size 的列表，例如 gbs=16, probs=[0.5, 0.3, 0.2]：
        global_batch_domain_id = [0,0,0,0,0,0,0,0, 1,1,1,1,1, 2,2,2]

    每次 __iter__ yield 一条数据，domain_cand_off 就 +1，指向列表的下一个元素，
    从而决定了下一个样本从哪个 domain 取。一个周期走完后回到 0，循环往复。

    create_dataset 的两层含义
    ========================
    "dataset" 这个词在本文件中指两种东西：
      1. HuggingFace datasets 的 IterableDataset（通过 load_dataset('json', streaming=True) 创建）
         → 这是"数据文件读取器"，负责遍历 JSONL 的每一行
      2. torch.utils.data.IterableDataset 的子类（BaseIndexedJsonlDataset）
         → 这是"训练数据供给器"，负责 domain 调度 + 数据消费

    create_dataset() 创建的是第 1 种 —— 一个指向磁盘 JSONL 文件的可迭代读取器。
    """

    def __init__(
        self,
        path_likes,                      # 每个 domain 的数据路径列表，已 index 化的 JSONL 文件
        domain_probabilities,            # 各 domain 的采样概率，如 [0.5, 0.3, 0.2]
        domain_names,                    # 各 domain 的名称，如 ["math", "code", "chat"]
        global_batch_size,               # 全局 batch size，用于生成 domain 调度表
        train_data_consuming_progresses=None,  # 断点续训的消费进度，None 表示从头开始
        rank=0,                          # 全局 rank
        dp_rank=0,                       # 数据并行 rank
        dp_size=1,                       # 数据并行 world size
        num_workers=1,                   # DataLoader worker 数
        shuffle_buffer_size=1000,        # HuggingFace shuffle buffer 大小（当前未直接使用）
        seed=0,                          # 随机种子
        train=False,                     # True=训练集, False=验证/测试集
        top_domains_to_cut=1,            # domain 调度微调时，修改几个最大的 domain
    ):
        assert isinstance(path_likes, list)
        if domain_probabilities is not None:
            assert len(domain_probabilities) == len(path_likes)
        self.path_likes = path_likes
        self.domain_probabilities = domain_probabilities
        self.domain_names = domain_names
        self.rank = rank
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.num_workers = num_workers
        self.global_batch_size = global_batch_size
        assert top_domains_to_cut <= len(domain_names)
        self.top_domains_to_cut = top_domains_to_cut

        self.train = train
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train_data_consuming_progresses = train_data_consuming_progresses
        # 恢复当前 rank 的消费进度（断点续训的核心）
        self.consumed_by_this_rank = get_consumed_by_this_rank(
            train_data_consuming_progresses, self.rank, len(path_likes), num_workers
        )

        self.print_domain_id_map()

        # ds_list: 每个 domain 对应一个 HF datasets 迭代器，惰性初始化
        self.ds_list = None

        # 生成 domain 调度表并切片到当前 dp_rank
        # 例如 gbs=16, probs=[0.5,0.5], dp_rank=0, dp_size=2
        #   → 全部: [0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1]
        #   → rank 0 拿到前 8 个，rank 1 拿到后 8 个
        self.global_batch_domain_id = generate_global_batch_domain_id(
            gbs=global_batch_size,
            domains_probs=domain_probabilities,
            dp_rank=dp_rank,
            dp_world_size=dp_size,
            top_domains_to_cut=top_domains_to_cut,
        )
        if self.dp_rank in [0, 1]:
            print(
                f"Dataset init dp rank {dp_rank} train {train} "
                f"global_batch_domain_id {self.global_batch_domain_id}"
            )

        # domain_cand_off: 指向 global_batch_domain_id 的当前位置
        # 热启动时从上次保存的位置 +1 开始
        self.domain_cand_off = (
            self.consumed_by_this_rank.domain_cand_off + 1
        ) % self.global_batch_size

        self.in_iter = False
        # 文件句柄缓存：避免每次 read_and_parse_obj_from_jsonl 都 open/close
        self.eval_file_cache = {}
        self.train_file_cache = {}

    # =========================================================================
    # 初始化辅助
    # =========================================================================

    def print_domain_id_map(self):
        """打印 domain ID 到数据路径的映射表，方便调试。"""
        domain_id_map = []
        for domain_id, path_like in enumerate(self.path_likes):
            d = {
                'domain_id': domain_id,
                'domain_name': self.domain_names[domain_id],
                'domain_path_like': path_like,
            }
            if self.domain_probabilities:
                d['domain_probabilities'] = self.domain_probabilities[domain_id]
            domain_id_map.append(d)
        domain_id_map_str = type(self).__name__ + ' id / domain mapping ' + json.dumps(
            domain_id_map, indent=4
        )
        print_rank_0(domain_id_map_str)

    # =========================================================================
    # 数据文件读取器
    # =========================================================================

    def create_dataset(self, domain_id, path_like, new_epoch=False):
        """创建一个生成器，逐行 yield JSONL 数据。

        不再使用 HF load_dataset('json', ...)，因为其在 DataLoader 子进程中频繁返回空迭代器。
        改用 Python 原生 json.loads 逐行读取，稳定可靠。

        参数
        ----
        domain_id:  domain 编号
        path_like:  JSONL 文件路径（.jsonl 文件或目录）
        new_epoch:  是否是新 epoch

        返回
        ----
        生成器，每个元素是 {"conversations": [...], "images": [...], "domain_id": tensor}
        """
        if new_epoch:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            consumed_in_this_domain = get_consumed_in_this_domain(
                self.consumed_by_this_rank, domain_id
            )
            consumed_by_this_wk = get_consumed_by_this_worker(
                consumed_in_this_domain, worker_id
            )
            consumed_by_this_wk.epoch += 1
            consumed_by_this_wk.line = 0

        # 支持目录或单个 .jsonl 文件
        if os.path.isdir(path_like):
            files = sorted(glob.glob(os.path.join(path_like, '*.jsonl')))
        else:
            files = [path_like]

        def jsonl_generator():
            for fpath in files:
                with open(fpath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        obj['domain_id'] = torch.tensor(domain_id, dtype=torch.int64)
                        yield obj

        return jsonl_generator()

    def make_dasets_for_each_domain(self):
        """为所有 domain 创建 HF datasets 迭代器。

        创建后立即 peek 第一条数据，验证数据集非空。
        如果首次创建就为空，直接报错（数据路径或格式问题）。
        """
        ds_list = []
        for domain_id, path_like in enumerate(self.path_likes):
            ds = self.create_dataset(domain_id, path_like)
            # print(f"[rank {self.rank}] ds = {ds}, path={path_like}")
            ds_iter = iter(ds)
            first = next(ds_iter)
            ds_iter = itertools.chain([first], ds_iter)
            ds_list.append(ds_iter)
        return ds_list

    # =========================================================================
    # JSONL 文件读取
    # =========================================================================

    def read_and_parse_obj_from_jsonl(self, fname, offset, length):
        """从 JSONL 文件中读取一条记录。

        JSONL 格式：每行一个 JSON 对象，通过 offset/length 定位。
        文件句柄会被缓存（self.train_file_cache / self.eval_file_cache），
        避免每次读取都 open/close 文件。

        参数
        ----
        fname:  JSONL 文件名
        offset: 文件偏移量（字节）
        length: 记录长度（字节）

        返回
        ----
        dict: JSON 解析后的 Python 对象
        """
        if not self.train:
            file_cache = self.eval_file_cache
        else:
            file_cache = self.train_file_cache

        if fname in file_cache:
            inf = file_cache[fname]
        else:
            inf = open(fname, 'rb')
            file_cache[fname] = inf
        inf.seek(offset)
        line = inf.read(length)
        obj = json.loads(line)
        return obj

    def log_skip(self, domain_id, domain_name, to_skip):
        """打印 skip 进度的日志（用于断点续训时跳过已消费数据）。"""
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'[{time_str}] skip consumed_by_this_rank lines dname {domain_name}'
              + f' rank {torch.distributed.get_rank()}'
              + f' dp_rank {self.dp_rank}'
              + f' to_skip {to_skip}')

    def __iter__(self):
        raise NotImplementedError('Subclasses must implement __iter__')


class MegaIndexedJsonlDataset(BaseIndexedJsonlDataset):
    """文本预训练数据集 —— 将多个 document tokenize 后 pack 成固定长度序列。

    数据流
    ======
    JSONL 文件中的每一行是一个 document:
        {"content": "这是文档正文...", "docid": "doc_001", ...}
        {"content": "另一个文档...", "docid": "doc_002", ...}

    __iter__ 的处理流程:

    ① 按 domain 调度表决定下一个样本从哪个 domain 取
    ② 从该 domain 的 HF datasets 迭代器中取出一行 JSON
    ③ 跳过 deleted=True 的垃圾数据
    ④ 取 example['content']，tokenize → input_ids
    ⑤ 追加到 bufs[domain_id].input_ids 累积缓冲区
    ⑥ 当缓冲区长度 >= max_seq_len + 1 时，切出一条完整序列
    ⑦ 剩余的 token 留在缓冲区中，供下一条序列使用

    这样可以将多个短 document 拼接（pack）成一条训练序列，
    避免短文档造成的大量 padding 浪费。

    与 MegaIndexedJsonlDatasetMM 的对比
    ==================================
    V3:  text → tokenize → pack → yield tensor
         (适合纯文本预训练，多个 doc 拼成固定长度)

    MM:  text → yield raw JSON
         (适合多模态，上层 QwenVLDataset 处理图片后再 tokenize)
    """

    def __init__(
        self,
        tokenizer,                       # 文本 tokenizer
        max_seq_len,                     # 最大序列长度（序列 pack 的目标长度）
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
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        super().__init__(
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

    def __iter__(self):
        """Pack 模式的迭代器。

        核心机制
        --------
        每个 domain 维护一个 input_ids 缓冲区（bufs[domain_id]），
        不断从数据源取 document → tokenize → 追加到缓冲区。
        缓冲区达到 max_seq_len + 1 时切出一条序列 yield，剩余部分留在缓冲区。

        +1 的原因：自回归训练需要 input 和 label 有 1 个 token 的错位。
        这里 yield 的 input_ids 和 labels 长度相同，由上层做 shift。

        示意图
        ------
        doc_A: [t1 t2 t3 t4 t5]
        doc_B: [t6 t7 t8 t9 t10 t11 t12]
        max_seq_len = 6

        缓冲区累积:
          [t1 t2 t3 t4 t5] + [t6 t7 t8 t9 t10 t11 t12]
          = [t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t11 t12]

        切出 seq_len=7 (max_seq_len+1=7):
          yield [t1 t2 t3 t4 t5 t6 t7]

        剩余:
          [t8 t9 t10 t11 t12]  → 继续累积到 >= 7 再切

        yield 字段
        ----------
        input_ids:       [seq_len]   — 训练输入 token
        labels:          [seq_len]   — 训练目标（此处与 input_ids 相同，上层做 shift）
        train:           bool        — 是否训练集
        domain_id:       int         — 数据域 ID
        worker_id:       int         — DataLoader worker ID
        domain_epoch:    int         — 当前 domain 的 epoch 数
        domain_line:     int         — 本条序列 pack 了几个 document
        domain_cand_off: int         — 当前在 domain 调度表中的位置
        """
        assert not self.in_iter
        self.in_iter = True

        # 每个 domain 一个缓冲区：
        #   input_ids: 累积的 token 列表
        #   n_packed:  当前序列 pack 了多少个 document
        bufs = [
            SimpleNamespace(input_ids=[], n_packed=0)
            for domain_id, _ in enumerate(self.domain_names)
        ]
        if self.ds_list is None:
            self.ds_list = self.make_dasets_for_each_domain()

        while True:
            # ── 确定当前该从哪个 domain 取数据 ──
            domain_id = self.global_batch_domain_id[self.domain_cand_off]
            ds = self.ds_list[domain_id]

            # ── 累积 token 直到缓冲区 >= max_seq_len + 1 ──
            while len(bufs[domain_id].input_ids) < self.max_seq_len + 1:
                try:
                    idx = next(ds)
                except StopIteration:
                    # domain 数据耗尽 → 重开新 epoch
                    self.ds_list[domain_id] = iter(
                        self.create_dataset(domain_id, self.path_likes[domain_id], new_epoch=True)
                    )
                    ds = self.ds_list[domain_id]
                    idx = next(ds)

                fname = idx['data_file_name']
                offset = idx['offset']
                length = idx['length']
                assert idx['domain_id'].item() == domain_id
                worker_id = idx['worker_id']

                # 从 JSONL 文件中读取一条 document
                example = self.read_and_parse_obj_from_jsonl(fname, offset, length)
                bufs[domain_id].n_packed += 1
                if example.get('deleted', False):
                    # 黑名单：被标记删除的垃圾数据
                    continue
                text = example['content']
                doc_id = example['docid']
                _input_ids, _ = tokenize_text(self.tokenizer, text)
                bufs[domain_id].input_ids += _input_ids

            # ── 缓冲区够了 → 切出一条训练序列 ──
            input_ids = bufs[domain_id].input_ids[:self.max_seq_len + 1]
            labels = copy.deepcopy(input_ids)
            domain_epoch = get_consumed_by_this_worker(
                get_consumed_in_this_domain(self.consumed_by_this_rank, domain_id), worker_id
            ).epoch
            ret_d = {
                'input_ids': torch.tensor(input_ids, dtype=torch.int64),
                'labels': torch.tensor(labels, dtype=torch.int64),
                'train': self.train,
                'domain_id': torch.tensor(domain_id, dtype=torch.int64),
                'worker_id': torch.tensor(worker_id, dtype=torch.int64),
                'domain_epoch': torch.tensor(domain_epoch, dtype=torch.int64),
                'domain_line': torch.tensor(bufs[domain_id].n_packed, dtype=torch.int64),
                'domain_cand_off': self.domain_cand_off,
            }
            # 切掉已用部分，保留剩余 token 供下一条序列使用
            bufs[domain_id].input_ids = bufs[domain_id].input_ids[self.max_seq_len:]
            bufs[domain_id].n_packed = 0
            # 推进 domain 调度表
            self.domain_cand_off = (self.domain_cand_off + 1) % len(self.global_batch_domain_id)
            yield ret_d

        assert False, 'never reachable'


# build_train_valid_test_datasets 已删除：此为文本预训练路径（MegaIndexedJsonlDataset），
# 当前无调用入口。VLM 训练请使用 vlm_dataset.py 中的同名函数。


class MegaIndexedJsonlDatasetMM(BaseIndexedJsonlDataset):
    """多模态 Indexed JSONL 数据集（底层数据加载）。

    与 MegaIndexedJsonlDataset 的区别：
      - V3:  对文本做 tokenize + pack 成固定长度序列（文本预训练）
      - MM:  直接 yield 原始 json_data，不做 tokenize/pack（图文多模态）

    tokenization 和图片处理由上层 MultiModalDataset / QwenVLDataset 完成。
    """

    def __init__(
        self,
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
    ):
        super().__init__(
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

    def __iter__(self):
        """逐条 yield 原始 JSON 数据，不做 tokenize/pack。

        兼容两种 HF datasets 返回格式：
          1. 索引模式（通过目录加载 .jsonl）：idx = {data_file_name, offset, length, domain_id, worker_id}
             需要手动 read_and_parse_obj_from_jsonl(fname, offset, length) 读出原始 JSON
          2. 直接模式（通过单个文件加载）：idx 直接包含原始 JSON 字段 + domain_id
             无需额外读取，直接剔除元数据字段即可

        yield 字段:
            json_data:      原始 JSON dict（conversations + images）
            train:          是否为训练集
            domain_id:      数据域 ID
            worker_id:      DataLoader worker ID
            domain_epoch:   当前 domain epoch
            domain_line:    该样本计数（固定为 1）
            domain_cand_off: domain 候选偏移量
        """
        assert not self.in_iter
        self.in_iter = True
        if self.ds_list is None:
            self.ds_list = self.make_dasets_for_each_domain()

        while True:
            domain_id = self.global_batch_domain_id[self.domain_cand_off]
            ds = self.ds_list[domain_id]

            try:
                idx = next(ds)
            except StopIteration:
                # domain 耗尽 → 重建新 epoch，重试一次
                self.ds_list[domain_id] = iter(
                    self.create_dataset(domain_id, self.path_likes[domain_id], new_epoch=True)
                )
                ds = self.ds_list[domain_id]
                idx = next(ds)

            # 判断返回格式：有 data_file_name → 索引模式，否则 → 直接模式
            if 'data_file_name' in idx:
                # 索引模式：从索引字段定位到 JSONL 文件中的原始行
                fname = idx['data_file_name']
                offset = idx['offset']
                length = idx['length']
                assert idx['domain_id'].item() == domain_id
                worker_id = idx['worker_id']
                example = self.read_and_parse_obj_from_jsonl(fname, offset, length)
            else:
                # 直接模式：idx 本身就是 JSON 内容（含 conversations, images 等业务字段）
                worker_info = torch.utils.data.get_worker_info()
                worker_id = worker_info.id if worker_info is not None else 0
                # 过滤掉 HF datasets 添加的元数据字段，只保留业务数据
                meta_keys = {'domain_id', '__index_level_0__'}
                example = {k: v for k, v in idx.items() if k not in meta_keys}

            domain_epoch = get_consumed_by_this_worker(
                get_consumed_in_this_domain(self.consumed_by_this_rank, domain_id), worker_id
            ).epoch
            ret_d = {
                'json_data': example,
                'train': self.train,
                'domain_id': torch.tensor(domain_id, dtype=torch.int64),
                'worker_id': torch.tensor(worker_id, dtype=torch.int64),
                'domain_epoch': torch.tensor(domain_epoch, dtype=torch.int64),
                'domain_line': 1,
                'domain_cand_off': self.domain_cand_off,
            }
            self.domain_cand_off = (self.domain_cand_off + 1) % len(self.global_batch_domain_id)
            yield ret_d

        assert False, 'never reachable'

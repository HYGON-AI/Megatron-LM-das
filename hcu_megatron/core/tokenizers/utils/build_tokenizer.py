import math
from functools import wraps

from transformers import AutoTokenizer, Qwen2Tokenizer, AutoProcessor


def _vocab_size_with_padding(orig_vocab_size, args, logging_enabled=True):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by * args.tensor_model_parallel_size
    after = int(math.ceil(after / multiple) * multiple)
    if args.rank == 0 and logging_enabled:
        print(
            f' > padded vocab (size: {orig_vocab_size}) with '
            f'{after - orig_vocab_size} dummy tokens '
            f'(new size: {after})'
        )
    return after


def build_tokenizer_wrapper(build_tokenizer_func):
    @wraps(build_tokenizer_func)
    def wrapper(args, **kwargs):
        extra_tokenizer_types = {
            "Llama3Tokenizer",
            "QwenTokenizer",
            "DeepSeekV2Tokenizer",
            "Qwen2VLTokenizer",
        }
        if args.tokenizer_type in extra_tokenizer_types:
            if args.rank == 0:
                print('> building {} tokenizer ...'.format(args.tokenizer_type), flush=True)

            if args.tokenizer_type == 'Llama3Tokenizer':
                assert args.tokenizer_model is not None
                tokenizer = _Llama3Tokenizer(args.tokenizer_model)
            elif args.tokenizer_type == 'QwenTokenizer':
                tokenizer = _Qwen2Tokenizer(args.vocab_file, args.merge_file)
            elif args.tokenizer_type == "DeepSeekV2Tokenizer":
                tokenizer = _DeepSeekV2Tokenizer(args.tokenizer_model, args.extra_vocab_size)
                args.padded_vocab_size = tokenizer.vocab_size
            elif args.tokenizer_type == "Qwen2VLTokenizer":
                tokenizer = _Qwen2VLTokenizer(args.tokenizer_model, args.extra_vocab_size)
                args.padded_vocab_size = tokenizer.vocab_size

            # Add vocab size (if not already set from a checkpoint).
            if getattr(args, "padded_vocab_size", None) is None:
                args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size, args)

            return tokenizer

        return build_tokenizer_func(args, **kwargs)

    return wrapper


class _Llama3Tokenizer():
    """tiktokenTokenizer-Megatron llama3 改写"""
    # https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py

    def __init__(self, model_file):
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe
        tokenizer_path=model_file
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
            ] + [f"<|reserved_special_token_{i}|>" for i in range (5, 256 - 5)]
        mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
        self.tokenizer = tiktoken.Encoding(tokenizer_path,
            pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=mergeable_ranks,
            special_tokens={token: len (mergeable_ranks) + i for i, token in enumerate (special_tokens)},
            )

        self.eod_id = self.tokenizer.encode("<|end_of_text|>", allowed_special="all")[0]
    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab

    @property
    def vocab(self):
        return self.tokenizer.encode

    @property
    def inv_vocab(self):
        return self.tokenizer.encode

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.encode(token_ids)

    @property
    def eod(self):
        return self.eod_id


class _Qwen2Tokenizer():
    def __init__(self, vocab_file, merge_file, extra_vocab_size=0):
        self.tokenizer = Qwen2Tokenizer(vocab_file, merge_file)
        self.extra_vocab_size = extra_vocab_size
        self.tokenizer.add_special_tokens(special_tokens_dict=dict(pad_token="<|extra_0|>"))

    @property
    def vocab_size(self):
        return len(self.tokenizer.encoder) + self.extra_vocab_size

    @property
    def vocab(self):
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.tokenizer.eos_token_id

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id


class _DeepSeekV2Tokenizer():
    def __init__(self, tokenizer_path, extra_vocab_size):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            padding_side="right",
            trust_remote_code=True
        )
        self.extra_vocab_size = extra_vocab_size

        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ message['content'] + '\n\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
            try:
                test_conversation = [
                    {'role': 'user', 'content': 'hello world'}
                ]
                self.apply_chat_template(test_conversation)
            except Exception:
                # the default chat_template is invalid, assume user will not do SFT
                self.tokenizer.chat_template = None

    def __call__(self, text, return_tensors=None,
                 padding=None, max_length=None, truncation=None, add_special_tokens=None):

        return self.tokenizer(text, return_tensors=return_tensors, padding=padding,
                max_length=max_length, truncation=truncation, add_special_tokens=add_special_tokens)

    def apply_chat_template(self, conversations, tokenize:bool=True, **kwargs):
        return self.tokenizer.apply_chat_template(conversations, tokenize=tokenize, **kwargs)

    @property
    def vocab_size(self):
        return len(self.tokenizer) + self.extra_vocab_size - 2

    @property
    def vocab(self):
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.tokenizer.eos_token_id

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id


class _Qwen2VLTokenizer():
    def __init__(self, tokenizer_path, extra_vocab_size):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            padding_side="right",
            trust_remote_code=True,
          use_fast=False
        )
        self.extra_vocab_size = extra_vocab_size
        self.special_tokens_map = {k: v for k, v in
                                    zip(self.tokenizer.all_special_tokens, self.tokenizer.all_special_ids)}
        self.image_token = '<|image_pad|>'
        self.video_token = '<|video_pad|>'
        self.vision_start_token = '<|vision_start|>'
        self.vision_end_token = '<|vision_end|>'

        proc = AutoProcessor.from_pretrained(
            tokenizer_path,
            use_fast=False,
            trust_remote_code=True
        )
        # NOTE: In Qwen2-VL, template in chat_template.json is same within tokenizer_config.json and both can be used.
        # However, in Qwen 2.5-VL, the two templates are different and only the one in chat_template.json is OK.
        self.chat_template = proc.chat_template

    def __call__(self, text, return_tensors=None,
                    padding=None, max_length=None, truncation=None, add_special_tokens=None):
        return self.tokenizer(text, return_tensors=return_tensors, padding=padding,
                                max_length=max_length, truncation=truncation,
                                add_special_tokens=add_special_tokens)

    def apply_chat_template(self, conversations, tokenize: bool = True, **kwargs):
        return self.tokenizer.apply_chat_template(conversations, tokenize=tokenize,
                                                    chat_template=self.chat_template, **kwargs)

    @property
    def vocab_size(self):
        return len(self.tokenizer.encoder) + self.extra_vocab_size

    @property
    def vocab(self):
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.tokenizer.eos_token_id

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def image_token_id(self):
        return self.special_tokens_map[self.image_token]

    @property
    def video_token_id(self):
        return self.special_tokens_map[self.video_token]

    @property
    def vision_start_token_id(self):
        return self.special_tokens_map[self.vision_start_token]

    @property
    def vision_end_token_id(self):
        return self.special_tokens_map[self.vision_end_token]

    def encode(self, x):
        return self.tokenizer.encode(x)

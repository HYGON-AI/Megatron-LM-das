from functools import wraps

from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding

from dcu_megatron.training.tokenizer.tokenizer import (
    _Llama3Tokenizer,
    _Qwen2Tokenizer,
    _DeepSeekV2Tokenizer,
    _Qwen2VLTokenizer,
)

def build_tokenizer_wrapper(build_tokenizer_func):
    @wraps(build_tokenizer_func)
    def wrapper(args):
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

        return build_tokenizer_func(args)

    return wrapper

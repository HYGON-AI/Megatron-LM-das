python tools/preprocess_data.py \
    --input /path/to/oscar-1GB_head.jsonl \
    --output-prefix /path/to/oscar-1GB_head-qwen \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /path/to/hf-qwen \
    --append-eod \
    --workers 8

python tools/preprocess_data.py \
    --input /public/home/wangxj/Downloads/datasets/oscar-1GB-head/oscar-1GB_head.jsonl \
    --output-prefix /public/home/wangxj/Downloads/datasets/oscar-1GB-head/oscar-1GB_head-qwen \
    --vocab-file /public/home/wangxj/Downloads/model_weights/qwen1.5_14b/vocab.json \
    --tokenizer-type QwenTokenizer \
    --merge-file /public/home/wangxj/Downloads/model_weights/qwen1.5_14b/merges.txt \
    --append-eod \
    --workers 8

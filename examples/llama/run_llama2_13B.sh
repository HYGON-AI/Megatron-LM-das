for para in $*
do
    if [[ $para == --profiling* ]];then
        profiling=${para#*=}
    fi
done

# Those variables need to modify
GPUS="8"                 # how many gpus to use
# DTK_ENV="/opt/dtk/env.sh"
DTK_ENV="/public/home/wangxj/Downloads/blas/dtk-25.04.1-rc1/env.sh"              # where env.sh of dtk
# NCCL_ENV="/workspace/dcu_megatron/requirements/nccl_wz/env.sh"             # where env.sh of nccl (requirements/nccl_wz/env.sh or requirements/nccl_zz/env.sh)
NCCL_ENV="/public/home/wangxj/Projects/dcu_megatron/requirements/nccl_wz/env.sh"
HOST="localhost"                 # hostname
PORT="11451"                 # port id
# DATA_PATH="/data/datasets/oscar-1GB-head/oscar-1GB_head-llama2_text_document"            # path to oscar-1GB_head-llama2_text_document
DATA_PATH="/public/home/wangxj/Downloads/datasets/oscar-1GB-head/oscar-1GB_head-llama2_text_document"
# TOKENIZER_MODEL_PATH="/data/model_weights/llama2_7b_hf/tokenizer.model" # path to tokenizer.model
TOKENIZER_MODEL_PATH="/public/home/wangxj/Downloads/model_weights/llama2_7b_hf/tokenizer.model"
CHECKPOINT_PATH="./ckpt"      # path to ckpt

# Runs Llama2 7B model
mpirun -np ${GPUS}  --hostfile hostfile \
                    --allow-run-as-root \
                    --bind-to none \
                    --mca plm_rsh_no_tree_spawn 1 \
                    bash -c "
                    source ${DTK_ENV} && \
                    source ${NCCL_ENV} && \
                    ./train_llama2_13b_1nodes.sh \
                    ${HOST} \
                    ${PORT} \
                    --data_path=$DATA_PATH \
                    --tokenizer_path=$TOKENIZER_MODEL_PATH \
                    --checkpoint_path=$CHECKPOINT_PATH \
                    --profiling=$profiling" > ./log/log-$((${GPUS} / 8))nodes-`date +%F-%H%M`.log 2>&1

wait
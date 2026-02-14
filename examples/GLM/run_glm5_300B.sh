for para in $*
do
    if [[ $para == --profiling* ]];then
        profiling=${para#*=}
    fi
done

CURRENT_DIR=$( cd "$( dirname "$0" )" && pwd )
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))

# Those variables need to modify
# DTK_ENV="/public/home/evt_fugx1/requirements/dtk-25.04.1/env.sh"
DTK_ENV="/opt/dtk-25.04.2/env.sh"                                                                    # where env.sh of dtk
DATA_PATH="/public/home/evt_fugx1/datasets/GLM/moe_datasets/alpaca_text_document"                 # path to oscar-1GB_head-llama2_text_document
TOKENIZER_MODEL_PATH="/public/home/evt_fugx1/datasets/llama/llama2_7b_hf/tokenizer.model"                            # path to tokenizer.model
CHECKPOINT_PATH="./ckpt"                                                       # path to ckpt
NCCL_ENV=${MEGATRON_PATH}/requirements/env.sh                            # Please adjust the variables based on the actual NET being used
LAUNCH_WITH_BINDING=${MEGATRON_PATH}/requirements/launch_with_binding.sh # Please adjust the variables based on the actual NET being used

# Those variables no need to modify
HOSTFILE="hostfile_$(basename "$0" | sed -E 's/^run_(.+)\.sh$/\1/')"
GPUS=$(($(cat ${HOSTFILE}|sort|uniq |wc -l)*8))
HOST="$(cat ${HOSTFILE} |sed -n "1p"|awk -F ' ' '{print $1}')"
PORT="25900"

# Runs GLM5 300B model
source ${DTK_ENV}
source ${NCCL_ENV}
mpirun -np ${GPUS}  --hostfile ${HOSTFILE} \
                    --allow-run-as-root \
                    --bind-to none \
                    --mca plm_rsh_no_tree_spawn 1 \
                    bash -c "
                    source /public/home/evt_fugx1/anaconda3/etc/profile.d/conda.sh && \
	                conda activate megatron_fugx1_0.13.0 && \
                    source ${DTK_ENV} && \
                    source ${NCCL_ENV} && \
                    ./train_glm5_300b_$((${GPUS} / 8))nodes.sh \
                    ${HOST} \
                    ${PORT} \
                    --data_path=$DATA_PATH \
                    --tokenizer_path=$TOKENIZER_MODEL_PATH \
                    --checkpoint_path=$CHECKPOINT_PATH \
                    --launch_with_binding=${LAUNCH_WITH_BINDING} \
                    --profiling=$profiling" # > log-$((${GPUS} / 8))nodes-`date +%F-%H%M`.log 2>&1

wait

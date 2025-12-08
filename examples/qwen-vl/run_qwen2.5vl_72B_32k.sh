for para in $*
do
    if [[ $para == --profiling* ]];then
        profiling=${para#*=}
    fi
done

CURRENT_DIR=$( cd "$( dirname "$0" )" && pwd )
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))

# Those variables need to modify
DTK_ENV=""                                                               # where env.sh of dtk
DATA_PATH=""                                                             # path to wds data
TOKENIZER_MODEL_PATH=""                                                  # path to tokenizer.model
CHECKPOINT_PATH=""                                                       # path to ckpt
NCCL_ENV=${MEGATRON_PATH}/requirements/env.sh                            # Please adjust the variables based on the actual NET being used
LAUNCH_WITH_BINDING=${MEGATRON_PATH}/requirements/launch_with_binding.sh # Please adjust the variables based on the actual NET being used

# Those variables no need to modify
HOSTFILE="hostfile_$(basename "$0" | sed -E 's/^run_(.+)\.sh$/\1/')"
GPUS=$(($(cat ${HOSTFILE}|sort|uniq |wc -l)*8))
HOST="$(cat ${HOSTFILE} |sed -n "1p"|awk -F ' ' '{print $1}')"
PORT="25524"

# Runs Qwen 32B model
source ${DTK_ENV}
source ${NCCL_ENV}
mpirun -np ${GPUS}  --hostfile ${HOSTFILE} \
                    --allow-run-as-root \
                    --bind-to none \
                    --mca plm_rsh_no_tree_spawn 1 \
                    bash -c "
                    source ${DTK_ENV} && \
                    source ${NCCL_ENV} && \
		                source /public/home/user_name/miniconda3/etc/profile.d/conda.sh && \
		                conda activate env_name && \
                    ./train_qwen2.5vl_72B_32k.sh \
                    ${HOST} \
                    ${PORT} \
                    --data_path=$DATA_PATH \
                    --tokenizer_path=$TOKENIZER_MODEL_PATH \
                    --checkpoint_path=$CHECKPOINT_PATH \
                    --launch_with_binding=${LAUNCH_WITH_BINDING} \
                    --profiling=$profiling" | tee logs/log-$((${GPUS} / 8))nodes-qwen25-vl-72b-32k-`date +%F-%H%M`.log

wait


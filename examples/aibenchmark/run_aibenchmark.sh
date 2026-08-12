for para in $*
do
    if [[ $para == --profiling* ]];then
        profiling=${para#*=}
    fi
done

# Those variables no need to modify
hostfile_input=${1}
node_num=${2}

CURRENT_DIR=$( cd "$( dirname "$0" )" && pwd )
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))

# Those variables need to modify
DTK_ENV=""                                                               # where env.sh of dtk
DATA_PATH=""                                                             # path to redpajama_text_document
TOKENIZER_MODEL_PATH=""                                                  # path to tokenizer.model
CHECKPOINT_PATH=""                                                       # path to ckpt
NCCL_ENV=${MEGATRON_PATH}/requirements/env.sh                            # Please adjust the variables based on the actual NET being used
LAUNCH_WITH_BINDING=${MEGATRON_PATH}/requirements/launch_with_binding.sh # Please adjust the variables based on the actual NET being used

HOSTFILE="${hostfile_input}_slots"
rm -f ${HOSTFILE} 
cat ${hostfile_input} | sed -n "1,${node_num}p"|sed 's/$/ slots=8/' > ${HOSTFILE}

GPUS=$(($(cat ${HOSTFILE}|sort|uniq |wc -l)*8))
HOST="$(cat ${HOSTFILE} |sed -n "1p"|awk -F ' ' '{print $1}')"
PORT="25905"

# Runs aibenchmark model
source ${NCCL_ENV}
mpirun -np ${GPUS}  --hostfile ${HOSTFILE} \
                    --allow-run-as-root \
                    --bind-to none \
                    --mca plm_rsh_no_tree_spawn 1 \
                    bash -c "
                    source ${DTK_ENV} && \
                    source ${NCCL_ENV} && \
                    ./train_aibenchmark_$((${GPUS} / 8))nodes.sh \
                    ${HOST} \
                    ${PORT} \
                    --data_path=$DATA_PATH \
                    --tokenizer_path=$TOKENIZER_MODEL_PATH \
                    --checkpoint_path=$CHECKPOINT_PATH \
                    --launch_with_binding=${LAUNCH_WITH_BINDING} \
                    --profiling=$profiling" 2>&1 | tee log-$((${GPUS} / 8))nodes-`date +%F-%H%M`.log

wait

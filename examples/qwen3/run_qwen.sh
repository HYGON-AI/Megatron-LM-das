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
DATA_PATH=""                                                             # path to oscar-1GB_head-llama2_text_document
TOKENIZER_MODEL_PATH=""                                                  # HuggingFace path to model. example Qwen/Qwen3-32B
LAUNCHER="mpirun"                                                        # mpirun or torchrun
CHECKPOINT_PATH=""                                                       # path to ckpt
NCCL_ENV=${MEGATRON_PATH}/requirements/env.sh                            # Please adjust the variables based on the actual NET being used
LAUNCH_WITH_BINDING=${MEGATRON_PATH}/requirements/launch_with_binding.sh # Please adjust the variables based on the actual NET being used

# Those variables no need to modify
hostfile_input=${1}
node_num=${2}

if [[ "${LAUNCHER}" != "mpirun" && "${LAUNCHER}" != "torchrun" ]]; then
    echo "Only mpirun and torchrun are supported as launch methods"
    exit 1
fi

HOSTFILE="${hostfile_input}_slots"
rm -f ${HOSTFILE} 
if [[ "${LAUNCHER}" == "mpirun" ]]; then
    cat ${hostfile_input} | sed -n "1,${node_num}p"|sed 's/$/ slots=8/' > ${HOSTFILE}
else
    cat ${hostfile_input} | sed -n "1,${node_num}p"|sed 's/$/ slots=1/' > ${HOSTFILE}
fi

HOST="$(cat ${HOSTFILE} |sed -n "1p"|awk -F ' ' '{print $1}')"

NNODES=$(cat ${HOSTFILE} | sort | uniq | wc -l)
if [[ "$LAUNCHER" == "mpirun" ]]; then
    MPIRUN_NP=$((${NNODES}*8))
    PORT=${PORT:-25906}
else
    MPIRUN_NP=${NNODES}
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
    MASTER_ADDR=${MASTER_ADDR:-${HOST}}
    MASTER_PORT=${MASTER_PORT:-11452}
fi

torchrun_args=""
if [[ "${LAUNCHER}" == "torchrun" ]]; then
    torchrun_args+="
        -x PATH \
        -x LD_LIBRARY_PATH \
        -x PYTHONPATH \
        -x MASTER_ADDR=${MASTER_ADDR} \
        -x MASTER_PORT=${MASTER_PORT} \
        -x NNODES=${NNODES} \
        -x GPUS_PER_NODE=${GPUS_PER_NODE} \
    "
fi

# Runs qwen3 model
source ${NCCL_ENV}

CMD="mpirun -np ${MPIRUN_NP}  --hostfile ${HOSTFILE} \
    --allow-run-as-root \
    --bind-to none \
    --mca plm_rsh_no_tree_spawn 1 \
    --mca plm_rsh_args '-p 11451' \
    ${torchrun_args} \
    bash -c '
    source ${DTK_ENV} && \
    source ${NCCL_ENV} && \
    bash train_qwen3_8B.sh \
    ${HOST} \
    ${PORT} \
    --data_path=$DATA_PATH \
    --launch_backend=$LAUNCHER \
    --tokenizer_path=$TOKENIZER_MODEL_PATH \
    --checkpoint_path=$CHECKPOINT_PATH \
    --launch_with_binding=${LAUNCH_WITH_BINDING} \
    --profiling=$profiling' 2>&1|tee log-${NNODES}nodes-`date +%F-%H%M`.log
"

eval ${CMD}
wait

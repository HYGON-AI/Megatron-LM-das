#!/bin/bash

for para in "$@"
do
    if [[ $para == --profiling* ]]; then
        profiling=${para#*=}
    fi
done

# Those variables no need to modify
hostfile_input=${1}
node_num=${2}

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))

# Those variables need to modify
DTK_ENV=""                                                               # where env.sh of dtk
DATA_PATH=""                                                             # path to oscar-1GB_head-llama2_text_document
TOKENIZER_MODEL_PATH=""                                                  # path to tokenizer.model
LAUNCHER="mpirun"                                                        # mpirun or torchrun
CHECKPOINT_PATH=""                                                       # path to ckpt
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_llama3_8B.sh}                         # script under this example directory to run
NCCL_ENV=${MEGATRON_PATH}/requirements/env.sh                            # Please adjust the variables based on the actual NET being used
LAUNCH_WITH_BINDING=${MEGATRON_PATH}/requirements/launch_with_binding.sh # Please adjust the variables based on the actual NET being used

if [[ "${LAUNCHER}" != "mpirun" && "${LAUNCHER}" != "torchrun" ]]; then
    echo "Only mpirun and torchrun are supported as launch methods"
    exit 1
fi

HOSTFILE="${hostfile_input}_slots"
rm -f "${HOSTFILE}"
if [[ "${LAUNCHER}" == "mpirun" ]]; then
    sed -n "1,${node_num}p" "${hostfile_input}" | sed 's/$/ slots=8/' > "${HOSTFILE}"
else
    sed -n "1,${node_num}p" "${hostfile_input}" | sed 's/$/ slots=1/' > "${HOSTFILE}"
fi

HOST="$(sed -n "1p" "${HOSTFILE}" | awk -F ' ' '{print $1}')"

if [[ "${LAUNCHER}" == "mpirun" ]]; then
    MPIRUN_NP=$((node_num * 8))
    PORT=${PORT:-25900}
else
    MPIRUN_NP=${node_num}
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
    MASTER_ADDR=${MASTER_ADDR:-${HOST}}
    MASTER_PORT=${MASTER_PORT:-25900}
fi

torchrun_args=""
if [[ "${LAUNCHER}" == "torchrun" ]]; then
    torchrun_args+="
        -x PATH \
        -x LD_LIBRARY_PATH \
        -x PYTHONPATH \
        -x MASTER_ADDR=${MASTER_ADDR} \
        -x MASTER_PORT=${MASTER_PORT} \
        -x NNODES=${node_num} \
        -x GPUS_PER_NODE=${GPUS_PER_NODE} \
    "
fi

# Runs Llama3 model
source "${NCCL_ENV}"

CMD="mpirun -np ${MPIRUN_NP} --hostfile ${HOSTFILE} \
    --allow-run-as-root \
    --bind-to none \
    --mca plm_rsh_no_tree_spawn 1 \
    ${torchrun_args} \
    bash -c '
    source ${DTK_ENV} && \
    source ${NCCL_ENV} && \
    cd ${CURRENT_DIR} && \
    bash ${TRAIN_SCRIPT} \
    ${HOST} \
    ${PORT:-${MASTER_PORT}} \
    --data_path=${DATA_PATH} \
    --launch_backend=${LAUNCHER} \
    --tokenizer_path=${TOKENIZER_MODEL_PATH} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --launch_with_binding=${LAUNCH_WITH_BINDING} \
    --profiling=${profiling}' 2>&1 | tee log-${node_num}nodes-$(date +%F-%H%M).log
"

eval ${CMD}
wait

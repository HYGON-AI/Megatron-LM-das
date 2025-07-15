for para in $*
do
    if [[ $para == --profiling* ]];then
        profiling=${para#*=}
    fi
done

# Those variables need to modify
GPUS=""                 # how many gpus to use
DTK_ENV=""              # where env.sh of dtk
NCCL_ENV=""             # where env.sh of nccl (requirements/nccl_wz/env.sh or requirements/nccl_zz/env.sh)
HOST=""                 # hostname
PORT=""                 # port id
DATA_PATH=""            # path to redpajama_text_document
TOKENIZER_MODEL_PATH="" # path to tokenizer.model
CHECKPOINT_PATH=""      # path to ckpt

# Runs GPT 567B model
mpirun -np ${GPUS}  --hostfile hostfile_gpt_567B \
                    --allow-run-as-root \
                    --bind-to none \
                    --mca plm_rsh_no_tree_spawn 1 \
                    bash -c "
                    source ${DTK_ENV} && \
                    source ${NCCL_ENV} && \
                    ./train_gpt_567B_$((${GPUS} / 8))nodes.sh \
                    ${HOST} \
                    ${PORT} \
                    --data_path=$DATA_PATH \
                    --tokenizer_path=$TOKENIZER_MODEL_PATH \
                    --checkpoint_path=$CHECKPOINT_PATH \
                    --profiling=$profiling" > log-$((${GPUS} / 8))nodes-`date +%F-%H%M`.log 2>&1

wait
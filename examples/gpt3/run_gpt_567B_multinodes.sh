for para in $*
do
    if [[ $para == --profiling* ]];then
        profiling=${para#*=}
    fi
done

mpirun -np 512 --hostfile hostfile_gpt_567B \
              --allow-run-as-root \
              --bind-to none \
              --mca plm_rsh_no_tree_spawn 1 \
              train_gpt_567B_multinodes.sh node059 --profiling=$profiling > output.log 2>&1

wait

rm -rf CKPT
rm -rf gpt_dataset/redpajama_text_document
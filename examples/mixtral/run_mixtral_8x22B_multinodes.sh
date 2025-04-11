for para in $*
do
    if [[ $para == --profiling* ]];then
        profiling=${para#*=}
    fi
done

mpirun -np 64 --hostfile hostfile_mixtral_8x22B \
              --allow-run-as-root \
              --bind-to none \
              --mca plm_rsh_no_tree_spawn 1 \
              train_mixtral_8x22B_multinodes.sh node067 --profiling=$profiling > output.log 2>&1

wait

rm -rf CKPT
rm -rf mixtral_dataset/my-mixtral_text_document
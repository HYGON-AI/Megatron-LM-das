for para in $*
do
    if [[ $para == --profiling* ]];then
        profiling=${para#*=}
        export GPU_FLUSH_ON_EXECUTION=1
        export HIP_DIRECT_DISPATCH=0
    fi
done

mpirun -np 8  --allow-run-as-root \
              train_deepseek_v3_1node.sh localhost --profiling=$profiling > output.log 2>&1

wait

rm -rf CKPT

for para in $*
do
    if [[ $para == --profiling* ]];then
        profiling=${para#*=}
    fi
done

mpirun -np 8  --allow-run-as-root \
              train_mixtral_8x22B_1nodes.sh localhost --profiling=$profiling > output.log 2>&1

wait

rm -rf CKPT
rm -rf mixtral_dataset/my-mixtral_text_document

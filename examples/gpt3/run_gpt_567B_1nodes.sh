for para in $*
do
    if [[ $para == --profiling* ]];then
        profiling=${para#*=}
    fi
done

mpirun -np 8  --allow-run-as-root \
              train_gpt_567B_1nodes.sh localhost --profiling=$profiling > output.log 2>&1

wait

rm -rf CKPT
rm -rf gpt_dataset/redpajama_text_document

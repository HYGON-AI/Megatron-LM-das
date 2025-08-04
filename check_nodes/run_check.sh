
# 执行单机或多机分组
cat ./clushnode |sed 's/$/ slots=8/' > hostfile
hosts="./hostfile"
nodenum=${1}
if [ "$nodenum" -eq 1 ]; then
    average_tflops=210
elif [ "$nodenum" -eq 4 ]; then
    average_tflops=185
else
    echo "Unsupported nodenum: $nodenum"
    exit 1
fi

rm -rf hostslice node_checklog
mkdir -p hostslice node_checklog

# 统计hostfile中的节点数量
total_num=`grep -cve '^\s*$' ${hosts}`
((total_num=total_num/${nodenum} * ${nodenum}))
echo "total_num: ${total_num}"

for((i=1;i<=${total_num};i=i+${nodenum}))
do
	((j=i+${nodenum}-1))
	((k=j/${nodenum}))
	echo i,j,k = ${i},${j},${k}
	cat ${hosts} | sed -n "${i},${j}p" > ./hostslice/hostmp${k}

	sleep 1

	./check_nodes.sh ./hostslice/hostmp${k} > ./node_checklog/output_${k}.log 2>&1 &

	sleep 1
done

wait


# 检查性能
rm -rf ./node_checklog/host_check
rm -rf ./node_checklog/host_check_error

for i in `ls node_checklog/output_*.log`; do
    tflops=$(grep "TFLOP/s/GPU" "$i" | awk -F':' '{print $6}' | awk '{print $1}' | tail -n 1)
    index=$(basename "$i" | awk -F '_' '{print $2}' | awk -F '.' '{print $1}')

    if [ -z "${tflops}" ]; then
        echo "${i} error"
        awk -v outname=$(basename "$i") '{print $1, "> " outname}' hostslice/hostmp${index} >> ./node_checklog/host_check_error

    elif [ $(echo "${tflops} < ${average_tflops}" | bc) -eq 1 ]; then
        echo "${tflops} tflops of ${i} less than average ${average_tflops} tflops"
        awk -v outname=$(basename "$i") '{print $1, "> " outname}' hostslice/hostmp${index} >> ./node_checklog/host_check_error

    else
        echo "All nodes performance is normal"
        awk -v outname=$(basename "$i") '{print $1, "> " outname}' hostslice/hostmp${index} >> ./node_checklog/host_check
    fi
done

wait

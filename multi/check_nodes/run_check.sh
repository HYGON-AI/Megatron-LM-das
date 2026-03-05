#!/bin/bash 
start_time=$(date +%s)
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# 执行单机或多机分组
cat ./clushnode |sed 's/$/ slots=8/' > hostfile
hosts="./hostfile"
nodenum=${1}
if [ "$nodenum" -eq 1 ]; then
    average_tflops=210
elif [ "$nodenum" -eq 4 ]; then
    average_tflops=185
elif [ "$nodenum" -eq 8 ]; then
    average_tflops=200
elif [ "$nodenum" -eq 512 ]; then
    average_tflops=200
elif [ "$nodenum" -eq 1024 ]; then
    average_tflops=200
else
    echo "Unsupported nodenum: $nodenum"
    exit 1
fi

rm -rf hostslice node_checklog
mkdir -p hostslice node_checklog/tmp_results

# 统计hostfile中的节点数量
total_num=`grep -cve '^\s*$' ${hosts}`
((total_num=total_num/${nodenum} * ${nodenum}))
echo "total_num: ${total_num}"

pids=()
for((i=1;i<=${total_num};i=i+${nodenum}))
do
    ((j=i+${nodenum}-1))
    ((k=j/${nodenum}))
    echo i,j,k = ${i},${j},${k}
    cat ${hosts} | sed -n "${i},${j}p" > ./hostslice/hostmp${k}
    ./check_nodes.sh ./hostslice/hostmp${k} \
        > ./node_checklog/output_${k}.log 2>&1 &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait $pid
done

# 检查性能
MAX_JOBS=1000
job_count=0
pids=()

cd node_checklog
for i in output_*.log; do
{
    tflops=$(grep "TFLOP/s/GPU" "$i" | awk -F':' '{print $6}' | awk '{print $1}' | tail -n 1)
    index=$(basename "$i" | awk -F '_' '{print $2}' | awk -F '.' '{print $1}')
    current_hostfile=../hostslice/hostmp${index}
    node_list=$(awk '{print $1}' ${current_hostfile} | paste -sd,)

    tmp_file="./tmp_results/result_${index}.txt"

    if [ -z "${tflops}" ]; then
        printf "${RED}%-6s${NC} [%s]: %s has no tflops\n" \
            "[ERROR]" "${node_list}" "${i}" | tee "$tmp_file"
        awk -v outname=$(basename "$i") '{print $1, "> " outname}' ${current_hostfile} > "./tmp_results/error_${index}.txt"

    elif [ "$(echo "${tflops} < ${average_tflops}" | bc)" -eq 1 ]; then
        printf "${RED}%-6s${NC} [%s]: %s tflops of %s is less than average %s tflops\n" \
            "[ERROR]" "${node_list}" "${tflops}" "${i}" "${average_tflops}" | tee "$tmp_file"
        awk -v outname=$(basename "$i") '{print $1, "> " outname}' ${current_hostfile} > "./tmp_results/error_${index}.txt"

    else
        printf "${GREEN}%-6s${NC} [%s]: %s tflops of %s is greater than average %s tflops\n" \
            "[PASS]" "${node_list}" "${tflops}" "${i}" "${average_tflops}" | tee "$tmp_file"
        awk -v outname=$(basename "$i") '{print $1, "> " outname}' ${current_hostfile} > "./tmp_results/pass_${index}.txt"
    fi
} &

# 控制并发数量
((job_count++))
if (( job_count % MAX_JOBS == 0 )); then
    wait
fi
done

wait

# 合并结果（仅在文件存在时执行）
shopt -s nullglob  # 避免 *.txt 无匹配时报错
error_files=(./tmp_results/error_*.txt)
pass_files=(./tmp_results/pass_*.txt)

if ((${#error_files[@]})); then
    cat "${error_files[@]}" >> ./host_check_error
fi

if ((${#pass_files[@]})); then
    cat "${pass_files[@]}" >> ./host_check_pass
fi

# === 汇总统计 ===
echo -e "\n========== SUMMARY =========="

# 保证变量不为空
pass_count=$(grep -cve '^\s*$' ./host_check_pass 2>/dev/null || echo 0)
error_count=$(grep -cve '^\s*$' ./host_check_error 2>/dev/null || echo 0)
pass_count=${pass_count:-0}
error_count=${error_count:-0}
total_count=$((pass_count + error_count))

if ! [[ "$total_count" =~ ^[0-9]+$ ]]; then
    total_count=0
fi

if (( total_count == 0 )); then
    echo -e "${RED}[ERROR]${NC} No results found — please check logs or input settings."
    exit 1
fi

# 提取失败节点名
if (( error_count > 0 )); then
    fail_nodes=$(awk '{print $1}' ./host_check_error | sort -u | paste -sd, -)
    fail_nodes="[${fail_nodes}]"
else
    fail_nodes="[]"
fi

echo -e "Total nodes checked : ${total_count}"
echo -e "${GREEN}PASS nodes          : ${pass_count}${NC}"
echo -e "${RED}FAIL nodes          : ${error_count}${NC}"
echo -e "${RED}FAIL nodes list     : ${fail_nodes}${NC}"

if (( error_count > 0 )); then
    echo -e "\n${RED}❌ Some nodes failed performance check.${NC}"
    echo -e "See detailed log in: ${PWD}/host_check_error"
else
    echo -e "\n${GREEN}✅ All nodes passed performance check!${NC}"
fi

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf "Total time: %02d min %02d sec\n" $((elapsed/60)) $((elapsed%60))
echo "=============================="
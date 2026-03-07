clushnode='./clushnode' 
dtk_env='compiler/dtk/25.04.4'

clush --hostfile=$clushnode -f 1000 -b "free -g | grep -i mem" | sort -n -k 4
clush --hostfile=$clushnode -f 1000 -b "ps -ef | grep python | grep -v grep | grep -v gridview | grep -v platform-python | grep -v resource_tracker | wc -l"
clush --hostfile=$clushnode -f 1000 -b "module load ${dtk_env} && rocm-smi --showmemuse | grep 'HCU memory use'"
clush --hostfile=$clushnode -f 1000 -b "module load ${dtk_env} && rocminfo | grep amdgcn-amd-amdhsa--gfx936 | wc -l"
clush --hostfile=$clushnode -f 1000 -b "netstat -ant|awk '{print \$5}' | grep 25905  | wc -l"

# clush --hostfile=$clushnode -f 1000 -b "module load app/miniconda3/25.11.0"
# clush --hostfile=$clushnode -f 1000 -b "hy-smi"

ENV="/opt/dtk/env.sh"
clushnode='./clushnode' 
clush --hostfile=$clushnode "free -g | grep -i mem" | sort -k 3
clush --hostfile=$clushnode -f 100 -b "ps -ef | grep python | grep -v grep | grep -v gridview | grep -v platform-python | grep -v resource_tracker | wc -l"

clush --hostfile=$clushnode -f 100 -b "source ${ENV} && rocm-smi --showmemuse | grep 'HCU memory use'"
clush --hostfile=$clushnode -f 100 -b "source ${ENV} && rocminfo | grep amdgcn-amd-amdhsa--gfx936 | wc -l"

clush --hostfile=$clushnode -f 100 -b "rdma resource"
clush --hostfile=$clushnode -f 100 -b "ibstat | grep Active | wc -l"
clush --hostfile=$clushnode -f 100 -b "ibstat | grep Rate"

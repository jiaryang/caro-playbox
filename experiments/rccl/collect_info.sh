#!/usr/bin/env bash
# Collect host / ROCm / topology / RCCL library fingerprints for cross-node compare.
# Usage: collect_info.sh <out_info_dir>
set -euo pipefail

OUT="${1:?usage: collect_info.sh <out_info_dir>}"
mkdir -p "$OUT"

# --- host ---
{
  echo "hostname: $(hostname -f 2>/dev/null || hostname)"
  echo "date: $(date -Is)"
  echo "uname: $(uname -a)"
  echo
  echo "=== /proc/cpuinfo (model) ==="
  grep -m1 'model name' /proc/cpuinfo 2>/dev/null || true
  echo "cpu_count: $(nproc)"
  echo
  echo "=== /proc/meminfo (head) ==="
  head -n 5 /proc/meminfo 2>/dev/null || true
  echo
  echo "=== uptime ==="
  uptime 2>/dev/null || true
} >"${OUT}/host.txt"

# --- env ---
{
  echo "USER=${USER:-}"
  echo "HOME=${HOME:-}"
  echo "PWD=$(pwd)"
  echo "SHELL=${SHELL:-}"
  echo "PATH=${PATH:-}"
  echo
  echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-<unset>}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}"
  echo
  env | grep -E '^(NCCL_|RCCL_|HSA_|HIP_|ROCR_|ROC_|AMD_|GPU_)' | sort || true
} >"${OUT}/env.txt"

# --- docker / cgroup best-effort ---
{
  if [[ -f /.dockerenv ]]; then
    echo "in_docker: yes (.dockerenv present)"
  else
    echo "in_docker: maybe/no (no /.dockerenv)"
  fi
  echo
  echo "=== cgroup ==="
  cat /proc/1/cgroup 2>/dev/null | head -n 20 || true
  echo
  echo "=== hostname inside ==="
  hostname -f 2>/dev/null || hostname
} >"${OUT}/docker.txt"

# --- ROCm / driver ---
{
  echo "=== /opt/rocm/.info/version ==="
  cat /opt/rocm/.info/version 2>/dev/null || echo "<missing>"
  echo
  echo "=== ROCK / amdgpu module ==="
  if command -v rocminfo >/dev/null 2>&1; then
    rocminfo 2>/dev/null | grep -i version || true
  fi
  cat /sys/module/amdgpu/version 2>/dev/null && echo "(from /sys/module/amdgpu/version)" || true
  modinfo amdgpu 2>/dev/null | grep -E '^(version|filename|description):' || true
  echo
  echo "=== rocm-smi --showproductname / driver / fw ==="
  if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showproductname 2>&1 || true
    echo
    rocm-smi --showdriverversion 2>&1 || true
    echo
    rocm-smi --showfwinfo 2>&1 || true
  else
    echo "<rocm-smi not found>"
  fi
} >"${OUT}/rocm.txt"

# --- topology ---
{
  if command -v rocm-smi >/dev/null 2>&1; then
    echo "=== rocm-smi --showtopo ==="
    rocm-smi --showtopo 2>&1 || true
    echo
    echo "=== rocm-smi --showhw ==="
    rocm-smi --showhw 2>&1 || true
  fi
  if command -v amd-smi >/dev/null 2>&1; then
    echo
    echo "=== amd-smi topology ==="
    amd-smi topology 2>&1 || true
  fi
} >"${OUT}/topo.txt"

# --- live GPU snapshot ---
{
  if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi 2>&1 || true
    echo
    rocm-smi --showclocks --showtemp --showpower --showuse 2>&1 || true
  else
    echo "<rocm-smi not found>"
  fi
} >"${OUT}/gpus.txt"

# --- RCCL / related libs in container ---
{
  echo "=== librccl.so* ==="
  # shellcheck disable=SC2012
  ls -l /opt/rocm/lib/librccl.so* 2>/dev/null || ls -l /opt/rocm/*/lib/librccl.so* 2>/dev/null || echo "<not found under /opt/rocm>"
  echo
  echo "=== checksums ==="
  for f in /opt/rocm/lib/librccl.so* /opt/rocm/lib/librccl_static.a; do
    [[ -e "$f" ]] || continue
    if command -v sha256sum >/dev/null 2>&1; then
      sha256sum "$f" 2>/dev/null || true
    elif command -v md5sum >/dev/null 2>&1; then
      md5sum "$f" 2>/dev/null || true
    fi
  done
  echo
  echo "=== dpkg/rpm (best-effort) ==="
  dpkg -l 2>/dev/null | grep -iE 'rccl|rocm' | head -n 80 || true
  rpm -qa 2>/dev/null | grep -iE 'rccl|rocm' | head -n 80 || true
  echo
  echo "=== find librccl (capped) ==="
  find /opt/rocm /usr/local -name 'librccl.so*' 2>/dev/null | head -n 40 || true
} >"${OUT}/rccl_libs.txt"

# --- host tuning hints ---
{
  echo "=== numa_balancing ==="
  if [[ -r /proc/sys/kernel/numa_balancing ]]; then
    cat /proc/sys/kernel/numa_balancing
  else
    echo "<unreadable>"
  fi
  echo
  echo "=== grub iommu (best-effort) ==="
  if [[ -r /etc/default/grub ]]; then
    grep -E 'GRUB_CMDLINE|iommu' /etc/default/grub || true
  else
    echo "<cannot read /etc/default/grub>"
  fi
  echo
  echo "=== lscpu NUMA (head) ==="
  lscpu 2>/dev/null | grep -iE 'numa|socket|thread|cpu\(s\)|model name' || true
} >"${OUT}/host_tuning.txt"

# --- manifest snippet ---
{
  echo "info_dir=${OUT}"
  echo "collected_at=$(date -Is)"
  echo "hostname=$(hostname -f 2>/dev/null || hostname)"
  echo "hip_visible=${HIP_VISIBLE_DEVICES:-}"
  echo "files:"
  ls -1 "$OUT"
} >"${OUT}/INDEX.txt"

echo "Wrote info under ${OUT}"

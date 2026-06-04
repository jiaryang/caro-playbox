#!/usr/bin/env bash
set -euo pipefail

# Update the unified ROCm+AMDGPU build on this host.
# Ref: https://amd.atlassian.net/wiki/spaces/AMDGPU/pages/777579500
#
# Requires AMD internal network (VPN or office) to reach mkmartifactory.amd.com.
#
# Override any of these via env vars, e.g.
#   AMDGPU_BUILD=2343748 ROCM_BUILD=compute-rocm-dkms-no-npi-hipclang/17163 \
#     USECASE=rocm,dkms ./update_amdgpu_build.sh

AMDGPU_BUILD="${AMDGPU_BUILD:-2343748}"
ROCM_BUILD="${ROCM_BUILD:-compute-rocm-dkms-no-npi-hipclang/17163}"
USECASE="${USECASE:-dkms}"
INSTALLER_BRANCH="${INSTALLER_BRANCH:-26.12}"
CLEAN_FIRST="${CLEAN_FIRST:-0}"

# Detect Ubuntu version (e.g. 22.04, 24.04) unless overridden.
if [[ -z "${UBUNTU_VERSION:-}" ]]; then
  if [[ -r /etc/os-release ]]; then
    # shellcheck source=/dev/null
    . /etc/os-release
    UBUNTU_VERSION="${VERSION_ID:-22.04}"
  else
    UBUNTU_VERSION="22.04"
  fi
fi

echo "======================================================="
echo "AMDGPU/ROCm unified build update"
echo "  amdgpu-build (constructicon): ${AMDGPU_BUILD}"
echo "  rocm-build (job/build):       ${ROCM_BUILD}"
echo "  usecase:                      ${USECASE}"
echo "  installer branch:             ${INSTALLER_BRANCH}"
echo "  ubuntu:                       ${UBUNTU_VERSION}"
echo "======================================================="

if [[ "${CLEAN_FIRST}" == "1" ]]; then
  echo "--- Removing existing driver/build (CLEAN_FIRST=1)"
  sudo amdgpu-uninstall || true
  sudo amdgpu-repo --clean || true
fi

# A stale amdgpu-repo selection (e.g. a build that has since been purged from
# artifactory) leaves source files in /etc/apt/sources.list.d/ that make the
# very first `apt update` fail with a 404. Step 2 (amdgpu-repo) regenerates
# these from the requested build, so it's safe to drop them up front.
echo "--- Clearing stale amdgpu/rocm build source lists"
sudo rm -f /etc/apt/sources.list.d/amdgpu-build.list \
           /etc/apt/sources.list.d/rocm-build.list

echo "--- Step 1: install/update repo installer package"
sudo apt update
sudo apt dist-upgrade -y

INSTALLER="amdgpu-install-internal_${INSTALLER_BRANCH}-"
DEB="${INSTALLER}${UBUNTU_VERSION}-1_all.deb"
wget -N -P /tmp/ "https://mkmartifactory.amd.com/artifactory/list/amdgpu-deb/${DEB}"
sudo apt-get install -y "/tmp/${DEB}"

echo "--- Step 2: select build"
sudo amdgpu-repo --amdgpu-build="${AMDGPU_BUILD}" --rocm-build="${ROCM_BUILD}"

echo "--- Step 3: install driver packages (usecase=${USECASE})"
sudo amdgpu-install -y --usecase="${USECASE}"

echo "======================================================="
echo "Done. Verify with:"
echo "  rocminfo | grep -i gfx"
echo "  dkms status"
echo "  groups   # ensure 'render' group; re-login if just added"
echo "======================================================="

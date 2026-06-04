#!/usr/bin/env bash
set -euo pipefail

CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="${CONDA_ENV_NAME:-py_3.11}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

case "$(uname -m)" in
  x86_64) INSTALLER="Miniconda3-latest-Linux-x86_64.sh" ;;
  aarch64) INSTALLER="Miniconda3-latest-Linux-aarch64.sh" ;;
  *)
    echo "Unsupported architecture: $(uname -m)"
    exit 1
    ;;
esac
INSTALLER_URL="https://repo.anaconda.com/miniconda/${INSTALLER}"

echo "======================================================="
echo "Installing Miniconda to ${CONDA_DIR}"
echo "======================================================="
if [ -x "${CONDA_DIR}/bin/conda" ]; then
  echo "Conda already installed, skipping."
else
  tmp_installer="$(mktemp /tmp/miniconda.XXXXXX.sh)"
  curl -fsSL "${INSTALLER_URL}" -o "${tmp_installer}"
  bash "${tmp_installer}" -b -p "${CONDA_DIR}"
  rm -f "${tmp_installer}"
fi

# shellcheck source=/dev/null
source "${CONDA_DIR}/etc/profile.d/conda.sh"

echo "======================================================="
echo "Accepting Anaconda channel Terms of Service"
echo "======================================================="
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

echo "======================================================="
echo "Initializing conda for bash"
echo "======================================================="
"${CONDA_DIR}/bin/conda" init bash

echo "======================================================="
echo "Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
echo "======================================================="
if [ -d "${CONDA_DIR}/envs/${ENV_NAME}" ]; then
  echo "Environment ${ENV_NAME} already exists, skipping."
else
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

echo "======================================================="
echo "Done. Run:"
echo "  source ${CONDA_DIR}/etc/profile.d/conda.sh"
echo "  conda activate ${ENV_NAME}"
echo "======================================================="

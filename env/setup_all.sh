#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="${CONDA_ENV_NAME:-py_3.11}"

bash "${SCRIPT_DIR}/setup_conda.sh"

# shellcheck source=/dev/null
source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

bash "${SCRIPT_DIR}/install_the_rock.sh"
bash "${REPO_ROOT}/pyt/ut_smoke.sh"

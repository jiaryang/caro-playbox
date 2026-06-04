#!/usr/bin/env bash
set -euo pipefail

echo "======================================================="
echo "PyTorch ROCm smoke test"
echo "======================================================="
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"

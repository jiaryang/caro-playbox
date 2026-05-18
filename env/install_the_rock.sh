INDEX_URL="https://rocm.nightlies.amd.com/v2-staging/gfx94X-dcgpu/"
VERSION_TAG="7.13.0a20260512"

echo "======================================================="
echo "Installing ROCm Base Libraries & Development Tools"
echo "======================================================="
python -m pip install \
  -i ${INDEX_URL} \
  rocm[libraries,devel]==${VERSION_TAG}

echo "======================================================="
echo "Installing PyTorch & Ecosystem (Nightly/Staging)"
echo "======================================================="
python -m pip install \
  -i ${INDEX_URL} \
  torch==2.11.0+rocm${VERSION_TAG} \
  torchaudio==2.11.0+rocm${VERSION_TAG} \
  torchvision==0.26.0+rocm${VERSION_TAG} \
  triton==3.6.0+rocm${VERSION_TAG}
 
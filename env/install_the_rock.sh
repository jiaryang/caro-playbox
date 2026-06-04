INDEX_URL="https://rocm.nightlies.amd.com/v2-staging/gfx94X-dcgpu/"

if rocminfo 2>/dev/null | grep -qi 'gfx950'; then
  INDEX_URL="https://rocm.nightlies.amd.com/v2-staging/gfx950-dcgpu/"
fi

VERSION_TAG="7.14.0a20260603"

echo "======================================================="
echo "Installing ROCm Base Libraries & Development Tools"
echo "======================================================="
python -m pip install \
  -i ${INDEX_URL} \
  rocm[libraries,devel]==${VERSION_TAG}

echo "Checking rocm-sdk..."
rocm-sdk path --bin
#rocm-sdk test

echo "======================================================="
echo "Installing PyTorch & Ecosystem (Nightly/Staging)"
echo "======================================================="
python -m pip install \
  -i ${INDEX_URL} \
  torch==2.13.0a0+rocm${VERSION_TAG} \
  torchaudio==2.11.0a0+rocm${VERSION_TAG} \
  torchvision==0.26.0+rocm${VERSION_TAG} \
  triton==3.7.0+git3169ee52.rocm${VERSION_TAG}

echo "======================================================="
echo "Installed Versions"
echo "======================================================="
python <<'EOF'
import importlib.metadata as m

for pkg in ("rocm", "torch", "triton"):
    try:
        print(f"{pkg}: {m.version(pkg)}")
    except m.PackageNotFoundError:
        print(f"{pkg}: not installed")
EOF
 
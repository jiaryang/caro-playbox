rm -rf build
rm -rf torch.egg-info
rm -f compile_commands.json

export ROCM_DEVEL=/home/jiaryang/miniconda3/envs/py_3.11/lib/python3.11/site-packages/_rocm_sdk_devel
export ROCM_CORE=/home/jiaryang/miniconda3/envs/py_3.11/lib/python3.11/site-packages/_rocm_sdk_core

export ROCM_PATH="$ROCM_DEVEL"
export HIP_PATH="$ROCM_DEVEL"
export ROCM_SOURCE_DIR="$ROCM_DEVEL"
export ROCM_CORE="$ROCM_CORE"
                                                                                                                                                        export HIP_CLANG_PATH="$ROCM_CORE/lib/llvm/bin"
export HIP_HIPCONFIG_EXECUTABLE="$ROCM_DEVEL/bin/hipconfig"
export HIP_ROOT_DIR="$ROCM_DEVEL"

export PATH="$ROCM_DEVEL/bin:$ROCM_CORE/bin:$ROCM_CORE/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_DEVEL/lib:$ROCM_CORE/lib:$ROCM_CORE/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$ROCM_DEVEL/lib:$ROCM_CORE/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CMAKE_PREFIX_PATH="$ROCM_DEVEL:$ROCM_CORE${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
export USE_ROCM=1
export PYTORCH_ROCM_ARCH=gfx942                                                                                                                      export CMAKE_FRESH=1


git config --global --add safe.directory ~/pytorch-main/pytorch && \
    git submodule sync && \
    git submodule update --init --recursive  --progress && \
    python tools/amd_build/build_amd.py

MAX_JOBS=$(nproc) PYTORCH_ROCM_ARCH="gfx942"  USE_MKLDNN=0 USE_ROCM=1  USE_AOTRITON=0   BUILD_TEST=0 CMAKE_PREFIX_PATH=$(python -c 'import sys; print(f"{sys.prefix}")')  python -m pip install --no-build-isolation -v -e . 2>&1 | tee build.log
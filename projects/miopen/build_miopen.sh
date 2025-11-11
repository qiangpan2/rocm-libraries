#!/bin/bash

export PATH=/opt/ompi/bin:/opt/ucx/bin:/opt/cache/bin:/opt/rocm/llvm/bin:/opt/rocm/opencl/bin:/opt/rocm/hip/bin:/opt/rocm/hcc/bin:/opt/rocm/bin:/opt/conda/envs/py_3.12/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin;
export DEPS_PREFIX="${HOME}/miopen-deps"
export MIOPEN_PREFIX="${HOME}/miopen-install"

# 配置 CMake with proper GPU target flags
echo "Configuring CMake..."
cmake -B build \
    -DCMAKE_PREFIX_PATH="${DEPS_PREFIX}" \
    -DCMAKE_INSTALL_PREFIX="${MIOPEN_PREFIX}" \
    -DMIOPEN_BACKEND=HIP \
    -DMIOPEN_USE_COMPOSABLEKERNEL=ON \
    -DMIOPEN_USE_CKTILE_COMPOSABLEKERNEL=ON \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
    -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DBUILD_TESTING=ON \
    -DCMAKE_CXX_FLAGS="-I/root/miopen-deps/include" \
    -DCMAKE_HIP_FLAGS="-I/root/miopen-deps/include" \
    -G Ninja --debug-output > cmake_config.log 2>&1

# 构建项目
cmake --build build -j8 > build.log 2>&1

# 3dconv solver test
LD_LIBRARY_PATH=/workspace/repo/rocm-libraries/projects/miopen/build/lib:$LD_LIBRARY_PATH

export HIP_VISIBLE_DEVICES=1
export MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_CHANNEL_LAST_FWD_WMMAOPS=1
./bin/test_conv3d_channel_last_wmmaops 
./bin/MIOpenDriver convfp16 -n 1 -c 16 --in_d 5 -H 104 -W 60 -k 16 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 --in_layout NDHWC --fil_layout NDHWC --out_layout NDHWC -m conv -g 1 -F 1 -t 1

# 安装项目
echo "Installing MIOpen..."
cmake --install build

echo "Build completed successfully!"
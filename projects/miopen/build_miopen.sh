#!/bin/bash
export HIP_VISIBLE_DEVICES=1
export PATH=/opt/ompi/bin:/opt/ucx/bin:/opt/cache/bin:/opt/rocm/llvm/bin:/opt/rocm/opencl/bin:/opt/rocm/hip/bin:/opt/rocm/hcc/bin:/opt/rocm/bin:/opt/conda/envs/py_3.12/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin;
export DEPS_PREFIX="${HOME}/miopen-deps"
export GPU_TARGETS="gfx1100;gfx1201"

# 配置 CMake with proper GPU target flags
echo "Configuring CMake..."
cmake -B build \
    -DCMAKE_PREFIX_PATH="${DEPS_PREFIX}" \
    -DCMAKE_INSTALL_PREFIX="${DEPS_PREFIX}" \
    -DMIOPEN_BACKEND=HIP \
    -DMIOPEN_USE_COMPOSABLEKERNEL=ON \
    -DMIOPEN_USE_CKTILE_COMPOSABLEKERNEL=ON \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DGPU_TARGETS="${GPU_TARGETS}" \
    -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
    -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DBUILD_TESTING=ON \
    -DCMAKE_CXX_FLAGS="-I${DEPS_PREFIX}/include" \
    -DCMAKE_HIP_FLAGS="-I${DEPS_PREFIX}/include" \
    -G Ninja > cmake_config.log 2>&1

# 构建项目
cmake --build build -j8 2>&1 | tee build.log

# 3dconv solver test
LD_LIBRARY_PATH=/workspace/repo/rocm-libraries/projects/miopen/build/lib:$LD_LIBRARY_PATH


export MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_CHANNEL_LAST_FWD_WMMAOPS=1
export MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD=0
# 强制只使用指定的 solver，跳过其他 solver 的评估
export MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvHipImplicitGemm3DChannelLastFwdWmmaops
export MIOPEN_ENABLE_LOGGING=1
export MIOPEN_ENABLE_LOGGING_CMD=1
export MIOPEN_LOG_LEVEL=6


./bin/test_conv3d_channel_last_wmmaops 
./bin/MIOpenDriver convfp16 -n 1 -c 16 --in_d 5 -H 104 -W 60 -k 16 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 --in_layout NDHWC --fil_layout NDHWC --out_layout NDHWC -m conv -g 1 -F 1 -t 1

./bin/MIOpenDriver convbfp16 -n 1 -c 16 --in_d 5 -H 104 -W 60 -k 16 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 --in_layout NDHWC --fil_layout NDHWC --out_layout NDHWC -m conv -g 1 -F 1 -t 1

#hang case check
MIOpenDriver convbfp16 -n 1 -c 1024 --in_d 3 -H 138 -W 102 -k 1024 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 --in_layout NDHWC --fil_layout NDHWC --out_layout NDHWC -m conv -g 1 -F 1 -t 1 -V 0 

MIOpenDriver convbfp16 -n 1 -c 1024 --in_d 1 -H 272 -W 200 -k 512 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 --in_layout NDHWC --fil_layout NDHWC --out_layout NDHWC -m conv -g 1 -F 1 -t 1 -V 0 > convbfp16_hang.log


export MIOPEN_DEBUG_2D_CONV_IMPLICIT_GEMM_HIP_GROUPED_FWD_DLOPS=1
export MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvHipImplicitGemm2DGroupedFwdDlops
./bin/MIOpenDriver conv -n 1 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 --spatial_dim 2 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -m conv -g 1 -F 1 -t 1 -V 0 > convfp32_dlops.log

# 安装项目
echo "Installing MIOpen..."
cmake --install build

echo "Build completed successfully!"
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/dispatcher/grouped_conv_problem.hpp"
#include "ck_tile/dispatcher/grouped_conv_registry.hpp"
#include "ck_tile/dispatcher/register_all_grouped_conv_kernels.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <string>

namespace {
constexpr int skip_code = 77;
std::string normalize_arch(std::string arch)
{
    const auto separator = arch.find(58);
    if(separator != std::string::npos) arch.resize(separator);
    return arch;
}
bool check_catalog(const ck_tile::dispatcher::GroupedConvRegistry& registry)
{
    bool found_2d = false;
    bool found_3d = false;
    for(const auto* kernel : registry.all_kernels())
    {
        if(kernel == nullptr || !kernel->executable()) return false;
        const auto& key = kernel->key();
        if(key.arch != "gfx1100" || key.dtype_in != "fp16" || key.dtype_wei != "fp16" ||
           key.dtype_out != "fp16" || key.op != ck_tile::dispatcher::GroupedConvOp::Forward ||
           key.wave_size != 32 || key.instruction_family != "wmma" ||
           key.pipeline != "compv3" || key.memory_operation != "set") return false;
        found_2d = found_2d || key.ndim_spatial == 2;
        found_3d = found_3d || key.ndim_spatial == 3;
    }
    return found_2d && found_3d;
}

ck_tile::dispatcher::GroupedConvProblem make_problem(int ndim, int groups)
{
    using ck_tile::dispatcher::GroupedConvProblemBuilder;
    auto builder = GroupedConvProblemBuilder{}.batch(1).channels(64 * groups, 64 * groups)
                       .groups(groups).data_types("fp16", "fp16", "fp16").arch("gfx1100");
    if(ndim == 2)
        return builder.ndim(2)
            .layout("nhwgc_gkyxc_nhwgk")
            .input_size(4, 4)
            .filter_size(1, 1)
            .build();
    return builder.ndim(3)
        .layout("ndhwgc_gkzyxc_ndhwgk")
        .input_size(2, 3, 3)
        .filter_size(1, 1, 1)
        .build();
}

bool run_case(ck_tile::dispatcher::GroupedConvRegistry& registry,
              int device,
              hipStream_t stream,
              int ndim,
              int groups)
{
    const auto problem = make_problem(ndim, groups);
    const std::size_t spatial = ndim == 2 ? 16 : 18;
    const std::size_t input_count = spatial * static_cast<std::size_t>(problem.C);
    const std::size_t weight_count = static_cast<std::size_t>(problem.K * problem.C / problem.G);
    const std::size_t output_count = spatial * static_cast<std::size_t>(problem.K);
    std::vector<__half> input(input_count, __float2half(1.0F));
    std::vector<__half> weight(weight_count, __float2half(1.0F));
    std::vector<__half> output(output_count, __float2half(0.0F));
    void* input_device = nullptr;
    void* weight_device = nullptr;
    void* output_device = nullptr;
    if(hipMalloc(&input_device, input.size() * sizeof(__half)) != hipSuccess ||
       hipMalloc(&weight_device, weight.size() * sizeof(__half)) != hipSuccess ||
       hipMalloc(&output_device, output.size() * sizeof(__half)) != hipSuccess)
        return false;
    const auto cleanup = [&] {
        (void)hipFree(input_device);
        (void)hipFree(weight_device);
        (void)hipFree(output_device);
    };
    if(hipMemcpyAsync(input_device, input.data(), input.size() * sizeof(__half),
                      hipMemcpyHostToDevice, stream) != hipSuccess ||
       hipMemcpyAsync(weight_device, weight.data(), weight.size() * sizeof(__half),
                      hipMemcpyHostToDevice, stream) != hipSuccess)
    {
        cleanup();
        return false;
    }

    ck_tile::dispatcher::GroupedConvDispatcher dispatcher(&registry);
    if(dispatcher.select_kernel(problem) == nullptr)
    {
        cleanup();
        return false;
    }
    try
    {
        dispatcher.run(input_device, weight_device, output_device, problem, device,
                       reinterpret_cast<void*>(stream));
    }
    catch(const std::exception& error)
    {
        std::cerr << "dispatcher execution failed: " << error.what() << "\n";
        cleanup();
        return false;
    }
    hipEvent_t complete = nullptr;
    if(hipEventCreate(&complete) != hipSuccess || hipEventRecord(complete, stream) != hipSuccess ||
       hipEventSynchronize(complete) != hipSuccess ||
       hipMemcpy(output.data(), output_device, output.size() * sizeof(__half),
                 hipMemcpyDeviceToHost) != hipSuccess)
    {
        if(complete != nullptr) (void)hipEventDestroy(complete);
        cleanup();
        return false;
    }
    (void)hipEventDestroy(complete);
    cleanup();
    const float expected = static_cast<float>(problem.C / problem.G);
    for(const auto value : output)
        if(std::abs(__half2float(value) - expected) > 0.5F) return false;
    return true;
}

} // namespace
int main()
{
    int device_count = 0;
    if(hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) return skip_code;
    int device = 0;
    if(hipGetDevice(&device) != hipSuccess) return skip_code;
    hipDeviceProp_t properties{};
    if(hipGetDeviceProperties(&properties, device) != hipSuccess ||
       normalize_arch(properties.gcnArchName) != "gfx1100") return skip_code;

    ck_tile::dispatcher::GroupedConvRegistry registry;
    ck_tile::dispatcher::register_all_grouped_conv_fwd_kernels(registry, "gfx1100");
    if(!check_catalog(registry))
    {
        std::cerr << "generated gfx1100 FP16 catalog is empty or has invalid metadata\n";
        return 1;
    }

    hipStream_t stream = nullptr;
    if(hipStreamCreate(&stream) != hipSuccess) return 1;
    const auto mismatch_problem = make_problem(2, 1);
    ck_tile::dispatcher::GroupedConvDispatcher dispatcher(&registry);
    if(dispatcher.select_kernel(mismatch_problem) == nullptr)
    {
        (void)hipStreamDestroy(stream);
        std::cerr << "no generated kernel supports the 2D smoke problem\n";
        return 1;
    }
    bool rejected_mismatch = false;
    try
    {
        dispatcher.run(nullptr, nullptr, nullptr, mismatch_problem, device,
                       reinterpret_cast<void*>(stream), [](int) { return "gfx942"; });
    }
    catch(const std::runtime_error&)
    {
        rejected_mismatch = true;
    }
    const bool passed = rejected_mismatch && run_case(registry, device, stream, 2, 1) &&
                        run_case(registry, device, stream, 3, 2);
    (void)hipStreamDestroy(stream);
    if(!passed)
    {
        std::cerr << "gfx1100 grouped convolution production-path gate failed\n";
        return 1;
    }
    return 0;
}

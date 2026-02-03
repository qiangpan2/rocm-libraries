/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <vector>
#include <cstdint>

#include <miopen/conv/solvers.hpp>
#include <miopen/env.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
// Enable DL kernels - this is required to access DL instances
#define DL_KERNELS ON
#include <miopen/solver/ck_utility_common.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp>
#endif

#include <miopen/solver/implicitgemm_ck_util.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_2D_CONV_IMPLICIT_GEMM_HIP_GROUPED_FWD_DLOPS)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

// Device operation type definition - FP32 only
// Note: Use full type paths to avoid conflict with implicitgemm_ck_util.hpp (which defines 3D layouts)
using DeviceOpDLFwdF32 = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
    2,
    ck::tensor_layout::convolution::NHWGC,
    ck::tensor_layout::convolution::GKYXC,
    ck::Tuple<>,
    ck::tensor_layout::convolution::NHWGK,
    float,
    float,
    ck::Tuple<>,
    float,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>;

// Instance Factory - FP32 only
using DeviceOpDLFwdF32Ptrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceOpDLFwdF32>;

// Force linker to include DL instances from CK library
// Without this, the linker strips unused symbols due to --exclude-libs,ALL
namespace {
// Use volatile to prevent compiler from optimizing away the static initialization
static volatile bool force_link_dl_instances = []() {
    std::vector<std::unique_ptr<DeviceOpDLFwdF32>> dummy;
    ck::tensor_operation::device::instance::add_device_grouped_conv2d_fwd_dl_nhwgc_gkyxc_nhwgk_f32_instances(dummy);
    return true;
}();
// Also create a function pointer reference to ensure symbol is not stripped
static auto* volatile force_link_func_ptr = 
    &ck::tensor_operation::device::instance::add_device_grouped_conv2d_fwd_dl_nhwgc_gkyxc_nhwgk_f32_instances;
} // namespace force_link

namespace {

// CK Arguments structure for DL FP32 forward convolution
struct CKArgsDL
{
    CKArgsDL(const ProblemDescription& problem)
    {
        G  = ProblemInterpreter::GetGroupCountG(problem);
        N  = ProblemInterpreter::GetBatchN(problem);
        K1 = ProblemInterpreter::GetOutputChannelK(problem);
        C1 = ProblemInterpreter::GetInputChannelC(problem);
        C  = C1 / G; // Number of input Channel per group
        K  = K1 / G; // Number of output Channel per group
        Hi = ProblemInterpreter::GetInputHeightHi(problem);
        Wi = ProblemInterpreter::GetInputWidthWi(problem);
        Ho = ProblemInterpreter::GetOutputHeightHo(problem);
        Wo = ProblemInterpreter::GetOutputWidthWo(problem);
        Y  = ProblemInterpreter::GetFilterHeightY(problem);
        X  = ProblemInterpreter::GetFilterWidthX(problem);

        input  = {G, N, C, Hi, Wi};
        output = {G, N, K, Ho, Wo};
        weight = {G, K, C, Y, X};

        // strides from NHWGC to GNCHW layout
        in_strides  = {C, Hi * Wi * G * C, 1, Wi * G * C, G * C};
        out_strides = {K, Ho * Wo * G * K, 1, Wo * G * K, G * K};
        wei_strides = {K * Y * X * C, Y * X * C, 1, X * C, C};
        strides     = {ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                       ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        dilation    = {ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                       ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding    = {ProblemInterpreter::GetInputLeftPadH(problem),
                       ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding    = {ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                       ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
    }

    CKArgsDL(const CKArgsDL&) = default;
    CKArgsDL(CKArgsDL&&)      = default;
    CKArgsDL& operator=(const CKArgsDL&) = default;

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr,
                    ConstData_t in,
                    ConstData_t w,
                    Data_t out,
                    float alpha,
                    float beta) const
    {
        (void)alpha;
        (void)beta;
        return conv_ptr->MakeArgumentPointer(in,
                                             w,
                                             {},
                                             out,
                                             input,
                                             in_strides,
                                             weight,
                                             wei_strides,
                                             {},
                                             {},
                                             output,
                                             out_strides,
                                             strides,
                                             dilation,
                                             lPadding,
                                             rPadding,
                                             {},
                                             {},
                                             {});
    }

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr,
                    const ConvDataTensors& tensors,
                    float alpha,
                    float beta) const
    {
        return MakeArgPtr(conv_ptr, tensors.in, tensors.w, tensors.out, alpha, beta);
    }

    template <typename ConvPtr>
    bool IsSupportedBy(const ConvPtr& conv_ptr) const
    {
        auto arg_ptr = MakeArgPtr(conv_ptr, nullptr, nullptr, nullptr, 1.0f, 0.0f);
        return conv_ptr->IsSupportedArgument(arg_ptr.get());
    }

    int G;
    int N;
    int K1;
    int C1;
    int K;
    int C;
    int Hi;
    int Wi;
    int Ho;
    int Wo;
    int Y;
    int X;
    std::array<ck::index_t, 5> input;
    std::array<ck::index_t, 5> in_strides;
    std::array<ck::index_t, 5> output;
    std::array<ck::index_t, 5> out_strides;
    std::array<ck::index_t, 5> weight;
    std::array<ck::index_t, 5> wei_strides;
    std::array<ck::index_t, 2> strides;
    std::array<ck::index_t, 2> dilation;
    std::array<ck::index_t, 2> lPadding;
    std::array<ck::index_t, 2> rPadding;
};

} // namespace

// Initialize valid kernel IDs for FP32
void PerformanceConfigHipImplicitGemm2DGroupedFwdDlops::Init(const ProblemDescription& problem)
{
    if(valid_kernels.empty())
        valid_kernels = FillValidKernelsIDs<DeviceOpDLFwdF32Ptrs, CKArgsDL>(problem);
    index     = 0;
    kernel_id = valid_kernels.empty() ? "" : valid_kernels[index];
}

// Check if kernel is supported by CK args
bool PerformanceConfigHipImplicitGemm2DGroupedFwdDlops::CheckIsSupportCKArgs(
    const ProblemDescription& problem) const
{
    return IsCKArgsSupported<DeviceOpDLFwdF32Ptrs, CKArgsDL>(problem, kernel_id);
}

// Check if CK DL instances are applicable
bool ConvHipImplicitGemm2DGroupedFwdDlops::CheckCKApplicability(
    const ProblemDescription& problem) const
{
    return IsCKApplicable<DeviceOpDLFwdF32Ptrs, CKArgsDL>(problem);
}

#endif // MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

void PerformanceConfigHipImplicitGemm2DGroupedFwdDlops::HeuristicInit(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem)
{
    index     = 0;
    kernel_id = "";

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    if(problem.IsFp32())
    {
        Init(problem);
    }
#endif
}

bool PerformanceConfigHipImplicitGemm2DGroupedFwdDlops::SetNextValue(
    const ProblemDescription& problem)
{
#if MIOPEN_USE_COMPOSABLEKERNEL
    if(valid_kernels.empty())
    {
        if(problem.IsFp32())
        {
            Init(problem);
        }
        if(valid_kernels.empty())
            return false;
        return true;
    }
    if((index + 1) < valid_kernels.size())
    {
        ++index;
        kernel_id = valid_kernels[index];
        return true;
    }
    else
#endif
        return false;
}

bool PerformanceConfigHipImplicitGemm2DGroupedFwdDlops::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigHipImplicitGemm2DGroupedFwdDlops::IsValid(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    if(problem.IsFp32())
    {
        return CheckIsSupportCKArgs(problem);
    }
#endif
    return false;
}

bool PerformanceConfigHipImplicitGemm2DGroupedFwdDlops::operator==(
    const PerformanceConfigHipImplicitGemm2DGroupedFwdDlops& other) const
{
    return this->kernel_id == other.kernel_id;
}

PerformanceConfigHipImplicitGemm2DGroupedFwdDlops
ConvHipImplicitGemm2DGroupedFwdDlops::GetDefaultPerformanceConfig(
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    PerformanceConfigHipImplicitGemm2DGroupedFwdDlops pp;
    pp.HeuristicInit(ctx, problem);
    return pp;
}

bool ConvHipImplicitGemm2DGroupedFwdDlops::IsValidPerformanceConfig(
    const ExecutionContext& ctx,
    const ProblemDescription& problem,
    const PerformanceConfigHipImplicitGemm2DGroupedFwdDlops& config) const
{
    return config.IsValid(ctx, problem);
}

size_t ConvHipImplicitGemm2DGroupedFwdDlops::GetWorkspaceSize(
    const ExecutionContext&, const ProblemDescription&) const
{
    // DL kernels do not require workspace
    return 0;
}

PerformanceConfigHipImplicitGemm2DGroupedFwdDlops
ConvHipImplicitGemm2DGroupedFwdDlops::Search(const ExecutionContext& ctx,
                                             const ProblemDescription& problem,
                                             const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

bool ConvHipImplicitGemm2DGroupedFwdDlops::IsApplicable(
    const ExecutionContext&,
    const ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    // Default enabled - set MIOPEN_DEBUG_2D_CONV_IMPLICIT_GEMM_HIP_GROUPED_FWD_DLOPS=0 to disable
    if(env::disabled(MIOPEN_DEBUG_2D_CONV_IMPLICIT_GEMM_HIP_GROUPED_FWD_DLOPS))
        return false;
    if(problem.GetConv().attribute.deterministic)
        return false;
    if(problem.HasNonPackedTensors())
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(problem.IsTensorsCasted())
        return false;
    if(problem.HasMixedDataTypes())
        return false;
    if(!problem.IsDirectionForward())
        return false;
    if(!problem.Is2d())
        return false;
    if(!problem.IsLayoutNHWC())
        return false;
    if(!problem.IsFp32())
        return false;
    // Note: We do NOT check is_ck_whitelist - DL kernels work on all GPUs
    return CheckCKApplicability(problem);
#else
    std::ignore = problem;
    return false;
#endif
}

ConvSolution ConvHipImplicitGemm2DGroupedFwdDlops::GetSolution(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem,
    [[maybe_unused]] const PerformanceConfigHipImplicitGemm2DGroupedFwdDlops& config) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    // FP32 only - use NHWC invoker factory
    return InitInvokerFactoryNHWC<false,
                                  DeviceOpDLFwdF32Ptrs,
                                  CKArgsDL,
                                  miopen::conv::DataInvokeParams>(
        ctx, problem, config.kernel_id);
#else
    return {};
#endif
}

} // namespace conv
} // namespace solver
} // namespace miopen

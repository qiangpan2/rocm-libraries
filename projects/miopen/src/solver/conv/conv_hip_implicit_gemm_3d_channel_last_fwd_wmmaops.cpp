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
#include <miopen/config.h>
#include <miopen/conv/solvers.hpp>
#include <miopen/env.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <miopen/invoker.hpp>
#include <iostream>
#include <functional>

// Include Composable Kernel headers for 3D convolution with channel last layout
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_CKTILE_COMPOSABLEKERNEL
// Include CK tile utility header for 3D convolution
#include <miopen/solver/implicitgemm_ck_tile_util.hpp>
// Include specific CK tile headers if needed beyond what's in the utility
#include <miopen/solver/grouped_convolution_ck_tiles_utils.hpp>
#endif

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_CHANNEL_LAST_FWD_WMMAOPS)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_CKTILE_COMPOSABLEKERNEL

// Use type aliases from the new CK Tile utility header
using namespace miopen::solver::conv_ck_tile;

// Number of spatial dimensions for 3D convolution
static constexpr ck_tile::index_t NumDimSpatial = 3;

template <typename DataType>
using DeviceOp3DChannelLastFwd =
    ck_tile::GroupedConvFwdKernelArgs<
        ck_tile::GroupedConvTraits<NumDimSpatial,
                                   ck_tile::ConvolutionSpecialization::Default,
                                   ck_tile::tensor_layout::convolution::NDHWGC, // Channel Last Input with Group
                                   ck_tile::tensor_layout::convolution::GKZYXC, // Weight
                                   ck_tile::tuple<>,
                                   ck_tile::tensor_layout::convolution::NDHWGK>, // Channel Last Output with Group
        ck_tile::element_wise::PassThrough>; // Add the missing element-wise operation parameter

// Type alias for Host Arguments - use the correct type
using GroupedConvFwdHostArgs = ck_tile::GroupedConvFwdHostArgs<ck_tile::element_wise::PassThrough>;

namespace {

using MemoryOpSet =
    std::integral_constant<ck_tile::memory_operation_enum, ck_tile::memory_operation_enum::set>;
using MemoryOpAtomicAdd = std::integral_constant<ck_tile::memory_operation_enum,
                                                 ck_tile::memory_operation_enum::atomic_add>;

// Base configuration struct following CK tile examples
struct ConvConfigBase
{
    static constexpr ck_tile::index_t VectorSizeA = 4;
    static constexpr ck_tile::index_t VectorSizeB = 8;
    static constexpr ck_tile::index_t VectorSizeC = 8;

    static constexpr int kBlockPerCu                = 1;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
    static constexpr ck_tile::index_t NumWaveGroups = 1;

    static constexpr ck_tile::index_t NumGroupsToMerge = 1;
};

// Pipeline type traits following CK tile examples
template <ck_tile::GemmPipeline PipelineId>
struct PipelineTypeTraits;

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::MEMORY>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrMem<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrMem<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_V3>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_V4>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV4<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV4<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_V5>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV5<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV5<PipelineProblem>;
};

// Structure to hold arguments for the Composable Kernel
template <typename DataType>
struct CKArgs3DChannelLastFwd
{
    CKArgs3DChannelLastFwd(const ProblemDescription& problem)
    {
        // Extract dimensions from the problem description
        G  = ProblemInterpreter::GetGroupCountG(problem);
        N  = ProblemInterpreter::GetBatchN(problem);
        K1 = ProblemInterpreter::GetOutputChannelK(problem);
        C1 = ProblemInterpreter::GetInputChannelC(problem);
        C  = C1 / G; // Number of input Channels per group
        K  = K1 / G; // Number of output Channels per group
        Di = ProblemInterpreter::GetInputDepthDi(problem);
        Hi = ProblemInterpreter::GetInputHeightHi(problem);
        Wi = ProblemInterpreter::GetInputWidthWi(problem);
        Do = ProblemInterpreter::GetOutputDepthDo(problem);
        Ho = ProblemInterpreter::GetOutputHeightHo(problem);
        Wo = ProblemInterpreter::GetOutputWidthWo(problem);
        Z  = ProblemInterpreter::GetFilterDepthZ(problem);
        Y  = ProblemInterpreter::GetFilterHeightY(problem);
        X  = ProblemInterpreter::GetFilterWidthX(problem);

        // Set strides, dilations, and padding
        filter_strides   = {ProblemInterpreter::GetAdjustedConvolutionStrideD(problem),
                            ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                            ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        filter_dilations = {ProblemInterpreter::GetAdjustedConvolutionDilationD(problem),
                            ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                            ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding         = {ProblemInterpreter::GetInputLeftPadD(problem),
                            ProblemInterpreter::GetInputLeftPadH(problem),
                            ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding         = {ProblemInterpreter::GetAdjustedInputRightPadD(problem),
                            ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                            ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
    }

    // Helper function to create ConvParam
    ck_tile::conv::ConvParam MakeConvParam() const
    {
        return ck_tile::conv::ConvParam(
            NumDimSpatial,  // num_dim_spatial
            G,              // group_count
            N,              // n_batch
            K,              // n_out_channels
            C,              // n_in_channels
            {Z, Y, X},      // filters_len
            {Di, Hi, Wi},   // input_len
            filter_strides, // strides
            filter_dilations, // dilations
            lPadding,       // left_pads
            rPadding        // right_pads
        );
    }

    // Function to create Host Arguments for CK tile
    GroupedConvFwdHostArgs MakeHostArgs(const miopen::conv::DataInvokeParams& data_ctx) const
    {
        auto conv_param = MakeConvParam();

        // Get tensor descriptors
        const auto& tensors = data_ctx.tensors;

        // Create Host Arguments
        GroupedConvFwdHostArgs host_args(
            conv_param,
            static_cast<const void*>(tensors.in),  // in_ptr
            static_cast<const void*>(tensors.w),   // wei_ptr
            {},                                    // ds_ptr (empty for now)
            static_cast<void*>(tensors.out),       // out_ptr
            1                                      // k_batch (default to 1)
        );

        return host_args;
    }

    // Function to create Kernel Arguments for CK tile
    // This is the main function that will be called by the invoker
    auto
    MakeArgument(const miopen::conv::DataInvokeParams& data_ctx) const
    {
        // Create Host Arguments
        auto host_args = MakeHostArgs(data_ctx);

        // Return the host args directly for the new kernel invoker
        return host_args;
    }

    // Tensor dimensions
    int G;  // Groups
    int N;  // Batch size
    int K1; // Output channels
    int C1; // Input channels
    int C;  // Input channels per group
    int K;  // Output channels per group
    int Di; // Input depth
    int Hi; // Input height
    int Wi; // Input width
    int Do; // Output depth
    int Ho; // Output height
    int Wo; // Output width
    int Z;  // Filter depth
    int Y;  // Filter height
    int X;  // Filter width

    // Convolution parameters
    std::vector<ck_tile::index_t> filter_strides;
    std::vector<ck_tile::index_t> filter_dilations;
    std::vector<ck_tile::index_t> lPadding;
    std::vector<ck_tile::index_t> rPadding;
};

} // namespace

// Performance configuration methods implementation
void PerformanceConfigConv3DChannelLastFwdWmmaops::HeuristicInit(
    const miopen::conv::ProblemDescription&)
{
    instance_id = 0;
}

bool PerformanceConfigConv3DChannelLastFwdWmmaops::SetNextValue(
    const miopen::conv::ProblemDescription&)
{
    // For simplicity
    return false;
}

bool PerformanceConfigConv3DChannelLastFwdWmmaops::IsValidValue() const
{
    // For simplicity
    return true;
}

bool PerformanceConfigConv3DChannelLastFwdWmmaops::IsValid(
    const miopen::conv::ProblemDescription&) const
{
    // For simplicity,  assume any configuration is valid
    return true;
}

bool PerformanceConfigConv3DChannelLastFwdWmmaops::operator==(
    const PerformanceConfigConv3DChannelLastFwdWmmaops& other) const
{
    return instance_id == other.instance_id;
}


// Check if this solver is applicable for the given problem
bool ConvHipImplicitGemm3DChannelLastFwdWmmaops::IsApplicable(
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    // Check if the solver is enabled by environment variable
    if(env::disabled(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_CHANNEL_LAST_FWD_WMMAOPS))
    {
        return false;
    }

    // Check if HIP backend is used
    if(!ctx.use_hip_kernels)
    {
        return false;
    }

    // Check if the hardware is supported
    if(!ck_tile_utility::is_ck_tile_supported_hardware(ctx.GetStream()))
    {
        return false;
    }

    // Check if it's a 3D convolution
    if(!problem.Is3d())
    {
        return false;
    }

    // Check if it's a forward convolution
    if(!problem.IsDirectionForward())
    {
        return false;
    }

    // Check if it's channel last layout (NHWC for 2D, NDHWC for 3D)
    if(!problem.IsLayoutNHWC())
    {
        return false;
    }

    if(!problem.IsFp16() && !problem.IsBfp16())
    {
        return false;
    }

    // Check if tensors fit into int
    if(!problem.AllTensorsDimsFitIntoInt())
    {
        return false;
    }

    // Check if it's a grouped convolution (including group count of 1)
    // This solver is designed for grouped convolutions
    if(problem.GetGroupCount() < 1)
    {
        return false;
    }

    // Check if tensors are not casted
    if(problem.IsTensorsCasted())
    {
        return false;
    }

    // Note: Vector size alignment will be checked dynamically at runtime
    // We provide multiple configurations with different vector sizes for compatibility
    // The actual configuration selection happens in GetSolution based on C and K values

    return true;
}

// Get the default performance configuration
PerformanceConfigConv3DChannelLastFwdWmmaops
ConvHipImplicitGemm3DChannelLastFwdWmmaops::GetDefaultPerformanceConfig(
    const ExecutionContext&, const ProblemDescription& problem) const
{
    // For now, return a default configuration
    PerformanceConfigConv3DChannelLastFwdWmmaops config;
    config.HeuristicInit(problem);
    return config;
}

// Check if a performance configuration is valid
bool ConvHipImplicitGemm3DChannelLastFwdWmmaops::IsValidPerformanceConfig(
    const ExecutionContext& ctx,
    const ProblemDescription& problem,
    const PerformanceConfigConv3DChannelLastFwdWmmaops& config) const
{
    // For now, we assume any configuration is valid
    // In a real implementation, you would validate the configuration
    return config.IsValid(ctx, problem);
}

// Search for the best performance configuration
PerformanceConfigConv3DChannelLastFwdWmmaops
ConvHipImplicitGemm3DChannelLastFwdWmmaops::Search(
    const ExecutionContext& ctx,
    const ProblemDescription& problem,
    const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

// Configuration struct for 3D Convolution following CK Tile examples
// This config is based on ConvConfigComputeV3_WMMA from the reference examples
template <typename PrecType>
struct MIOPENConvConfig3D : public ConvConfigBase
{
    // Vector sizes for memory operations
    static constexpr ck_tile::index_t VectorSizeA = 4;
    static constexpr ck_tile::index_t VectorSizeB = 8;
    static constexpr ck_tile::index_t VectorSizeC = 8;

    // Tile dimensions optimized for 3D convolution with WMMA
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 64 / sizeof(PrecType);

    // Warp-level dimensions
    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    // Warp tile dimensions
    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    // Pipeline configuration
    static constexpr bool DoubleSmemBuffer = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
    static constexpr auto Scheduler = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr ck_tile::index_t NumGroupsToMerge = 1;
    
    // Block occupancy
    static constexpr int kBlockPerCu = 2;
};

// Alternative configuration with smaller vector sizes for better compatibility
template <typename PrecType>
struct MIOPENConvConfig3D_Small : public ConvConfigBase
{
    // Smaller vector sizes for cases where C/K are not divisible by 8
    static constexpr ck_tile::index_t VectorSizeA = 4;
    static constexpr ck_tile::index_t VectorSizeB = 4;
    static constexpr ck_tile::index_t VectorSizeC = 4;

    // Tile dimensions
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 64 / sizeof(PrecType);

    // Warp-level dimensions
    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    // Warp tile dimensions
    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    // Pipeline configuration
    static constexpr bool DoubleSmemBuffer = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
    static constexpr auto Scheduler = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr ck_tile::index_t NumGroupsToMerge = 1;
    
    // Block occupancy
    static constexpr int kBlockPerCu = 2;
};

// Minimal vector size configuration for maximum compatibility
template <typename PrecType>
struct MIOPENConvConfig3D_Minimal : public ConvConfigBase
{
    // Minimal vector sizes for maximum compatibility
    static constexpr ck_tile::index_t VectorSizeA = 1;
    static constexpr ck_tile::index_t VectorSizeB = 1;
    static constexpr ck_tile::index_t VectorSizeC = 1;

    // Tile dimensions
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 64 / sizeof(PrecType);

    // Warp-level dimensions
    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    // Warp tile dimensions
    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    // Pipeline configuration
    static constexpr bool DoubleSmemBuffer = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
    static constexpr auto Scheduler = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr ck_tile::index_t NumGroupsToMerge = 1;
    
    // Block occupancy
    static constexpr int kBlockPerCu = 2;
};

// Helper function to select appropriate configuration based on problem dimensions
template<typename CKDataType>
auto CreateKernelInvoker(const ProblemDescription& problem, const CKArgs3DChannelLastFwd<CKDataType>& ck_args) {
    // Get dimensions
    const auto G = ProblemInterpreter::GetGroupCountG(problem);
    const auto C1 = ProblemInterpreter::GetInputChannelC(problem);
    const auto K1 = ProblemInterpreter::GetOutputChannelK(problem);
    const auto C = C1 / G;  // Input channels per group
    const auto K = K1 / G;  // Output channels per group

    // Select configuration based on channel divisibility
    // Try to use the largest vector size possible for better performance
    if(C % 8 == 0 && K % 8 == 0)
    {
        // Use default configuration with VectorSize 8
        return CreateKernelInvokerWithConfig<CKDataType, MIOPENConvConfig3D<CKDataType>>(problem, ck_args);
    }
    else if(C % 4 == 0 && K % 4 == 0)
    {
        // Use small configuration with VectorSize 4
        return CreateKernelInvokerWithConfig<CKDataType, MIOPENConvConfig3D_Small<CKDataType>>(problem, ck_args);
    }
    else
    {
        // Use minimal configuration with VectorSize 1 for maximum compatibility
        return CreateKernelInvokerWithConfig<CKDataType, MIOPENConvConfig3D_Minimal<CKDataType>>(problem, ck_args);
    }
}

// Helper function to convert miopen data type to CK tile data type and run the kernel
template<typename CKDataType, typename ConvConfig>
auto CreateKernelInvokerWithConfig(const ProblemDescription& problem, const CKArgs3DChannelLastFwd<CKDataType>& ck_args) {
    return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
        const auto& data_ctx = primitive_params.CastTo<miopen::conv::DataInvokeParams>();

        // Create the host arguments
        auto host_args = ck_args.MakeHostArgs(data_ctx);

        // Define types for the kernel
        using DataType = CKDataType;
        using AccDataType = float;         // FP32 accumulation

        // Create a stream_config object for CK Tile
        ck_tile::stream_config ck_stream_config{handle.GetStream(), handle.IsProfilingEnabled()};

        // Define types matching the example
        using InDataType = DataType;
        using WeiDataType = DataType;
        using OutDataType = DataType;
        using DsDataType = ck_tile::tuple<>;

        // Use 3D spatial dimension
        constexpr ck_tile::index_t NDimSpatial = 3;
        constexpr auto ConvSpec = ck_tile::ConvolutionSpecialization::Default;
        using InLayout = ck_tile::tensor_layout::convolution::NDHWGC;
        using WeiLayout = ck_tile::tensor_layout::convolution::GKZYXC;
        using OutLayout = ck_tile::tensor_layout::convolution::NDHWGK;
        using DsLayout = ck_tile::tuple<>;

        // Implicit GEMM Traits - as per the example
        using GemmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<ConvConfig::M_Tile, ConvConfig::N_Tile, ConvConfig::K_Tile>,
            ck_tile::sequence<ConvConfig::M_Warp, ConvConfig::N_Warp, ConvConfig::K_Warp>,
            ck_tile::sequence<ConvConfig::M_Warp_Tile,
                              ConvConfig::N_Warp_Tile,
                              ConvConfig::K_Warp_Tile>>;

        using GroupedConvTraitsType = ck_tile::GroupedConvTraits<NDimSpatial,
                                                                 ConvSpec,
                                                                 InLayout,
                                                                 WeiLayout,
                                                                 DsLayout,
                                                                 OutLayout,
                                                                 ConvConfig::VectorSizeA,
                                                                 ConvConfig::VectorSizeB,
                                                                 ConvConfig::VectorSizeC,
                                                                 ConvConfig::NumGroupsToMerge>;

        using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
            GemmShape,
            GroupedConvTraitsType::FixedGemmParams::TilePartitionerGroupNum,
            GroupedConvTraitsType::FixedGemmParams::TilePartitionerM01>;

        using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<
            GroupedConvTraitsType::FixedGemmParams::kPadM,
            GroupedConvTraitsType::FixedGemmParams::kPadN,
            GroupedConvTraitsType::FixedGemmParams::kPadK,
            ConvConfig::DoubleSmemBuffer,
            typename GroupedConvTraitsType::AsLayoutFwd,
            typename GroupedConvTraitsType::BsLayoutFwd,
            typename GroupedConvTraitsType::CLayoutFwd,
            GroupedConvTraitsType::FixedGemmParams::TransposeC,
            GroupedConvTraitsType::FixedGemmParams::UseStructuredSparsity,
            GroupedConvTraitsType::FixedGemmParams::Persistent,
            ConvConfig::NumWaveGroups>;

        using GemmPipelineProblem = ck_tile::GemmPipelineProblem<
            InDataType,
            WeiDataType,
            AccDataType,
            GemmShape,
            typename GroupedConvTraitsType::template GroupedConvImplicitGemmTraitsFwd<
                ConvConfig::NumWaveGroups>,
            ck_tile::element_wise::PassThrough,
            ck_tile::element_wise::PassThrough,
            OutDataType,
            GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,
            GroupedConvTraitsType::VectorSizeA,
            GroupedConvTraitsType::VectorSizeB>;

        using BaseGemmPipeline = typename PipelineTypeTraits<
            ConvConfig::Pipeline>::template UniversalGemmPipeline<GemmPipelineProblem>;

        const ck_tile::index_t gemm_k =
            host_args.C_ * std::accumulate(host_args.filter_spatial_lengths_.begin(),
                                      host_args.filter_spatial_lengths_.end(),
                                      1,
                                      std::multiplies<ck_tile::index_t>());

        // Split-K parameters
        const ck_tile::index_t k_grain     = host_args.k_batch * ConvConfig::K_Tile;
        const ck_tile::index_t K_split     = (gemm_k + k_grain - 1) / k_grain * ConvConfig::K_Tile;
        const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
        float ave_time{0};

        // Proper pipeline selection based on configuration
        const auto Run =
            [&](const auto has_hot_loop_, const auto tail_number_, const auto memory_operation_) {
                constexpr bool has_hot_loop_v   = has_hot_loop_.value;
                constexpr auto tail_number_v    = tail_number_.value;
                constexpr auto scheduler        = ConvConfig::Scheduler;
                constexpr auto memory_operation = memory_operation_.value;

                using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
                    InDataType,
                    WeiDataType,
                    AccDataType,
                    GemmShape,
                    GemmUniversalTraits,
                    scheduler,
                    has_hot_loop_v,
                    tail_number_v,
                    ck_tile::element_wise::PassThrough,
                    ck_tile::element_wise::PassThrough,
                    OutDataType,
                    GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,
                    GroupedConvTraitsType::VectorSizeA,
                    GroupedConvTraitsType::VectorSizeB>;

                using GemmPipeline = typename PipelineTypeTraits<
                    ConvConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;

                using ConvEpilogue = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
                    InDataType,
                    WeiDataType,
                    DsDataType,
                    AccDataType,
                    OutDataType,
                    typename GroupedConvTraitsType::ImplicitGemmDsLayout,
                    typename GroupedConvTraitsType::FixedGemmParams::ELayout,
                    ck_tile::element_wise::PassThrough,
                    TilePartitioner::MPerBlock,
                    TilePartitioner::NPerBlock,
                    ConvConfig::M_Warp,
                    ConvConfig::N_Warp,
                    ConvConfig::M_Warp_Tile,
                    ConvConfig::N_Warp_Tile,
                    ConvConfig::K_Warp_Tile,
                    GroupedConvTraitsType::FixedGemmParams::TransposeC,
                    memory_operation,
                    ConvConfig::NumWaveGroups,
                    GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,
                    ConvConfig::VectorSizeC>>;

                using Kernel = ck_tile::GroupedConvolutionForwardKernel<GroupedConvTraitsType,
                                                                        TilePartitioner,
                                                                        GemmPipeline,
                                                                        ConvEpilogue>;
                auto kargs   = Kernel::MakeKernelArgs(host_args);

                const dim3 grids  = Kernel::GridSize(kargs);
                const dim3 blocks = Kernel::BlockSize();

                if(!Kernel::IsSupportedArgument(kargs))
                {
                    throw std::runtime_error("Wrong! Arguments not supported! Skipping conv!\n");
                }

                if(ck_stream_config.log_level_ > 0)
                {
                    std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
                              << "shape: " << GemmShape::GetName() << '\n'
                              << "problem: " << UniversalGemmProblem::GetName() << '\n'
                              << "pipeline: " << GemmPipeline::GetName() << '\n'
                              << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                              << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z
                              << "}" << '\n'
                              << "Vector size A: " << GemmPipeline::GetVectorSizeA()
                              << ", Vector size B: " << GemmPipeline::GetVectorSizeB()
                              << ", Vector size C: " << ConvEpilogue::GetVectorSizeC() << std::endl;
                }

                ave_time = ck_tile::launch_kernel(ck_stream_config,
                                                  ck_tile::make_kernel<ConvConfig::kBlockPerCu>(
                                                      Kernel{}, grids, blocks, 0, kargs));
                return ave_time;
            };

        // Split-K lambda with proper memory operation selection
        const auto RunSplitk = [&](const auto has_hot_loop_, const auto tail_number_) {
            if(host_args.k_batch == 1)
            {
                Run.template operator()(has_hot_loop_, tail_number_, MemoryOpSet{});
            }
            else
            {
                Run.template operator()(has_hot_loop_, tail_number_, MemoryOpAtomicAdd{});
            }
        };

        // Execute the kernel using TailHandler pattern from CK examples
        BaseGemmPipeline::TailHandler(RunSplitk, has_hot_loop, tail_num);

        if(handle.IsProfilingEnabled())
        {
            handle.ResetKernelTime();
            handle.AccumKernelTime(ave_time);
        }
    };
}

// Get the solution for the given problem and configuration
ConvSolution ConvHipImplicitGemm3DChannelLastFwdWmmaops::GetSolution(
    const ExecutionContext& ctx,
    const ProblemDescription& problem,
    const PerformanceConfigConv3DChannelLastFwdWmmaops& config) const
{
    ConvSolution sol;

    // Set up the invoker factory
    sol.invoker_factory = [=](const std::vector<Kernel>& kernels) -> Invoker {
        // Create CK arguments with the appropriate data type based on problem
        if(problem.IsFp16()) {
            CKArgs3DChannelLastFwd<ck_tile::half_t> ck_args(problem);
            return CreateKernelInvoker<ck_tile::half_t>(problem, ck_args);
        }
        else if(problem.IsBfp16()) {
            CKArgs3DChannelLastFwd<ck_tile::bf16_t> ck_args(problem);
            return CreateKernelInvoker<ck_tile::bf16_t>(problem, ck_args);
        }
        else {
            // This should not happen if IsApplicable was called first, but just in case
            throw std::runtime_error("Unsupported data type for 3D Channel Last Forward Convolution");
        }
    };

    return sol;
}

// Get the workspace size required for this solver
size_t ConvHipImplicitGemm3DChannelLastFwdWmmaops::GetWorkspaceSize(
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    // For now, return 0 as we're not using workspace
    return 0;
}

} // namespace conv
} // namespace solver
} // namespace miopen

#endif // MIOPEN_BACKEND_HIP && MIOPEN_USE_CKTILE_COMPOSABLEKERNEL

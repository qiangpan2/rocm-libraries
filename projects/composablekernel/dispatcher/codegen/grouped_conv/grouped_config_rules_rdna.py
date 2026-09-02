#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Small deterministic production catalog for wave32 RDNA grouped convolution.

Applies to any architecture whose arch spec declares wave32 plus the 16x16x16
FP16 WMMA warp tile, so the supported arch list lives in arch_specs, not here.
"""

from typing import List

from grouped_conv.grouped_config_rules_default import get_warp_size

FP16_DTYPE_KEY = "fp16_fp16_fp32"
WMMA_WARP_TILE = [16, 16, 16]

TILE_K = 32
# 4/8/8 drives the CompV3 buffer-load instruction count to zero for these
# wave shapes; 2/2/8 keeps tile_m*tile_k/(block_size*vector_a) >= 1.
VECTOR_SIZES = (2, 2, 8)


def get_configs(arch: str, variants: List, ndims: List[int], datatypes: List[str]) -> List:
    """Return the Phase-0 FP16 forward catalog in stable order."""
    from arch_specs_generated import get_warp_configs, get_warp_tile_combos
    from unified_grouped_conv_codegen import (
        GroupedConvKernelConfig,
        GroupedConvTraitConfig,
        GroupedConvVariant,
        TileConfig,
    )

    if "fp16" not in datatypes or GroupedConvVariant.FORWARD not in variants:
        return []
    if get_warp_size(arch) != 32:
        return []
    if WMMA_WARP_TILE not in get_warp_tile_combos(arch, FP16_DTYPE_KEY):
        return []

    waves = get_warp_configs(arch)
    warp_tile_m, warp_tile_n, warp_tile_k = WMMA_WARP_TILE

    configs = []
    for ndim in sorted(set(ndims)):
        if ndim not in (2, 3):
            continue
        for wave_m, wave_n, wave_k in waves:
            configs.append(
                GroupedConvKernelConfig(
                    tile=TileConfig(
                        tile_m=wave_m * warp_tile_m,
                        tile_n=wave_n * warp_tile_n,
                        tile_k=TILE_K,
                        warp_m=wave_m,
                        warp_n=wave_n,
                        warp_k=wave_k,
                        warp_tile_m=warp_tile_m,
                        warp_tile_n=warp_tile_n,
                        warp_tile_k=warp_tile_k,
                    ),
                    trait=GroupedConvTraitConfig(
                        pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
                        pad_m=True, pad_n=True, pad_k=True,
                    ),
                    variant=GroupedConvVariant.FORWARD,
                    ndim_spatial=ndim,
                    arch=arch,
                    vector_sizes=VECTOR_SIZES,
                    datatype="fp16",
                )
            )
    return configs

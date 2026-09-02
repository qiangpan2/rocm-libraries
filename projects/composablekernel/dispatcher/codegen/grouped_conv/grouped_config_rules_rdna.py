#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Small deterministic production catalog for wave32 RDNA grouped convolution.

Applies to any architecture whose arch spec declares wave32 plus the 16x16x16
WMMA warp tile for a datatype in WMMA_DATATYPES, so both the supported arch list
and the per-arch datatype list live in arch_specs, not here.
"""

from typing import List

from grouped_conv.grouped_config_rules_default import get_warp_size

WMMA_DATATYPES = ("fp16", "bf16")
WMMA_WARP_TILE = [16, 16, 16]

TILE_K = 32
# 4/8/8 drives the CompV3 buffer-load instruction count to zero for these
# wave shapes; 2/2/8 keeps tile_m*tile_k/(block_size*vector_a) >= 1.
VECTOR_SIZES = (2, 2, 8)


def supported_datatypes(arch: str) -> List[str]:
    """Datatypes this rule set can generate for arch; empty means unsupported."""
    from arch_specs_generated import get_warp_tile_combos

    if get_warp_size(arch) != 32:
        return []
    return [
        dtype for dtype in WMMA_DATATYPES
        if WMMA_WARP_TILE in get_warp_tile_combos(arch, f"{dtype}_{dtype}_fp32")
    ]


def get_configs(arch: str, variants: List, ndims: List[int], datatypes: List[str]) -> List:
    """Return the forward catalog for arch in stable order."""
    from arch_specs_generated import get_warp_configs
    from unified_grouped_conv_codegen import (
        GroupedConvKernelConfig,
        GroupedConvTraitConfig,
        GroupedConvVariant,
        TileConfig,
    )

    if GroupedConvVariant.FORWARD not in variants:
        return []

    selected = [dtype for dtype in supported_datatypes(arch) if dtype in datatypes]
    if not selected:
        return []

    waves = get_warp_configs(arch)
    warp_tile_m, warp_tile_n, warp_tile_k = WMMA_WARP_TILE

    configs = []
    for dtype in selected:
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
                    datatype=dtype,
                )
            )
    return configs

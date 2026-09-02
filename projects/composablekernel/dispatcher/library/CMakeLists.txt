# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Dispatcher instance OBJECT libraries for CK Tile grouped convolution.
#
# Defines ck_add_dispatcher_conv_instances() and creates targets:
#   ck_dispatcher_grouped_conv_fwd
#   ck_dispatcher_grouped_conv_bwd_weight
#   ck_dispatcher_grouped_conv_bwd_data
#
# Each target is an OBJECT library compiled from codegen-generated kernel sources.
# Include via add_subdirectory() from a parent that has already called find_package(Python3).

# ---- Configuration (inherits from parent scope) ----

set(DISPATCHER_DIR "${PROJECT_SOURCE_DIR}/dispatcher")
set(DISPATCHER_RULE_SET "tests" CACHE STRING "Dispatcher rule set")

# An RDNA target anywhere in GPU_TARGETS selects the production catalog.
include(${DISPATCHER_DIR}/rdna_targets.cmake)
string(REPLACE ";" " " _GPU_TARGETS_SPACE "${GPU_TARGETS}")
string(REPLACE " " ";" _GPU_TARGETS_LIST "${_GPU_TARGETS_SPACE}")
list(FILTER _GPU_TARGETS_LIST EXCLUDE REGEX "^$")
ck_dispatcher_rdna_target("${_GPU_TARGETS_LIST}" _RDNA_TARGET)
if(_RDNA_TARGET)
  set(_GPU_TARGET "${_RDNA_TARGET}")
  set(_DISPATCHER_VARIANTS fwd)
  set(_DISPATCHER_EFFECTIVE_RULE_SET rdna)
  # No --datatype: the generator derives the set from arch_specs for this target.
  set(_DISPATCHER_EXTRA_ARGS)
else()
  list(GET _GPU_TARGETS_LIST 0 _GPU_TARGET)
  set(_DISPATCHER_VARIANTS fwd bwd_weight bwd_data)
  set(_DISPATCHER_EFFECTIVE_RULE_SET ${DISPATCHER_RULE_SET})
  set(_DISPATCHER_EXTRA_ARGS)
endif()

# ---- Variant-to-path mapping tables ----

set(_DISP_GEN_SCRIPT "generate_profiler_kernels.py")

set(_DISP_HEADER_PREFIX_fwd "grouped_conv_fwd_")
set(_DISP_HEADER_PREFIX_bwd_weight "grouped_conv_bwd_weight_")
set(_DISP_HEADER_PREFIX_bwd_data "grouped_conv_bwd_data_")


# =============================================================================
# ck_add_dispatcher_conv_instances(<variant>)
#
# Creates an OBJECT library target: ck_dispatcher_grouped_conv_<variant>
# and a convenience target: dispatcher_<variant>_lib
#
# <variant> must be one of: fwd, bwd_weight, bwd_data
# =============================================================================
function(ck_add_dispatcher_conv_instances VARIANT)
  # Look up variant-specific paths
  set(HEADER_PREFIX "${_DISP_HEADER_PREFIX_${VARIANT}}")

  if(NOT HEADER_PREFIX)
    message(FATAL_ERROR "Unknown dispatcher variant: ${VARIANT}")
  endif()

  set(SCRIPT_PATH "${DISPATCHER_DIR}/scripts/${_DISP_GEN_SCRIPT}")
  set(KERNEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/dispatcher_${VARIANT}_kernels")
  set(TARGET_NAME "ck_dispatcher_grouped_conv_${VARIANT}")

  # --- Step 1: Run Python codegen at configure time ---
  # Always regenerate at configure time. The driver stages output and atomically
  # replaces KERNEL_DIR only after the complete catalog succeeds.
  execute_process(
    COMMAND ${Python3_EXECUTABLE} ${SCRIPT_PATH}
            --variant ${VARIANT}
            --output-dir ${KERNEL_DIR}
            --arch ${_GPU_TARGET}
            --rule-set ${_DISPATCHER_EFFECTIVE_RULE_SET}
            ${_DISPATCHER_EXTRA_ARGS}
    RESULT_VARIABLE ret
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
    WORKING_DIRECTORY ${DISPATCHER_DIR}/codegen
  )
  if(NOT ret EQUAL 0)
    message(FATAL_ERROR "Dispatcher ${VARIANT} kernel generation failed.\nReturn: ${ret}\nOutput: ${output}\nError: ${error}")
  endif()
  message(STATUS "Dispatcher ${VARIANT} kernels generated (rule set: ${_DISPATCHER_EFFECTIVE_RULE_SET}): ${output}")

  # --- Step 2: Create .cpp wrappers for each generated kernel header ---
  file(GLOB _KERNEL_HEADERS "${KERNEL_DIR}/${HEADER_PREFIX}*.hpp")
  set(_KERNEL_SOURCES "")
  foreach(HEADER ${_KERNEL_HEADERS})
    get_filename_component(STEM ${HEADER} NAME_WE)
    set(WRAPPER_CPP "${KERNEL_DIR}/${STEM}.cpp")
    if(NOT EXISTS ${WRAPPER_CPP})
      file(WRITE ${WRAPPER_CPP} "// Auto-generated wrapper\n#include \"${STEM}.hpp\"\n")
    endif()
    list(APPEND _KERNEL_SOURCES ${WRAPPER_CPP})
  endforeach()

  if(NOT _KERNEL_SOURCES)
    message(WARNING "Dispatcher ${VARIANT}: no kernel sources generated, skipping target creation")
    return()
  endif()

  # --- Step 3: Add chunked registration .cpp files ---
  file(GLOB _REG_CPPS "${KERNEL_DIR}/register_*.cpp")
  foreach(REG_CPP ${_REG_CPPS})
    list(APPEND _KERNEL_SOURCES "${REG_CPP}")
    set_source_files_properties("${REG_CPP}" PROPERTIES
      COMPILE_OPTIONS "-Wno-ctad-maybe-unsupported;-Wno-old-style-cast"
    )
  endforeach()

  # --- Step 4: Create OBJECT library ---
  add_library(${TARGET_NAME} OBJECT ${_KERNEL_SOURCES})
  target_include_directories(${TARGET_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/dispatcher/include
    ${KERNEL_DIR}
  )
  target_compile_options(${TARGET_NAME} PRIVATE
    -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=0
    -DCK_TILE_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK=1
    -Wno-undefined-func-template
    -Wno-float-equal
    -Wno-header-hygiene
    -Wno-unused-parameter
    -Wno-missing-variable-declarations
    # New clang -Weverything suggestion firing on builder-pattern setters that
    # return *this in the shared dispatcher headers; not a real defect.
    -Wno-lifetime-safety-intra-tu-suggestions
  )
  if(NOT WIN32 AND ${hip_VERSION_FLAT} GREATER 600241132)
    target_compile_options(${TARGET_NAME} PRIVATE --offload-compress)
  endif()
  set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  if(_RDNA_TARGET)
    set_target_properties(${TARGET_NAME} PROPERTIES HIP_ARCHITECTURES ${_RDNA_TARGET})
  endif()

  # Progress monitoring
  list(LENGTH _KERNEL_SOURCES _TOTAL_COUNT)
  math(EXPR _KERNEL_COUNT "${_TOTAL_COUNT} - 1")
  message(STATUS "Dispatcher ${VARIANT}: ${_TOTAL_COUNT} compilation units (${_KERNEL_COUNT} kernels + registration)")

  # Convenience ninja target: ninja dispatcher_<variant>_lib
  add_custom_target(dispatcher_${VARIANT}_lib
    DEPENDS ${TARGET_NAME}
    COMMENT "All ${_TOTAL_COUNT} dispatcher ${VARIANT} instances compiled."
  )
endfunction()


# =============================================================================
# Create the dispatcher instance libraries selected for this architecture
# =============================================================================
foreach(_VARIANT IN LISTS _DISPATCHER_VARIANTS)
  ck_add_dispatcher_conv_instances(${_VARIANT})
endforeach()

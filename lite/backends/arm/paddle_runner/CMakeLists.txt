# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.20)
project(arm_paddle_runner)
option(SEMIHOSTING "Enable semihosting" OFF)

set(TARGET_BOARD "corstone-300" CACHE STRING "Target board")

# Example ExecuTorch demo for bare metal Cortex-M based systems

# Dependencies from the Ethos-U Core This is the platform target of
# Corstone-300, that includes ethosu_core_driver and bare-metal bringup
# libraries. We link against ethosu_target_init which includes all of these
# dependencies.

if(MODEL STREQUAL "pplcnetv2")
	set(MODEL_FLAG 1)
elseif(MODEL STREQUAL "ppocr_det")
	set(MODEL_FLAG 2)
elseif(MODEL STREQUAL "picodetv2")
	set(MODEL_FLAG 3)
elseif(MODEL STREQUAL "humansegv2")
	set(MODEL_FLAG 4)
elseif(MODEL STREQUAL "ppocr_layout")
	set(MODEL_FLAG 5)
elseif(MODEL STREQUAL "ppocr_rec")
	set(MODEL_FLAG 6)
elseif(MODEL STREQUAL "tinypose")
	set(MODEL_FLAG 7)
else()
	set(MODEL_FLAG 0)
endif()

if(TARGET_BOARD STREQUAL "corstone-300")
  message(STATUS "corstone-300 ${ETHOS_SDK_PATH}\n")
  add_subdirectory(${ETHOS_SDK_PATH}/core_platform/targets/corstone-300 target)
  target_compile_definitions(ethosu_target_common INTERFACE
      # ETHOSU_MODEL=0 place pte file/data in SRAM area
      # ETHOSU_MODEL=1 place pte file/data in DDR area
      ETHOSU_MODEL=1
      # Configure NPU architecture timing adapters
      # Ethos_U55_High_End_Embedded
      # This is just example numbers and you should make this match your hardware
      # SRAM
      ETHOSU_TA_MAXR_0=8
      ETHOSU_TA_MAXW_0=8
      ETHOSU_TA_MAXRW_0=0
      ETHOSU_TA_RLATENCY_0=32
      ETHOSU_TA_WLATENCY_0=32
      ETHOSU_TA_PULSE_ON_0=3999
      ETHOSU_TA_PULSE_OFF_0=1
      ETHOSU_TA_BWCAP_0=4000
      ETHOSU_TA_PERFCTRL_0=0
      ETHOSU_TA_PERFCNT_0=0
      ETHOSU_TA_MODE_0=1
      ETHOSU_TA_HISTBIN_0=0
      ETHOSU_TA_HISTCNT_0=0
      # Flash
      ETHOSU_TA_MAXR_1=2
      ETHOSU_TA_MAXW_1=0
      ETHOSU_TA_MAXRW_1=0
      ETHOSU_TA_RLATENCY_1=64
      ETHOSU_TA_WLATENCY_1=0
      ETHOSU_TA_PULSE_ON_1=320
      ETHOSU_TA_PULSE_OFF_1=80
      ETHOSU_TA_BWCAP_1=50
      ETHOSU_TA_PERFCTRL_1=0
      ETHOSU_TA_PERFCNT_1=0
      ETHOSU_TA_MODE_1=1
      ETHOSU_TA_HISTBIN_1=0
      ETHOSU_TA_HISTCNT_1=0
      MODEL_FLAG=${MODEL_FLAG}
      )
elseif(TARGET_BOARD STREQUAL "corstone-320")
  message(STATUS "corstone-320 ${ETHOS_SDK_PATH}\n")
  add_subdirectory(${ETHOS_SDK_PATH}/core_platform/targets/corstone-320 target)
  target_compile_definitions(ethosu_target_common INTERFACE
      # ETHOSU_MODEL=0 place pte file/data in SRAM area
      # ETHOSU_MODEL=1 place pte file/data in DDR area
      ETHOSU_MODEL=1
      # Configure NPU architecture timing adapters
      # Ethos_U85_SYS_DRAM_Mid
      # This is just example numbers and you should make this match your hardware
      # SRAM
      ETHOSU_TA_MAXR_0=8
      ETHOSU_TA_MAXW_0=8
      ETHOSU_TA_MAXRW_0=0
      ETHOSU_TA_RLATENCY_0=32
      ETHOSU_TA_WLATENCY_0=32
      ETHOSU_TA_PULSE_ON_0=3999
      ETHOSU_TA_PULSE_OFF_0=1
      ETHOSU_TA_BWCAP_0=4000
      ETHOSU_TA_PERFCTRL_0=0
      ETHOSU_TA_PERFCNT_0=0
      ETHOSU_TA_MODE_0=1
      ETHOSU_TA_HISTBIN_0=0
      ETHOSU_TA_HISTCNT_0=0
      # DRAM
      ETHOSU_TA_MAXR_1=64
      ETHOSU_TA_MAXW_1=32
      ETHOSU_TA_MAXRW_1=0
      ETHOSU_TA_RLATENCY_1=500
      ETHOSU_TA_WLATENCY_1=250
      ETHOSU_TA_PULSE_ON_1=4000
      ETHOSU_TA_PULSE_OFF_1=1000
      ETHOSU_TA_BWCAP_1=3750
      ETHOSU_TA_PERFCTRL_1=0
      ETHOSU_TA_PERFCNT_1=0
      ETHOSU_TA_MODE_1=1
      ETHOSU_TA_HISTBIN_1=0
      ETHOSU_TA_HISTCNT_1=0
      MODEL_FLAG=${MODEL_FLAG}
      )
else()
  message(FATAL_ERROR "Unsupported TARGET_BOARD: ${TARGET_BOARD}")
endif()


# The arm_paddle_runner executable
add_executable(arm_paddle_runner)
target_include_directories(arm_paddle_runner PRIVATE weight)
#link_directories(${PDLITE_BUILD_DIR}/lite/api/)
target_sources(
  arm_paddle_runner PRIVATE arm_paddle_runner.cpp arm_perf_monitor.cpp VelaBinStream.cpp pico_postprocess.cpp ocr_layout_preprocess.cpp
)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# Include the target's bare-metal linker script
ethosu_eval_link_options(arm_paddle_runner)

# Need whole-archive to ensure C++ ctor's are called - this may be wasteful for
# bin size as we link in a number of other symbols
set(LINK_FLAGS "-Wl,--version-script ${PDLITE_ROOT_DIR}/lite/core/lite.map")
set_target_properties(arm_paddle_runner PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
target_link_libraries(
  arm_paddle_runner
  ethosu_target_init
  "-Wl,--whole-archive"
  ${PDLITE_ROOT_DIR}/cmake-out/lite/api/libpaddle_api_light_bundled.a
  "-Wl,--no-whole-archive"
  -Xlinker -Map=arm_paddle_runner.map
)


# ET headers and generated headers includes
target_include_directories(
  arm_paddle_runner PRIVATE  ${CMAKE_CURRENT_BINARY_DIR}
)
message(STATUS "${PDLITE_ROOT_DIR}")
message(STATUS "${ETHOS_SDK_PATH}")
include_directories(${PDLITE_ROOT_DIR}/lite/api)

# Fixup compilation of retarget.c
if(SEMIHOSTING)
  # Remove this when MLBEDSW-8910 is closed.
  set_source_files_properties(
    ${ETHOS_SDK_PATH}/core_platform/targets/${TARGET_BOARD}/retarget.c
    PROPERTIES HEADER_FILE_ONLY TRUE
  )
endif()
message(STATUS "cxx flags ${CMAKE_CXX_FLAGS}")

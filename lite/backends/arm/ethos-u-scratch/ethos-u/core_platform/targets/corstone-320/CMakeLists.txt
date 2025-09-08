#
# SPDX-FileCopyrightText: Copyright 2020-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#############################################################################
# Default parameters
#############################################################################

set(TARGET_CPU "cortex-m85" CACHE INTERNAL "")

if (NOT CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/toolchain/armclang.cmake")
endif()

set(ETHOSU_TARGET_NPU_CONFIG "ethos-u85-256" CACHE STRING "NPU configuration")

set(ETHOSU_PMU_EVENT_0 -1 CACHE STRING "PMU Event #0")
set(ETHOSU_PMU_EVENT_1 -1 CACHE STRING "PMU Event #1")
set(ETHOSU_PMU_EVENT_2 -1 CACHE STRING "PMU Event #2")
set(ETHOSU_PMU_EVENT_3 -1 CACHE STRING "PMU Event #3")
set(ETHOSU_PMU_EVENT_4 -1 CACHE STRING "PMU Event #4")
set(ETHOSU_PMU_EVENT_5 -1 CACHE STRING "PMU Event #5")
set(ETHOSU_PMU_EVENT_6 -1 CACHE STRING "PMU Event #6")
set(ETHOSU_PMU_EVENT_7 -1 CACHE STRING "PMU Event #7")

set(FAST_MEMORY_SIZE 0 CACHE STRING "Size of relocated fast memory scratch tensor")

if (${FAST_MEMORY_SIZE} EQUAL 0)
    set(MEMORY_DEFAULT "sram")
else()
    set(MEMORY_DEFAULT "dram")
endif()

set(MEMORY_MODEL ${MEMORY_DEFAULT} CACHE STRING "Memory config for model")
set(MEMORY_ARENA ${MEMORY_DEFAULT} CACHE STRING "Memory config for arena")

set(UART0_BASE "0x48203000" CACHE INTERNAL "UART base address")
set(SYSTEM_CORE_CLOCK "25000000" CACHE INTERNAL "System core clock (Hz)")

#############################################################################
# Project
#############################################################################

cmake_minimum_required(VERSION 3.21)

project(ethos-u-corstone-320 VERSION 0.0.1)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/helpers.cmake)

#############################################################################
# Corstone-320
#############################################################################

get_filename_component(ETHOSU_TARGET ${CMAKE_CURRENT_SOURCE_DIR} NAME)
message("Configuring target ${ETHOSU_TARGET}")

# Enable CTest
# TODO: Replace command with run_ctest.py script when target is supported.
include(CTest)
ethosu_get_architecture(${ETHOSU_TARGET_NPU_CONFIG})
set(ETHOSU_COMMAND_DEFAULT FVP_Corstone_SSE-320
                            -C mps4_board.subsystem.ethosu.num_macs=${ETHOSU_NUM_MACS}
                            -C mps4_board.visualisation.disable-visualisation=1
                            -C vis_hdlcd.disable_visualisation=1
                            -C mps4_board.telnetterminal0.start_telnet=0
                            -C mps4_board.uart0.out_file="-"
                            -C mps4_board.uart0.unbuffered_output=1
                            -C mps4_board.uart0.shutdown_on_eot=1
    CACHE INTERNAL "Default test command")

# Disable trustzone support in core_software
set(TRUSTZONE_BUILD OFF)

# Include common targets
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../common target)

target_compile_definitions(ethosu_target_common INTERFACE
    # Declare availability of NPU timing adapters
    ETHOSU_HAS_TA

    # Configure PMU events
    ETHOSU_PMU_EVENT_0=${ETHOSU_PMU_EVENT_0}
    ETHOSU_PMU_EVENT_1=${ETHOSU_PMU_EVENT_1}
    ETHOSU_PMU_EVENT_2=${ETHOSU_PMU_EVENT_2}
    ETHOSU_PMU_EVENT_3=${ETHOSU_PMU_EVENT_3}
    ETHOSU_PMU_EVENT_4=${ETHOSU_PMU_EVENT_4}
    ETHOSU_PMU_EVENT_5=${ETHOSU_PMU_EVENT_5}
    ETHOSU_PMU_EVENT_6=${ETHOSU_PMU_EVENT_6}
    ETHOSU_PMU_EVENT_7=${ETHOSU_PMU_EVENT_7}

    # Placement or TLFu model and area. 0 = SRAM, 1 = DRAM
    ETHOSU_FAST_MEMORY_SIZE=${FAST_MEMORY_SIZE}
    ETHOSU_MODEL=$<STREQUAL:${MEMORY_MODEL},dram>
    ETHOSU_ARENA=$<STREQUAL:${MEMORY_ARENA},dram>)

# AXI Timing adaptors
set(registers MAXR MAXW MAXRW RLATENCY WLATENCY PULSE_ON PULSE_OFF BWCAP PERFCTRL PERFCNT MODE HISTBIN HISTCNT)

foreach(register ${registers})
    foreach(index RANGE 0 1)
        set(name ETHOSU_TA_${register}_${index})
        set(${name} -1 CACHE STRING "${name}")

        if (${name} GREATER_EQUAL 0)
            target_compile_definitions(ethosu_target_common INTERFACE ${name}=${${name}})
        endif()
    endforeach()
endforeach()

# Linker script
set(LINK_FILE platform CACHE STRING "Link file")

ethosu_target_link_options(ethosu_target_link INTERFACE
    LINK_FILE ${LINK_FILE}
    ENTRY Reset_Handler)

# Add drivers
target_sources(ethosu_target_startup INTERFACE
    retarget.c
    target.cpp)

target_link_libraries(ethosu_target_startup INTERFACE
    $<$<TARGET_EXISTS:ethosu_core_driver>:ethosu_core_driver;timing_adapter>
    mpu
    ethosu_mhu_dummy
    ethosu_uart_cmsdk_apb)

###############################################################################
# Applications
###############################################################################

# Add all applications
add_subdirectory(../../applications applications)

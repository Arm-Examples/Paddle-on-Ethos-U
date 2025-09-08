#
# SPDX-FileCopyrightText: Copyright 2022-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

if(NOT TARGET freertos_kernel)
    message("Skipping message handler openamp")
    return()
endif()

#############################################################################
# Configuration
#############################################################################

set(MESSAGE_HANDLER_MODEL_0 "" CACHE STRING "Path to built in model 0")
set(MESSAGE_HANDLER_MODEL_1 "" CACHE STRING "Path to built in model 1")
set(MESSAGE_HANDLER_MODEL_2 "" CACHE STRING "Path to built in model 2")
set(MESSAGE_HANDLER_MODEL_3 "" CACHE STRING "Path to built in model 3")

set(MESSAGE_HANDLER_ARENA_SIZE 0x200000 CACHE STRING "Total size of all message handler tensor arenas in bytes (Must be a power of two)")

#############################################################################
# TFLM arena
#############################################################################

# Split total tensor arena equally for each NPU
if(TARGET ethosu_core_driver AND ETHOSU_TARGET_NPU_COUNT GREATER 0)
    set(NUM_ARENAS ${ETHOSU_TARGET_NPU_COUNT})
else()
    set(NUM_ARENAS 1)
endif()

math(EXPR TENSOR_ARENA_SIZE "${MESSAGE_HANDLER_ARENA_SIZE} / ${NUM_ARENAS}")

#############################################################################
# Message handler application
#############################################################################

ethosu_add_executable(message_handler_openamp
    SOURCES
        main.cpp
        core_driver_mutex.cpp
        freertos_allocator.cpp
        inference_runner.cpp
        message_handler.cpp
        remoteproc.cpp
    LIBRARIES
        $<$<TARGET_EXISTS:ethosu_core_driver>:ethosu_core_driver>
        ethosu_log
        ethosu_mailbox
        ethosu_profiler
        freertos_kernel
        inference_process
        openamp-freertos)

target_include_directories(message_handler_openamp PRIVATE
    ${LINUX_DRIVER_STACK_PATH}/kernel/include)

target_compile_definitions(message_handler_openamp PRIVATE
    $<$<BOOL:${ENABLE_REMOTEPROC_TRACE_BUFFER}>:REMOTEPROC_TRACE_BUFFER>
    TENSOR_ARENA_SIZE=${TENSOR_ARENA_SIZE}
    $<$<BOOL:${MESSAGE_HANDLER_MODEL_0}>:MODEL_0=${MESSAGE_HANDLER_MODEL_0}>
    $<$<BOOL:${MESSAGE_HANDLER_MODEL_1}>:MODEL_1=${MESSAGE_HANDLER_MODEL_1}>
    $<$<BOOL:${MESSAGE_HANDLER_MODEL_2}>:MODEL_2=${MESSAGE_HANDLER_MODEL_2}>
    $<$<BOOL:${MESSAGE_HANDLER_MODEL_3}>:MODEL_3=${MESSAGE_HANDLER_MODEL_3}>)

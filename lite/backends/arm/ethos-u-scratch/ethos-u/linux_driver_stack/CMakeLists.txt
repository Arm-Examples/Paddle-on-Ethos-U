#
# SPDX-FileCopyrightText: Copyright 2020-2022, 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

cmake_minimum_required(VERSION 3.0.2)

# Set the project name and version
project("linux_driver_stack" VERSION 1.0)

# Default options
option(BUILD_KERNEL "Build the kernel driver" ON)
option(BUILD_MAILBOX "Build the MHU mailbox kernel module" ON)
option(BUILD_REMOTEPROC "Build the remoteproc kernel module" ON)

# Set variables
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(ARCH "arm64")
else()
    set(ARCH "arm")
endif()

string(REGEX REPLACE "(.*)gcc$" "\\1" CROSS_COMPILE "${CMAKE_C_COMPILER}")

# Add rpath to library directory
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# Add include directory
include_directories("kernel/include")

# Add sub directories
if(BUILD_KERNEL)
    add_subdirectory(kernel)
endif()

if(BUILD_MAILBOX)
    add_subdirectory(mailbox)
endif()

if(BUILD_REMOTEPROC)
    add_subdirectory(remoteproc)
endif()

# Build driver library
add_subdirectory(driver_library)

# Build utils
add_subdirectory(utils)

# Build tests
if(ETHOSU_TARGET_NPU_CONFIG)
    add_subdirectory(tests)
endif()

#
# Print build status
#

message(STATUS "*******************************************************")
message(STATUS "PROJECT_NAME                           : ${PROJECT_NAME}")
message(STATUS "BUILD_KERNEL                           : ${BUILD_KERNEL}")
message(STATUS "BUILD_MAILBOX                          : ${BUILD_MAILBOX}")
message(STATUS "BUILD_REMOTEPROC                       : ${BUILD_REMOTEPROC}")
message(STATUS "KDIR                                   : ${KDIR}")
message(STATUS "*******************************************************")

#
# SPDX-FileCopyrightText: Copyright 2021-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#
# Compile options
#
add_compile_options(-Werror)

# Build layer_by_layer_profiler
add_subdirectory(layer_by_layer_profiler)

# Build arm_profiler
add_subdirectory(arm_profiler)

# Build ethosu_monitor
add_subdirectory(ethosu_monitor)

# Build ethosu_logging
add_subdirectory(ethosu_log)

# Build ethosu_profiler
add_subdirectory(ethosu_profiler)

# Build crc lib
add_subdirectory(crc)

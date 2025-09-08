#
# Copyright (c) 2022 Arm Limited.
#
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

set(CORE_PLATFORM_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../core_platform)

file(GLOB models LIST_DIRECTORIES true "${CORE_PLATFORM_PATH}/applications/baremetal/models/${ETHOSU_TARGET_NPU_CONFIG}/*")

function(add_test_executable testname)
    # Build executable
    foreach(model ${models})
        get_filename_component(modelname ${model} NAME)

        add_executable(lds_${testname}_${modelname} ${testname}.cpp)

        target_include_directories(lds_${testname}_${modelname} PRIVATE
            ${model})

        # Link agains ethosu library
        target_link_libraries(lds_${testname}_${modelname} PRIVATE ethosu)

        # Install target
        install(TARGETS lds_${testname}_${modelname} DESTINATION "bin")
    endforeach()
endfunction()

add_test_executable(run_inference_test)
add_test_executable(cancel_inference_test)

#
# SPDX-FileCopyrightText: Copyright 2021-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

find_package(PythonInterp 3.5 REQUIRED)
if(NOT ${PYTHONINTERP_FOUND})
    message(FATAL_ERROR "Python 3.5 or greater is required to build python driver, but was not found")
endif()

find_package(SWIG 3.0.12 REQUIRED)
if(NOT ${SWIG_FOUND})
    message(FATAL_ERROR "SWIG 3.0.12 or greater is required to build python driver, but was not found")
endif()

set(SETUP_PY            "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
set(SWIG_GENERATE       "${CMAKE_CURRENT_BINARY_DIR}/swig_generate.py")
set(OUT_WRAP            "${CMAKE_CURRENT_BINARY_DIR}/pydriver.wrap.timestamp")

# local env variables passed down to the python scripts
# scripts can thus be used standalone
set(DRIVER_ENV ETHOS_U_DRIVER_INCLUDE="${PROJECT_SOURCE_DIR}/include"
               ETHOS_U_DRIVER_LIB="${PROJECT_BINARY_DIR}")

# common step - generates swig wrappers
add_custom_command(OUTPUT ${OUT_WRAP}
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/README.md ${CMAKE_CURRENT_BINARY_DIR}
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/../../LICENSE-APACHE-2.0.txt ${CMAKE_CURRENT_BINARY_DIR}/LICENSE
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/swig_generate.py ${CMAKE_CURRENT_BINARY_DIR}
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR}/src
                   COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green "Clearing Python build ..."
                   COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} --quiet clean --all
                   COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green "Generating SWIG wrappers ..."
                   COMMAND ${PYTHON_EXECUTABLE} ${SWIG_GENERATE}
                   DEPENDS ethosu)

# source package
if(BUILD_PYTHON_SRC)
    set(OUT_SRC "${CMAKE_CURRENT_BINARY_DIR}/pydriver.src.timestamp")
    add_custom_command(OUTPUT  ${OUT_SRC}
                       COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green "Building Python source package ..."
                       COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} sdist
                       COMMAND ${CMAKE_COMMAND} -E touch ${OUT_SRC}
                       DEPENDS ${OUT_WRAP})
endif()
# wheel package
if(BUILD_PYTHON_WHL)

    find_package(PythonLibs 3.5 REQUIRED)
    if(NOT ${PYTHONLIBS_FOUND})
        message(FATAL_ERROR "Python 3.5 or greater development libraries were not found.")
    endif()

    set(OUT_WHL "${CMAKE_CURRENT_BINARY_DIR}/pydriver.whl.timestamp")
    add_custom_command(OUTPUT  ${OUT_WHL}
                       COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green "Building Python binary package ..."
                       COMMAND ${CMAKE_COMMAND} -E env ${DRIVER_ENV} CXX=${CMAKE_CXX_COMPILER} CXXFLAGS=${CMAKE_CXX_FLAGS} CC=${CMAKE_C_COMPILER} CFLAGS=${CMAKE_C_FLAGS} ${PYTHON_EXECUTABLE} ${SETUP_PY} bdist_wheel
                       COMMAND ${CMAKE_COMMAND} -E touch ${OUT_WHL}
                       DEPENDS ${OUT_WRAP})
endif()
add_custom_target(pydriver ALL DEPENDS ${OUT_WRAP} ${OUT_SRC} ${OUT_WHL})

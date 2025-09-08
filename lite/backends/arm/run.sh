#!/usr/bin/env bash
set -eu
script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
target="ethos-u85-128"
echo "script_dir= ${script_dir}"
root_dir=${script_dir}/ethos-u-scratch
echo "root_dir= ${root_dir}"
output_folder="."
echo "output_folder= ${output_folder}"
mkdir -p ${output_folder}
ethos_u_root_dir="$(cd ${root_dir}/ethos-u && pwd)"
ethos_u_build_dir=${ethos_u_root_dir}/core_platform/build
setup_path_script=${root_dir}/setup_path.sh

pdlite_root_dir=$(cd ${script_dir}/../../.. && pwd)
pdlite_build_dir=${pdlite_root_dir}/cmake-out
echo "pdlite_root_dir= ${pdlite_root_dir}"
echo "pdlite_build_dir= ${pdlite_build_dir}"
paddle_runner_path=${script_dir}/paddle_runner
echo "paddle_runner_path== ${paddle_runner_path}"
fvp_model=FVP_Corstone_SSE-300_Ethos-U55
if [[ ${target} =~ "ethos-u85" ]]
then
    echo "target is ethos-u85 variant so switching to CS320 FVP"
    fvp_model=FVP_Corstone_SSE-320
fi
toolchain_cmake=${script_dir}/ethos-u-setup/arm-none-eabi-gcc.cmake
echo "toolchain_cmake= ${toolchain_cmake}"

[[ -f ${setup_path_script} ]] \
    || { echo "Missing ${setup_path_script}"; exit 1; }
source ${root_dir}/setup_path.sh

# basic checks before we get started
hash ${fvp_model} \
    || { echo "Could not find ${fvp_model} on PATH"; exit 1; }

hash arm-none-eabi-gcc \
    || { echo "Could not find arm baremetal toolchain on PATH"; exit 1; }

[[ -f ${toolchain_cmake} ]] \
    || { echo "Could not find ${toolchain_cmake} file"; exit 1; }

[[ -f ${pdlite_root_dir}/CMakeLists.txt ]] \
    || { echo "paddlelite repo doesn't contain CMakeLists.txt file at root level"; exit 1; }



ARCH=arm64-v8a
TOOLCHAIN=gcc
WITH_EXTRA=OFF
WITH_PYTHON=OFF
WITH_STATIC_LIB=ON
WITH_CV=OFF
WITH_LOG=ON
WITH_EXCEPTION=OFF
WITH_STRIP=OFF
OPTMODEL_DIR=""
WITH_STATIC_MKL=OFF
WITH_AVX=ON
WITH_OPENCL=OFF
WITH_METAL=OFF
SKIP_SUPPORT_0_DIM_TENSOR_PASS=OFF
WITH_ROCKCHIP_NPU=OFF
WITH_NNADAPTER=OFF
NNADAPTER_WITH_ROCKCHIP_NPU=OFF
NNADAPTER_WITH_EEASYTECH_NPU=OFF
NNADAPTER_WITH_IMAGINATION_NNA=OFF
NNADAPTER_WITH_HUAWEI_ASCEND_NPU=OFF
NNADAPTER_WITH_AMLOGIC_NPU=OFF
NNADAPTER_WITH_CAMBRICON_MLU=OFF
NNADAPTER_WITH_VERISILICON_TIMVX=OFF
NNADAPTER_WITH_NVIDIA_TENSORRT=OFF
NNADAPTER_WITH_QUALCOMM_QNN=OFF
NNADAPTER_WITH_KUNLUNXIN_XTCL=OFF
NNADAPTER_WITH_INTEL_OPENVINO=OFF
NNADAPTER_WITH_GOOGLE_XNNPACK=OFF
WITH_KUNLUNXIN_XPU=OFF
WITH_KUNLUNXIN_XFT=OFF
WITH_TRAIN=OFF
WITH_TINY_PUBLISH=ON
BUILD_ARM82_FP16=OFF
WITH_ARM_DOTPROD=ON
WITH_PROFILE=OFF
WITH_PRECISION_PROFILE=OFF
WITH_BENCHMARK=OFF
WITH_ARM_DNN_LIBRARY=OFF
readonly NUM_PROC=${LITE_BUILD_THREADS:-4}

function init_cmake_options {
    if [ "${ARCH}" == "x86" ]; then
        with_x86=ON
        arm_target_os=""
        WITH_TINY_PUBLISH=OFF
    else
        with_arm=ON
        arm_arch=$ARCH
        #arm_target_os="armlinux"
        arm_target_os="arm_baremetal"
        WITH_AVX=OFF
    fi
    cmake_mutable_options="-DLITE_WITH_ARM=$with_arm \
                        -DLITE_WITH_X86=OFF \
			-DLITE_WITH_BAREMETAL=ON \
                        -DARM_TARGET_ARCH_ABI=$arm_arch \
                        -DARM_TARGET_OS=$arm_target_os \
                        -DARM_TARGET_LANG=$TOOLCHAIN \
                        -DLITE_BUILD_EXTRA=$WITH_EXTRA \
                        -DLITE_WITH_PYTHON=$WITH_PYTHON \
                        -DLITE_WITH_STATIC_LIB=$WITH_STATIC_LIB \
                        -DLITE_WITH_CV=$WITH_CV \
                        -DLITE_WITH_LOG=$WITH_LOG \
                        -DLITE_WITH_EXCEPTION=$WITH_EXCEPTION \
                        -DLITE_BUILD_TAILOR=$WITH_STRIP \
                        -DLITE_OPTMODEL_DIR=$OPTMODEL_DIR \
                        -DWITH_STATIC_MKL=$WITH_STATIC_MKL \
                        -DWITH_AVX=$WITH_AVX \
                        -DLITE_WITH_OPENCL=$WITH_OPENCL \
                        -DLITE_WITH_OPENMP=OFF \
                        -DLITE_WITH_METAL=$WITH_METAL \
                        -DLITE_WITH_RKNPU=$WITH_ROCKCHIP_NPU \
                        -DLITE_WITH_XPU=$WITH_KUNLUNXIN_XPU \
                        -DXPU_WITH_XFT=$WITH_KUNLUNXIN_XFT \
                        -DLITE_WITH_TRAIN=$WITH_TRAIN  \
                        -DLITE_WITH_NNADAPTER=$WITH_NNADAPTER \
                        -DNNADAPTER_WITH_ROCKCHIP_NPU=$NNADAPTER_WITH_ROCKCHIP_NPU \
                        -DNNADAPTER_WITH_EEASYTECH_NPU=$NNADAPTER_WITH_EEASYTECH_NPU \
                        -DNNADAPTER_WITH_IMAGINATION_NNA=$NNADAPTER_WITH_IMAGINATION_NNA \
                        -DNNADAPTER_WITH_HUAWEI_ASCEND_NPU=$NNADAPTER_WITH_HUAWEI_ASCEND_NPU \
                        -DNNADAPTER_WITH_AMLOGIC_NPU=$NNADAPTER_WITH_AMLOGIC_NPU \
                        -DNNADAPTER_WITH_CAMBRICON_MLU=$NNADAPTER_WITH_CAMBRICON_MLU \
                        -DNNADAPTER_WITH_VERISILICON_TIMVX=$NNADAPTER_WITH_VERISILICON_TIMVX \
                        -DNNADAPTER_WITH_NVIDIA_TENSORRT=$NNADAPTER_WITH_NVIDIA_TENSORRT \
                        -DNNADAPTER_WITH_QUALCOMM_QNN=$NNADAPTER_WITH_QUALCOMM_QNN \
                        -DNNADAPTER_WITH_KUNLUNXIN_XTCL=$NNADAPTER_WITH_KUNLUNXIN_XTCL \
                        -DNNADAPTER_WITH_INTEL_OPENVINO=$NNADAPTER_WITH_INTEL_OPENVINO \
                        -DNNADAPTER_WITH_GOOGLE_XNNPACK=$NNADAPTER_WITH_GOOGLE_XNNPACK \
                        -DLITE_WITH_PROFILE=${WITH_PROFILE} \
                        -DLITE_WITH_ARM82_FP16=$BUILD_ARM82_FP16 \
                        -DWITH_ARM_DOTPROD=$WITH_ARM_DOTPROD \
                        -DLITE_WITH_PRECISION_PROFILE=${WITH_PRECISION_PROFILE} \
                        -DLITE_SKIP_SUPPORT_0_DIM_TENSOR_PASS=$SKIP_SUPPORT_0_DIM_TENSOR_PASS \
                        -DLITE_WITH_ARM_DNN_LIBRARY=$WITH_ARM_DNN_LIBRARY \
			-DLITE_WITH_OPENMP=OFF \
			-DLITE_THREAD_POOL=OFF \
                        -DLITE_ON_TINY_PUBLISH=$WITH_TINY_PUBLISH"

}

function prepare_workspace {
    GEN_CODE_PATH_PREFIX=$pdlite_build_dir/lite/gen_code
    mkdir -p ${GEN_CODE_PATH_PREFIX}
    touch ${GEN_CODE_PATH_PREFIX}/__generated_code__.cc
    # 2.Prepare debug tool
    DEBUG_TOOL_PATH_PREFIX=$pdlite_build_dir/lite/tools/debug
    mkdir -p ${DEBUG_TOOL_PATH_PREFIX}
    cp $pdlite_root_dir/lite/tools/debug/analysis_tool.py ${DEBUG_TOOL_PATH_PREFIX}/
}


readonly CMAKE_COMMON_OPTIONS="-DCMAKE_BUILD_TYPE=Debug \
                            -DWITH_MKLDNN=OFF \
                            -DWITH_TESTING=OFF"

if [[ ${target} == *"ethos-u55"*  ]]; then
    target_cpu=cortex-m55
    target_board=corstone-300
else
    target_cpu=cortex-m85
    target_board=corstone-320
fi
function build_paddlelite() {
    set -x
    init_cmake_options

    [[ -d "${pdlite_build_dir}" ]] \
        && echo "[${FUNCNAME[0]}] Warn: using already existing build-dir for paddlelite: ${pdlite_build_dir}!!"
    rm -rf "${pdlite_build_dir}"
    mkdir -p "${pdlite_build_dir}"

    cd "${pdlite_build_dir}"
    rm -f $pdlite_root_dir/lite/api/paddle_use_ops.h
    rm -f $pdlite_root_dir/lite/api/paddle_use_kernels.h
    prepare_workspace
    cmake ../ \
	  -DTARGET_CPU=${target_cpu} \
          -DTARGET_BOARD=${target_board} \
          ${CMAKE_COMMON_OPTIONS} \
          ${cmake_mutable_options} \
	  -DCMAKE_TOOLCHAIN_FILE="${toolchain_cmake}"
    echo "[${FUNCNAME[0]}] Configured CMAKE"
    make publish_inference -j8
}

function build_paddle_runner() {
    set -x
    echo "start build_paddle_runner"
    cd ${script_dir}/paddle_runner
    rm -rf cmake-out
    mkdir -p "${pdlite_build_dir}"
    cmake -DCMAKE_TOOLCHAIN_FILE=${toolchain_cmake}     \
          -DTARGET_CPU=${target_cpu}                    \
	  -DPDLITE_ROOT_DIR=${pdlite_root_dir}          \
          -DTARGET_BOARD=${target_board}                \
	  -DPDLITE_BUILD_DIR=${pdlite_build_dir}        \
          -DETHOSU_TARGET_NPU_CONFIG=${target}          \
          -B ${paddle_runner_path}/cmake-out          \
          -DETHOS_SDK_PATH:PATH=${ethos_u_root_dir}     \
          -DPYTHON_EXECUTABLE=$(which python3)
    echo "[${FUNCNAME[0]}] Configured CMAKE"
    cmake --build ${paddle_runner_path}/cmake-out --parallel -- arm_paddle_runner
}

# Execute the paddle_runner on FVP Simulator
function run_fvp() {
    [[ $# -ne 1 ]] && { echo "[${FUNCNAME[0]}]" "Expexted elf binary name, got $*"; exit 1; }
    local elf_name=${1}
    elf=$(find ${paddle_runner_path} -name "${elf_name}")
    [[ ! -f $elf ]] && { echo "[${FUNCNAME[0]}]: Unable to find paddle_runner elf: ${elf}"; exit 1; }
    num_macs=$(echo ${target} | cut -d - -f 3)

    if [[ ${target} == *"ethos-u55"*  ]]; then
        echo "Running ${elf} for ${target} run with FVP:${fvp_model} num_macs:${num_macs}"
        ${fvp_model}                                            \
            -C ethosu.num_macs=${num_macs}                      \
            -C mps3_board.visualisation.disable-visualisation=1 \
            -C mps3_board.telnetterminal0.start_telnet=0        \
            -C mps3_board.uart0.out_file='-'                    \
            -C mps3_board.uart0.shutdown_on_eot=1               \
            -a "${elf}"                                         \
            --timelimit 120 || true # seconds
        echo "[${FUNCNAME[0]}] Simulation complete, $?"
    elif [[ ${target} == *"ethos-u85"*  ]]; then
        echo "Running ${elf} for ${target} run with FVP:${fvp_model} num_macs:${num_macs}"
    	${fvp_model}                                            \
            -C mps4_board.subsystem.ethosu.num_macs=${num_macs} \
            -C mps4_board.visualisation.disable-visualisation=1 \
            -C vis_hdlcd.disable_visualisation=1                \
            -C mps4_board.telnetterminal0.start_telnet=0        \
            -C mps4_board.uart0.out_file='-'                    \
            -C mps4_board.uart0.shutdown_on_eot=1               \
            -a "${elf}"                                         \
            --timelimit 120 || true # seconds
        echo "[${FUNCNAME[0]}] Simulation complete, $?"
    else
        echo "Running ${elf} for ${target} is not supported"
        exit 1
    fi
}
build_paddlelite
build_paddle_runner
run_fvp arm_paddle_runner

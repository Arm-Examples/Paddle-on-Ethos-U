#!/bin/bash


SOURCE_FILE=$(realpath "${BASH_SOURCE[0]}")
CURRENT_DIR=$(dirname "$SOURCE_FILE")
echo "Add ${CURRENT_DIR}/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi/bin to PATH"
echo "Add ${CURRENT_DIR}/FVP-corstone300/models/Linux64_GCC-9.3 to PATH"
echo "Add ${CURRENT_DIR}/FVP-corstone320/models/Linux64_GCC-9.3 to PATH"
echo "Add ${CURRENT_DIR}/FVP-corstone320/python/lib/ to PATH"
export PATH=${PATH}:${CURRENT_DIR}/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi/bin
export PATH=${PATH}:${CURRENT_DIR}/FVP-corstone300/models/Linux64_GCC-9.3
export PATH=${PATH}:${CURRENT_DIR}/FVP-corstone320/models/Linux64_GCC-9.3
export LD_LIBRARY_PATH=${CURRENT_DIR}/FVP-corstone320/python/lib/


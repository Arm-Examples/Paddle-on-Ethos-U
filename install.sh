#!/bin/bash

# Setup environment variables
source lite/backends/arm/ethos-u-scratch/setup_path.sh

# Build libpaddle_api_light_bundled.a
RUN_SH_PATH=lite/backends/arm/run.sh
# Ignore build commands
sed -i 's/^\(build_paddle_runner\)/#\1/' "$RUN_SH_PATH"
sed -i 's/^\(run_fvp arm_paddle_runner\)/#\1/' "$RUN_SH_PATH"

./$RUN_SH_PATH
sed -i 's/^#\(build_paddle_runner\)/\1/' "$RUN_SH_PATH"
sed -i 's/^#\(run_fvp arm_paddle_runner\)/\1/' "$RUN_SH_PATH"

# Install TOSA Serialization Library.
pushd readnb/serialization_lib
pip install -e .
popd

# Install dependencies
pip install -r readnb/requirements.txt

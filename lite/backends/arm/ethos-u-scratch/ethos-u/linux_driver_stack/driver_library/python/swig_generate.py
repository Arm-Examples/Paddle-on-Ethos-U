# SPDX-FileCopyrightText: Copyright 2021-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""
This script executes SWIG commands to generate C++ library wrappers.
"""
import subprocess
from pathlib import Path

__current_dir = Path(__file__).parent.absolute()


def generate_wrap(name, extr_includes):
    print('Generating wrappers for {}'.format(name))
    subprocess.check_output("swig -v -c++ -python" +
                            " -Wall" +
                            " -DSWIGWORDSIZE64 " + # Force 64-bit word size for uint64_t vector to work
                            " -o {}/src/ethosu_driver/_generated/{}_wrap.cpp ".format(__current_dir, name) +
                            "-outdir {}/src/ethosu_driver/_generated ".format(__current_dir) +
                            "{} ".format(extr_includes) +
                            "-I{}/src/ethosu_driver/swig ".format(__current_dir) +
                            "{}/src/ethosu_driver/swig/{}.i".format(__current_dir, name),
                            shell=True,
                            stderr=subprocess.STDOUT)


if __name__ == "__main__":
    includes = ["{}/../../driver_library/include".format(__current_dir)]
    generate_wrap('driver', "-I{} ".format(' -I'.join(includes)))

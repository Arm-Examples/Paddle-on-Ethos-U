#
# SPDX-FileCopyrightText: Copyright 2021-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
from argparse import ArgumentParser
import os
import logging
from pathlib import Path
from typing import List

import ethosu_driver as driver
try:
    import numpy as np
    with_numpy = True
except ImportError:
    with_numpy = False


def read_bin_file_to_buf(file_path: str) -> bytearray:
    with open(file_path, 'rb') as f:
        return bytearray(f.read())


def read_npy_file_to_buf(file_path: str) -> bytearray:
    ifm_arr = np.load(file_path).astype(dtype=np.int8, order='C')
    return ifm_arr.flatten().data


def read_ifms(ifm_files: List[str], use_npy: bool = False):
    read_file_to_buf = read_npy_file_to_buf if use_npy else read_bin_file_to_buf
    for ifm_file in ifm_files:
        yield read_file_to_buf(ifm_file)


def write_npy(dir: str, file_name: str, data: memoryview):
    ar = np.frombuffer(data, dtype=np.int8)
    file_path = os.path.join(dir, "{}.npy".format(file_name))
    if os.path.isfile(file_path):
        os.remove(file_path)
    np.save(file_path, ar)
    logging.info("File saved to {}".format(file_path))


def write_bin_file(dir: str, file_name: str, data: memoryview):
    file_path = os.path.join(dir, "{}.bin".format(file_name))
    if os.path.isfile(file_path):
        os.remove(file_path)
    with open(file_path, "wb") as f:
        f.write(data)
        logging.info("File saved to {}".format(file_path))


def write_ofm(buf: memoryview, ofm_index: int, model_path: str, output_dir: str, use_npy: bool = False):
    write_buf_to_file = write_npy if use_npy else write_bin_file
    model_file_name = Path(model_path).name
    ofm_name = "{}_ofm_{}".format(model_file_name, ofm_index)
    write_buf_to_file(output_dir, ofm_name, buf)


def main():
    format = "%(asctime)s %(levelname)s -  %(message)s"
    logging.basicConfig(format=format, level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("--device", help="Npu device name. Default: ethosu0", default="ethosu0")
    parser.add_argument("--model", help="Tflite model file path", required=True)
    parser.add_argument("--timeout", help="Inference timout in seconds, Default: infinite", default=-1, type=int)
    parser.add_argument("--inputs", nargs='+', help="list of files containing input feature maps", required=True)
    parser.add_argument("--output_dir", help="directory to store inference results, output feature maps. "
                                             "Default: current directory", default=os.getcwd())
    parser.add_argument("--npy", help="Use npy input/output", default=0, type=int)
    parser.add_argument("--profile_counters", help="Performance counters to profile", nargs=4, type=int, required=True)
    args = parser.parse_args()

    use_numpy = with_numpy & bool(int(args.npy))
    if use_numpy:
        logging.info("Running with numpy inputs/outputs")
    else:
        logging.info("Running with byte array inputs/outputs")

    # @TODO: Discuss if this is needed anymore. Remove this commented line, if not.
    # driver.reset()

    ifms_data = read_ifms(args.inputs, use_numpy)

    runner = driver.InferenceRunner(args.device, args.model)
    runner.set_enabled_counters(args.profile_counters)
    ofm_buffers = runner.run(list(ifms_data), int(args.timeout))

    for index, buffer_out in enumerate(ofm_buffers):
        logging.info("Output buffer size: {}".format(buffer_out.size()))
        write_ofm(buffer_out.data(), index, args.model, args.output_dir, use_numpy)

    inference_pmu_counters = runner.get_pmu_counters()

    # Profiling
    total_cycles = runner.get_pmu_total_cycles()
    for pmu, value in inference_pmu_counters:
        logging.info("\tNPU %d counter: %d", pmu, value)
    logging.info("\tNPU TOTAL cycles: %d", total_cycles)

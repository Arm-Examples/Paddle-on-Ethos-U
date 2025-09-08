#
# SPDX-FileCopyrightText: Copyright 2021-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import os
import ethosu_driver as driver
from ethosu_driver.inference_runner import read_npy_file_to_buf


def run_inference_test(runner, timeout, input_files, golden_outputs, shared_data_folder):

    full_path_input_files = []
    for input_file in input_files:
        full_path_input_files.append(os.path.join(shared_data_folder, input_file))

    ifms_data = []
    for ifm_file in full_path_input_files:
        ifms_data.append(read_npy_file_to_buf(ifm_file))

    ofm_buffers = runner.run(ifms_data, timeout)

    for index, buffer_out in enumerate(ofm_buffers):
        golden_output = read_npy_file_to_buf(os.path.join(shared_data_folder, golden_outputs[index]))
        assert buffer_out.data().nbytes == golden_output.nbytes
        for index, golden_value in enumerate(golden_output):
            assert golden_value == buffer_out.data()[index]


@pytest.mark.parametrize('device_name, model_name, timeout, input_files, golden_outputs',
                        [('ethosu0', 'model.tflite', 5000000000, ['model_ifm.npy'], ['model_ofm.npy'])])
def test_inference(device_name, model_name, input_files, timeout, golden_outputs, shared_data_folder):
    # Prepate full path of model and inputs
    full_path_model_file = os.path.join(shared_data_folder, model_name)

    runner = driver.InferenceRunner(device_name, full_path_model_file)
    run_inference_test(runner, timeout, input_files, golden_outputs, shared_data_folder)


@pytest.mark.parametrize('device_name, model_name, timeout, input_files, golden_outputs',
                         [('ethosu0', 'model.tflite', 5000000000,
                           [['model_ifm.npy'], ['model_ifm.npy']],
                           [['model_ofm.npy'], ['model_ofm.npy']])])
def test_inference_loop(device_name, model_name, input_files, timeout, golden_outputs, shared_data_folder):
    # Prepare full path of model and inputs
    full_path_model_file = os.path.join(shared_data_folder, model_name)

    runner = driver.InferenceRunner(device_name, full_path_model_file)
    for input_file, golden_output in zip(input_files, golden_outputs):
        run_inference_test(runner, timeout, input_file, golden_output, shared_data_folder)

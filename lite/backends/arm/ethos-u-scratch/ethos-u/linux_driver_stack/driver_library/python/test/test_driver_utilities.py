#
# SPDX-FileCopyrightText: Copyright 2021-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import os
import ethosu_driver as driver
from ethosu_driver.inference_runner import read_npy_file_to_buf


@pytest.fixture()
def device(device_name):
    device = driver.open_device(device_name)
    yield device


@pytest.fixture()
def network(device, model_name, shared_data_folder):
    network_file = os.path.join(shared_data_folder, model_name)
    network = driver.load_model(device, network_file)
    yield network


@pytest.mark.parametrize('device_name', ['blabla'])
def test_open_device_wrong_name(device_name):
    with pytest.raises(RuntimeError) as err:
        device = driver.open_device(device_name)
    # Only check for part of the exception since the exception returns
    # absolute path which will change on different machines.
    assert 'Failed to open device' in str(err.value)


@pytest.mark.parametrize('device_name', ['ethosu0'])
def test_network_filenotfound_exception(device, shared_data_folder):

    network_file = os.path.join(shared_data_folder, "some_unknown_model.tflite")

    with pytest.raises(RuntimeError) as err:
        driver.load_model(device, network_file)

    # Only check for part of the exception since the exception returns
    # absolute path which will change on different machines.
    assert 'Failed to open file:' in str(err.value)


@pytest.mark.parametrize('device_name', ['ethosu0'])
@pytest.mark.parametrize('model_name', ['model.tflite'])
def test_check_network_ifm_size(network):
    assert network.getIfmSize() > 0


@pytest.mark.parametrize('device_name', ['ethosu0'])
def test_allocate_buffers(device):
    buffers = driver.allocate_buffers(device, [128, 256])
    assert len(buffers) == 2
    assert buffers[0].size() == 128
    assert buffers[1].size() == 256


@pytest.mark.parametrize('device_name', ['ethosu0'])
@pytest.mark.parametrize('model_name', ['model.tflite'])
@pytest.mark.parametrize('ifms_file_list', [['model_ifm.npy']])
def test_set_ifm_buffers(device, network, ifms_file_list, shared_data_folder):
    full_path_input_files = []
    for input_file in ifms_file_list:
        full_path_input_files.append(os.path.join(shared_data_folder, input_file))

    ifms_data = []
    for ifm_file in full_path_input_files:
        ifms_data.append(read_npy_file_to_buf(ifm_file))

    ifms = driver.allocate_buffers(device, network.getIfmDims())
    driver.populate_buffers(ifms_data, ifms)
    assert len(ifms) > 0


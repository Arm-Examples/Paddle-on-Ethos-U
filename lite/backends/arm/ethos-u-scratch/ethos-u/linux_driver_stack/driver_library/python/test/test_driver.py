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
    device = driver.Device("/dev/{}".format(device_name))
    yield device


@pytest.fixture()
def network_file(model_name, shared_data_folder):
    network_file = os.path.join(shared_data_folder, model_name)
    yield network_file

@pytest.fixture()
def network(device, network_file):
    network = driver.Network(device, network_file)
    yield network

@pytest.mark.parametrize('device_name', ['ethosu0'])
def test_check_device_swig_ownership(device):
    # Check to see that SWIG has ownership for parser. This instructs SWIG to take
    # ownership of the return value. This allows the value to be automatically
    # garbage-collected when it is no longer in use
    assert device.thisown


@pytest.mark.parametrize('device_name', ['ethosu0'])
def test_device_ping(device):
    device.ping()


@pytest.mark.parametrize('device_name', ['blabla'])
def test_device_wrong_name(device_name):
    with pytest.raises(RuntimeError) as err:
        driver.Device("/dev/{}".format(device_name))
    # Only check for part of the exception since the exception returns
    # absolute path which will change on different machines.
    assert 'Failed to open device' in str(err.value)


@pytest.mark.parametrize('device_name', ['ethosu0'])
@pytest.mark.parametrize('model_name', ['model.tflite'])
def test_driver_network_from_bytearray(device, network_file):
    network_data = None
    with open(network_file, 'rb') as file:
        network_data = file.read()
    network = driver.Network(device, network_data)


@pytest.mark.parametrize('device_name', ['ethosu0'])
def test_driver_network_from_empty_bytearray(device):
    with pytest.raises(RuntimeError) as err:
        network = driver.Network(device, bytearray())

    assert 'Failed to create the network, networkSize is zero' in str(err.value)


@pytest.mark.parametrize('device_name', ['ethosu0'])
@pytest.mark.parametrize('model_name', ['model.tflite'])
def test_driver_network_from_file(device, network_file):
        network = driver.Network(device, network_file)


@pytest.mark.parametrize('device_name', ['ethosu0'])
@pytest.mark.parametrize('model_name', ['some_unknown_model.tflite'])
def test_driver_network_filenotfound_exception(device, network_file):
    with pytest.raises(RuntimeError) as err:
        network = driver.Network(device, network_file)

    # Only check for part of the exception since the exception returns
    # absolute path which will change on different machines.
    assert 'Failed to open file:' in str(err.value)


@pytest.mark.parametrize('device_name', ['ethosu0'])
@pytest.mark.parametrize('model_name', ['model.tflite'])
def test_check_network_swig_ownership(network):
    # Check to see that SWIG has ownership for parser. This instructs SWIG to take
    # ownership of the return value. This allows the value to be automatically
    # garbage-collected when it is no longer in use
    assert network.thisown


@pytest.mark.parametrize('device_name', ['ethosu0'])
@pytest.mark.parametrize('model_name', ['model.tflite'])
def test_check_network_ifm_size(device, network):
    assert network.getIfmSize() > 0


@pytest.mark.parametrize('device_name', ['ethosu0'])
@pytest.mark.parametrize('model_name', ['model.tflite'])
def test_check_network_ofm_size(device, network):
    assert network.getOfmSize() > 0


@pytest.mark.parametrize('device_name', ['ethosu0'])
def test_check_buffer_swig_ownership(device):
    buffer = driver.Buffer(device, 1024)
    assert buffer.thisown


@pytest.mark.parametrize('device_name', ['ethosu0'])
def test_check_buffer_getFd(device):
    buffer = driver.Buffer(device, 1024)
    assert buffer.getFd() >= 0


@pytest.mark.parametrize('device_name', ['ethosu0'])
def test_check_buffer_size(device):
    buffer = driver.Buffer(device, 1024)
    assert buffer.size() == 1024


@pytest.mark.parametrize('device_name', ['ethosu0'])
@pytest.mark.parametrize('model_name', ['model.tflite'])
def test_check_buffer_clear(device, network_file):
    buffer = driver.Buffer(device, network_file)

    buffer.clear()
    for i in range(buffer.size()):
        assert buffer.data()[i] == 0


def test_getMaxPmuEventCounters():
    assert driver.Inference.getMaxPmuEventCounters() > 0


@pytest.fixture()
def inf(device_name, model_name, input_files, timeout, shared_data_folder):
    # Prepate full path of model and inputs
    full_path_model_file = os.path.join(shared_data_folder, model_name)
    full_path_input_files = []
    for input_file in input_files:
        full_path_input_files.append(os.path.join(shared_data_folder, input_file))

    ifms_data = []
    for ifm_file in full_path_input_files:
        ifms_data.append(read_npy_file_to_buf(ifm_file))

    device = driver.open_device(device_name)
    device.ping()
    network = driver.load_model(device, full_path_model_file)
    ofms = driver.allocate_buffers(device, network.getOfmDims())
    ifms = driver.allocate_buffers(device, network.getIfmDims())

    # ofm_buffers = runner.run(ifms_data,timeout, ethos_pmu_counters)
    driver.populate_buffers(ifms_data, ifms)
    ethos_pmu_counters = [1]
    enable_cycle_counter = True
    inf_inst = driver.Inference(network, ifms, ofms, ethos_pmu_counters, enable_cycle_counter)
    inf_inst.wait(int(timeout))

    yield inf_inst


@pytest.mark.parametrize('device_name, model_name, timeout, input_files',
                        [('ethosu0', 'model.tflite', 5000000000, ['model_ifm.npy'])])
def test_inf_get_cycle_counter(inf):
    total_cycles = inf.getCycleCounter()
    assert total_cycles >= 0


@pytest.mark.parametrize('device_name, model_name, timeout, input_files',
                        [('ethosu0', 'model.tflite', 5000000000, ['model_ifm.npy'])])
def test_inf_get_pmu_counters(inf):
    inf_pmu_counter = inf.getPmuCounters()
    assert len(inf_pmu_counter) > 0


@pytest.mark.parametrize('device_name', ['ethosu0'])
def test_capabilities(device):
    cap = device.capabilities()
    assert cap.hwId
    assert cap.hwCfg
    assert cap.driver

@pytest.mark.parametrize('device_name', ['ethosu0'])
def test_kernel_driver_version(device):
    version = device.getDriverVersion()
    zero_version = [0, 0, 0]
    # Validate that a version was returned
    assert zero_version != [version.major, version.minor, version.patch]
    # Check that supported kernel driver major versions are available in Python API
    assert driver.MAX_SUPPORTED_KERNEL_DRIVER_MAJOR_VERSION
    assert driver.MIN_SUPPORTED_KERNEL_DRIVER_MAJOR_VERSION

def test_driver_library_version():
    version = driver.getLibraryVersion()
    expected_version = [driver.DRIVER_LIBRARY_VERSION_MAJOR,
                        driver.DRIVER_LIBRARY_VERSION_MINOR,
                        driver.DRIVER_LIBRARY_VERSION_PATCH]
    # Validate that the expected version was returned
    assert expected_version == [version.major, version.minor, version.patch]

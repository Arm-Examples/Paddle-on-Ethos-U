# SPDX-FileCopyrightText: Copyright 2021-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
import logging
import time
from typing import List
from .._generated.driver import Device, Inference, Network, Buffer, InferenceStatus_OK


def open_device(device: str) -> Device:
    """Opens the Ethos-U device file descriptor.

        Args:
            device: device name.

        Returns:
            `Device`: Return the  object that represents Ethos-U device file descriptor and manages Ethos-U device lifecycle.
    """
    device = Device("/dev/{}".format(device))
    return device


def load_model(device: Device, model: str) -> Network:
    """Create a `Network` when providing `Device` object and a string containing tflite file path.

        Args:
            device: `Device` object that Ethos-U device file descriptor.
            model: tflite model file path .

        Returns:
            `Network`: Return the object that represent the neural __network file descriptor received from the Ethos-U device.
    """
    logging.info("Creating network")
    return Network(device, model)


def populate_buffers(input_data: List[bytearray], buffers: List[Buffer]):
    """Set the feature maps associated memory buffer with the given data.

        Args:
            input_data: list of input feature maps data.
            buffers: list of already initialized ifm buffers.
        Raises:
             RuntimeError: if input data size is incorrect.
    """
    number_of_buffers = len(buffers)

    if number_of_buffers != len(input_data):
        raise RuntimeError("Incorrect number of inputs, expected {}, got {}.".format(number_of_buffers, len(input_data)))

    for index, (buffer, data_chunk) in enumerate(zip(buffers, input_data)):
        size = buffer.size()
        logging.info("Copying data to a buffer {} of {} with size = {}".format(index + 1, number_of_buffers, size))

        if len(data_chunk) > size:
            raise RuntimeError("Buffer expects {} bytes, got {} bytes.".format(size, len(data_chunk)))
        buffer.from_buffer(data_chunk)


def allocate_buffers(device: Device, dimensions: List) -> List[Buffer]:
    """Returns output feature maps associated with memory buffers.

        Args:
            device: `Device` object that Ethos-U device file descriptor.
            dimensions: `Network` object that represent the neural __network file descriptor.

        Returns:
            list: output feature map buffers.
    """
    buffers = []
    total = len(dimensions)
    for index, size in enumerate(dimensions):
        logging.info("Allocating {} of {} buffer with size = {}".format(index + 1, total, size))
        buffer = Buffer(device, size)
        buffers.append(buffer)

    return buffers


def get_results(inference: Inference) -> List[Buffer]:
    """Retrieves output inference buffers

        Args:
            inference: `Inference` object that represents the inference file descriptor.

        Returns:
            list: list of buffer objects
        Raises:
            RuntimeError: in case of inference returned failure status.

    """
    if InferenceStatus_OK != inference.status():
        raise RuntimeError("Inference failed!")
    else:
        logging.info("Inference succeeded!")
        return inference.getOfmBuffers()


class InferenceRunner:
    """Helper class to execute inference."""

    def __init__(self, device_name: str, model: str):
        """Initialises instance to execute inferences on the given model with given device

            Device is opened with the name '/dev/<device_name>'.
            Input/Output feature maps memory is allocated.

            Args:
                device_name: npu device name
                model: Tflite model file path
        """
        self.__device = open_device(device_name)
        if not InferenceRunner.wait_for_ping(self.__device, 3):
            raise RuntimeError("Failed to communicate with device {}".format(device_name))

        self.__network = load_model(self.__device, model)
        # it is important to have a reference to current inference object to have access to OFMs.
        self.__inf = None
        self.__enabled_counters = ()

    @staticmethod
    def wait_for_ping(device: Device, count: int) -> bool:
        if count == 0:
            return False
        try:
            device.ping()
            return True
        except:
            logging.info("Waiting for device: {}".format(count))
            time.sleep(0.5)
            return InferenceRunner.wait_for_ping(device,  count-1)

    def set_enabled_counters(self, enabled_counters: List[int] = ()):
        """Set the enabled performance counter to use during inference.

            Args:
                enabled_counters: list of integer counter to enable.
            Raises:
                ValueError: in case of inference returned failure status or the Pmu counter requests exceed the maximum supported.
        """
        max_pmu_events = Inference.getMaxPmuEventCounters()
        if len(enabled_counters) > max_pmu_events:
            raise ValueError("Number of PMU counters requested exceed the maximum supported ({}).".format(max_pmu_events))
        self.__enabled_counters = enabled_counters

    def run(self, input_data: List[bytearray], timeout: int) -> List[Buffer]:
        """Run a inference with the given input feature maps data.

            Args:
                input_data: data list containing input data as binary arrays
                timeout: inference timout in nano seconds

            Returns:
                list: list of buffer objects
        """
        ofms = allocate_buffers(self.__device, self.__network.getOfmDims())
        ifms = allocate_buffers(self.__device, self.__network.getIfmDims())
        populate_buffers(input_data, ifms)

        self.__inf = Inference(
            self.__network,
            ifms,
            ofms,
            self.__enabled_counters,
            True)

        self.__inf.wait(int(timeout))
        return get_results(self.__inf)

    def get_pmu_counters(self) -> List:
        """Return the PMU data for the inference run.

            Returns:
                list: pairs of PMU type and cycle count value
        """
        return list(zip(self.__enabled_counters, self.__inf.getPmuCounters()))

    def get_pmu_total_cycles(self) -> int:
        """
        Returns the total cycle count, including idle cycles, as reported by
        the PMU

        Returns: total cycle count
        """
        return self.__inf.getCycleCounter()

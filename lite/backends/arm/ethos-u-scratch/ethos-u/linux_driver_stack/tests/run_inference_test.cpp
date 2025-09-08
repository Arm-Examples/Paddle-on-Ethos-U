/*
 * SPDX-FileCopyrightText: Copyright 2022-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ethosu.hpp>
#include <uapi/ethosu.h>

#include <cstring>
#include <errno.h>
#include <iostream>
#include <list>
#include <memory>
#include <sstream>
#include <stdio.h>
#include <string>
#include <unistd.h>

#include "input.h"
#include "model.h"
#include "output.h"
#include "test_assertions.hpp"

using namespace EthosU;

namespace {

int64_t defaultTimeout = 60000000000;

void testPing(const Device &device) {
    int r;
    try {
        r = device.ioctl(ETHOSU_IOCTL_PING);
    } catch (std::exception &e) { throw TestFailureException("Ping test: ", e.what()); }

    TEST_ASSERT(r == 0);
}

void testDriverVersion(const Device &device) {
    int r;
    struct ethosu_uapi_kernel_driver_version version = {};
    try {
        r = device.ioctl(ETHOSU_IOCTL_DRIVER_VERSION_GET, &version);
    } catch (std::exception &e) { throw TestFailureException("Driver version test: ", e.what()); }

    TEST_ASSERT(r == 0);
    TEST_ASSERT(version.major == ETHOSU_KERNEL_DRIVER_VERSION_MAJOR);
    TEST_ASSERT(version.minor == ETHOSU_KERNEL_DRIVER_VERSION_MINOR);
    TEST_ASSERT(version.patch == ETHOSU_KERNEL_DRIVER_VERSION_PATCH);
}

void testCapabilties(const Device &device) {
    Capabilities capabilities;
    try {
        capabilities = device.capabilities();
    } catch (std::exception &e) { throw TestFailureException("Capabilities test: ", e.what()); }

    TEST_ASSERT(capabilities.hwId.architecture > SemanticVersion());
    TEST_ASSERT(capabilities.hwCfg.type == HardwareConfiguration::DeviceType::SUBSYSTEM);
}

void testBufferSeek(const Device &device) {
    try {
        Buffer buf{device, 1024};

        // SEEK_END should return the size of the buffer
        TEST_ASSERT(lseek(buf.getFd(), 0, SEEK_END) == 1024);

        // SEEK_SET is supported when moving the file pointer to the start
        TEST_ASSERT(lseek(buf.getFd(), 0, SEEK_SET) == 0);

        // SEEK_CUR is not supported
        errno = 0;
        TEST_ASSERT(lseek(buf.getFd(), 0, SEEK_CUR) == -1);
        TEST_ASSERT(errno == EINVAL);

        // Non-zero offset is not supported
        errno = 0;
        TEST_ASSERT(lseek(buf.getFd(), 1, SEEK_CUR) == -1);
        TEST_ASSERT(errno == EINVAL);

        errno = 0;
        TEST_ASSERT(lseek(buf.getFd(), 2, SEEK_END) == -1);
        TEST_ASSERT(errno == EINVAL);

        errno = 0;
        TEST_ASSERT(lseek(buf.getFd(), 3, SEEK_SET) == -1);
        TEST_ASSERT(errno == EINVAL);
    } catch (std::exception &e) { throw TestFailureException("Buffer seek test: ", e.what()); }
}

void testNetworkInfoNotExistentIndex(const Device &device) {
    try {
        Network(device, 0);
        FAIL();
    } catch (Exception &e) {
        // good it should have thrown
    } catch (std::exception &e) { throw TestFailureException("NetworkInfo no index test: ", e.what()); }
}

void testNetworkInfoBuffer(const Device &device) {
    try {
        Network network(device, networkModelData, sizeof(networkModelData));
        TEST_ASSERT(network.getIfmDims().size() == 1);
        TEST_ASSERT(network.getOfmDims().size() == 1);
    } catch (std::exception &e) { throw TestFailureException("NetworkInfo buffer test: ", e.what()); }
}

void testNetworkInfoUnparsableBuffer(const Device &device) {
    try {
        try {
            Network network(device, networkModelData + sizeof(networkModelData) / 4, sizeof(networkModelData) / 4);
            FAIL();
        } catch (Exception) {
            // good, it should have thrown!
        }
    } catch (std::exception &e) { throw TestFailureException("NetworkInfo unparsable buffer test: ", e.what()); }
}

void testNetworkInvalidType(const Device &device) {
    const std::string expected_error = std::string("IOCTL cmd=NETWORK_CREATE") + " failed: " + std::strerror(EINVAL);
    struct ethosu_uapi_network_create net_req = {};
    net_req.type                              = ETHOSU_UAPI_NETWORK_INDEX + 1;
    try {
        int r = device.ioctl(ETHOSU_IOCTL_NETWORK_CREATE, &net_req);
        FAIL();
    } catch (Exception &e) {
        // The call is expected to throw
        TEST_ASSERT(expected_error.compare(e.what()) == 0);
    } catch (std::exception &e) { throw TestFailureException("NetworkCreate invalid type test: ", e.what()); }
}

void testNetworkInvalidDataPtr(const Device &device) {
    const std::string expected_error = std::string("IOCTL cmd=NETWORK_CREATE") + " failed: " + std::strerror(EINVAL);
    struct ethosu_uapi_network_create net_req = {};
    net_req.type                              = ETHOSU_UAPI_NETWORK_USER_BUFFER;
    net_req.network.data_ptr                  = 0U;
    net_req.network.size                      = 128U;
    try {
        int r = device.ioctl(ETHOSU_IOCTL_NETWORK_CREATE, &net_req);
        FAIL();
    } catch (Exception &e) {
        // The call is expected to throw
        TEST_ASSERT(expected_error.compare(e.what()) == 0);
    } catch (std::exception &e) { throw TestFailureException("NetworkCreate invalid data ptr: ", e.what()); }
}

void testNetworkInvalidDataSize(const Device &device) {
    const std::string expected_error = std::string("IOCTL cmd=NETWORK_CREATE") + " failed: " + std::strerror(EINVAL);
    struct ethosu_uapi_network_create net_req = {};
    net_req.type                              = ETHOSU_UAPI_NETWORK_USER_BUFFER;
    net_req.network.data_ptr                  = reinterpret_cast<uintptr_t>(networkModelData);
    net_req.network.size                      = 0U;
    try {
        int r = device.ioctl(ETHOSU_IOCTL_NETWORK_CREATE, &net_req);
        FAIL();
    } catch (Exception &e) {
        // The call is expected to throw
        TEST_ASSERT(expected_error.compare(e.what()) == 0);
    } catch (std::exception &e) { throw TestFailureException("NetworkCreate invalid data size: ", e.what()); }
}

void testRunInferenceBuffer(const Device &device) {
    try {
        auto network = std::make_shared<Network>(device, networkModelData, sizeof(networkModelData));

        std::vector<std::shared_ptr<Buffer>> inputBuffers;
        std::vector<std::shared_ptr<Buffer>> outputBuffers;

        auto inputBuffer = std::make_shared<Buffer>(device, sizeof(inputData));
        std::memcpy(inputBuffer->data(), inputData, sizeof(inputData));

        inputBuffers.push_back(inputBuffer);
        outputBuffers.push_back(std::make_shared<Buffer>(device, sizeof(expectedOutputData)));
        std::vector<uint8_t> enabledCounters(Inference::getMaxPmuEventCounters());

        auto inference = std::make_shared<Inference>(network,
                                                     inputBuffers.begin(),
                                                     inputBuffers.end(),
                                                     outputBuffers.begin(),
                                                     outputBuffers.end(),
                                                     enabledCounters,
                                                     false);

        bool timedout = inference->wait(defaultTimeout);
        TEST_ASSERT(!timedout);

        InferenceStatus status = inference->status();
        TEST_ASSERT(status == InferenceStatus::OK);

        bool success = inference->cancel();
        TEST_ASSERT(!success);

        TEST_ASSERT(std::memcmp(expectedOutputData, outputBuffers[0]->data(), sizeof(expectedOutputData)) == 0);

    } catch (std::exception &e) { throw TestFailureException("Inference run test: ", e.what()); }
}

} // namespace

int main() {
    Device device;

    try {
        testPing(device);
        testDriverVersion(device);
        testCapabilties(device);
        testBufferSeek(device);
        testNetworkInvalidType(device);
        testNetworkInvalidDataPtr(device);
        testNetworkInvalidDataSize(device);
        testNetworkInfoNotExistentIndex(device);
        testNetworkInfoBuffer(device);
        testNetworkInfoUnparsableBuffer(device);
        testRunInferenceBuffer(device);
    } catch (TestFailureException &e) {
        std::cerr << "Test failure: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

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

void testCancelInference(const Device &device) {
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

        InferenceStatus status = inference->status();
        TEST_ASSERT(status == InferenceStatus::RUNNING);

        bool success = inference->cancel();
        TEST_ASSERT(success);

        status = inference->status();
        TEST_ASSERT(status == InferenceStatus::ABORTED);

        bool timedout = inference->wait(defaultTimeout);
        TEST_ASSERT(!timedout);

    } catch (std::exception &e) { throw TestFailureException("Inference run test: ", e.what()); }
}

} // namespace

int main() {
    Device device;

    try {
        testCancelInference(device);
    } catch (TestFailureException &e) {
        std::cerr << "Test failure: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

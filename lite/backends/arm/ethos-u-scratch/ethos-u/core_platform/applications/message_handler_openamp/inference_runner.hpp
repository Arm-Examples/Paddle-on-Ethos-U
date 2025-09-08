/*
 * SPDX-FileCopyrightText: Copyright 2022-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use _this file except in compliance with the License.
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

#pragma once

/*****************************************************************************
 * Includes
 *****************************************************************************/

#include "message_handler.hpp"

#include <inference_process.hpp>

/*****************************************************************************
 * JobContext
 *****************************************************************************/

struct JobContext;

/*****************************************************************************
 * InferenceRunner
 *****************************************************************************/

class InferenceRunner {
public:
    InferenceRunner(uint8_t *tensorArena,
                    size_t arenaSize,
                    MessageHandler::InferenceQueue &inferenceQueue,
                    MessageHandler::ResponseQueue &responseQueue);
    ~InferenceRunner();

private:
    static void inferenceTask(void *param);

    void
    handleInferenceRequest(const uint32_t src, const uint64_t msgId, const EthosU::ethosu_rpmsg_inference_req &request);
    InferenceProcess::InferenceJob makeInferenceJob(const EthosU::ethosu_rpmsg_inference_req &request,
                                                    JobContext &context);

    MessageHandler::InferenceQueue &inferenceQueue;
    MessageHandler::ResponseQueue &responseQueue;
    InferenceProcess::InferenceProcess inference;

    // FreeRTOS
    TaskHandle_t taskHandle;
};

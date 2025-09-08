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

#include <mailbox.hpp>
#include <rpmsg/ethosu_rpmsg.h>

#include "queue.hpp"
#include "remoteproc.hpp"

/*****************************************************************************
 * Messages
 *****************************************************************************/

struct Message {
    Message() {}

    Message(const uint32_t _src,
            const EthosU::ethosu_rpmsg_type _type = EthosU::ETHOSU_RPMSG_MAX,
            const uint64_t msgId                  = 0,
            const uint32_t _length                = 0) :
        src(_src),
        length(_length) {
        rpmsg.header.magic  = ETHOSU_RPMSG_MAGIC;
        rpmsg.header.type   = _type;
        rpmsg.header.msg_id = msgId;
    }

    uint32_t src               = 0;
    uint32_t length            = 0;
    EthosU::ethosu_rpmsg rpmsg = {};
};

/*****************************************************************************
 * MessageHandler
 *****************************************************************************/

class MessageHandler : public Rpmsg {
public:
    using InferenceQueue = Queue<Message *>;
    using ResponseQueue  = Queue<Message *>;

    MessageHandler(RProc &rproc, const char *const name);
    virtual ~MessageHandler();

    InferenceQueue &getInferenceQueue() {
        return inferenceQueue;
    }

    InferenceQueue &getResponseQueue() {
        return responseQueue;
    }

protected:
    // Handle incoming rpmsg
    int handleMessage(void *data, size_t len, uint32_t src) override;

    // Outgoing messages
    void sendError(const uint32_t src, const EthosU::ethosu_rpmsg_err_type type, const char *message);
    void sendPong(const uint32_t src, const uint64_t msgId);
    void sendVersionRsp(const uint32_t src, const uint64_t msgId);
    void sendCapabilitiesRsp(const uint32_t src, const uint64_t msgId);
    void sendNetworkInfoRsp(const uint32_t src, const uint64_t msgId, EthosU::ethosu_rpmsg_network_buffer &network);
    void
    forwardInferenceReq(const uint32_t src, const uint64_t msgId, const EthosU::ethosu_rpmsg_inference_req &inference);
    void sendInferenceRsp(const uint32_t src, const uint64_t msgId, const EthosU::ethosu_rpmsg_status status);
    void sendCancelInferenceRsp(const uint32_t src, const uint64_t msgId, const EthosU::ethosu_rpmsg_status status);

    EthosU::ethosu_rpmsg_capabilities_rsp getCapabilities() const;
    bool getNetwork(EthosU::ethosu_rpmsg_network_buffer &buffer);

    // Tasks returning response messages
    static void responseTask(void *param);

private:
    InferenceQueue inferenceQueue;
    ResponseQueue responseQueue;
    EthosU::ethosu_rpmsg_capabilities_rsp capabilities;

    // FreeRTOS
    TaskHandle_t taskHandle;
};

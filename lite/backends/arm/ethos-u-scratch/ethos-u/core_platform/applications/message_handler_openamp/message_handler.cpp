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

/*****************************************************************************
 * Includes
 *****************************************************************************/

#include "message_handler.hpp"

#include <cinttypes>
#include <cstdlib>

#include <ethosu_log.h>
#include <inference_parser.hpp>

#ifdef ETHOSU
#include <ethosu_driver.h>
#endif

/*****************************************************************************
 * Networks
 *****************************************************************************/

namespace {
#if defined(__has_include)

#if defined(MODEL_0)
namespace Model0 {
#include STRINGIFY(MODEL_0)
}
#endif

#if defined(MODEL_1)
namespace Model1 {
#include STRINGIFY(MODEL_1)
}
#endif

#if defined(MODEL_2)
namespace Model2 {
#include STRINGIFY(MODEL_2)
}
#endif

#if defined(MODEL_3)
namespace Model3 {
#include STRINGIFY(MODEL_3)
}
#endif

#endif

bool getIndexedNetwork(const uint32_t index, void *&data, size_t &size) {
    switch (index) {
#if defined(MODEL_0)
    case 0:
        data = reinterpret_cast<void *>(Model0::networkModelData);
        size = sizeof(Model0::networkModelData);
        break;
#endif

#if defined(MODEL_1)
    case 1:
        data = reinterpret_cast<void *>(Model1::networkModelData);
        size = sizeof(Model1::networkModelData);
        break;
#endif

#if defined(MODEL_2)
    case 2:
        data = reinterpret_cast<void *>(Model2::networkModelData);
        size = sizeof(Model2::networkModelData);
        break;
#endif

#if defined(MODEL_3)
    case 3:
        data = reinterpret_cast<void *>(Model3::networkModelData);
        size = sizeof(Model3::networkModelData);
        break;
#endif

    default:
        LOG_WARN("Network model index out of range. index=%" PRIu32, index);
        return true;
    }

    return false;
}

} // namespace

/*****************************************************************************
 * MessageHandler
 *****************************************************************************/

MessageHandler::MessageHandler(RProc &_rproc, const char *const _name) :
    Rpmsg(_rproc, _name), inferenceQueue(), responseQueue(), capabilities(getCapabilities()), taskHandle() {
    BaseType_t ret = xTaskCreate(responseTask, "responseTask", 1024, this, 3, &taskHandle);
    if (ret != pdPASS) {
        LOG_ERR("Failed to create response task");
        abort();
    }
}

MessageHandler::~MessageHandler() {
    vTaskDelete(taskHandle);
}

int MessageHandler::handleMessage(void *data, size_t len, uint32_t src) {
    auto rpmsg = static_cast<EthosU::ethosu_rpmsg *>(data);

    LOG_DEBUG("Msg: src=%" PRIX32 ", len=%zu, magic=%" PRIX32 ", type=%" PRIu32,
              src,
              len,
              rpmsg->header.magic,
              rpmsg->header.type);

    if (rpmsg->header.magic != ETHOSU_RPMSG_MAGIC) {
        LOG_WARN("Msg: Invalid Magic");
        sendError(src, EthosU::ETHOSU_RPMSG_ERR_INVALID_MAGIC, "Invalid magic");
        return 0;
    }

    switch (rpmsg->header.type) {
    case EthosU::ETHOSU_RPMSG_PING: {
        LOG_INFO("Msg: Ping");
        sendPong(src, rpmsg->header.msg_id);
        break;
    }
    case EthosU::ETHOSU_RPMSG_VERSION_REQ: {
        LOG_INFO("Msg: Version request");
        sendVersionRsp(src, rpmsg->header.msg_id);
        break;
    }
    case EthosU::ETHOSU_RPMSG_CAPABILITIES_REQ: {
        if (len != sizeof(rpmsg->header)) {
            sendError(src, EthosU::ETHOSU_RPMSG_ERR_INVALID_PAYLOAD, "Incorrect capabilities request payload length.");
            break;
        }

        LOG_INFO("Msg: Capabilities request");

        sendCapabilitiesRsp(src, rpmsg->header.msg_id);
        break;
    }
    case EthosU::ETHOSU_RPMSG_INFERENCE_REQ: {
        if (len != sizeof(rpmsg->header) + sizeof(rpmsg->inf_req)) {
            sendError(src, EthosU::ETHOSU_RPMSG_ERR_INVALID_PAYLOAD, "Incorrect inference request payload length.");
            break;
        }

        forwardInferenceReq(src, rpmsg->header.msg_id, rpmsg->inf_req);
        break;
    }
    case EthosU::ETHOSU_RPMSG_CANCEL_INFERENCE_REQ: {
        if (len != sizeof(rpmsg->header) + sizeof(rpmsg->cancel_req)) {
            sendError(
                src, EthosU::ETHOSU_RPMSG_ERR_INVALID_PAYLOAD, "Incorrect cancel inference request payload length.");
            break;
        }

        auto &request = rpmsg->cancel_req;
        bool found    = false;
        inferenceQueue.erase([request, &found](auto &message) {
            if (message->rpmsg.header.msg_id == request.inference_handle) {
                found = true;
                delete message;
                return true;
            }

            return false;
        });

        if (found) {
            sendInferenceRsp(src, request.inference_handle, EthosU::ETHOSU_RPMSG_STATUS_ABORTED);
        }

        sendCancelInferenceRsp(src, rpmsg->header.msg_id, EthosU::ETHOSU_RPMSG_STATUS_OK);
        break;
    }
    case EthosU::ETHOSU_RPMSG_NETWORK_INFO_REQ: {
        if (len != sizeof(rpmsg->header) + sizeof(rpmsg->net_info_req)) {
            sendError(src, EthosU::ETHOSU_RPMSG_ERR_INVALID_PAYLOAD, "Incorrect network info request payload length.");
            break;
        }

        LOG_INFO("Msg: NetworkInfoReq. network={ type=%" PRIu32 ", index=%" PRIu32 ", buffer={ ptr=0x%" PRIX32
                 ", size=%" PRIu32 " } }",
                 rpmsg->net_info_req.network.type,
                 rpmsg->net_info_req.network.index,
                 rpmsg->net_info_req.network.buffer.ptr,
                 rpmsg->net_info_req.network.buffer.size);

        sendNetworkInfoRsp(src, rpmsg->header.msg_id, rpmsg->net_info_req.network);
        break;
    }
    default: {
        LOG_WARN("Msg: Unsupported message. type=%" PRIu32, rpmsg->header.type);

        char errMsg[128];
        snprintf(
            &errMsg[0], sizeof(errMsg), "Msg: Unknown message. type=%" PRIu32 ", length=%zu", rpmsg->header.type, len);

        sendError(src, EthosU::ETHOSU_RPMSG_ERR_UNSUPPORTED_TYPE, errMsg);
    }
    }

    return 0;
}

void MessageHandler::sendError(const uint32_t src, const EthosU::ethosu_rpmsg_err_type type, const char *msg) {
    auto message = new Message(src, EthosU::ETHOSU_RPMSG_ERR, 0, sizeof(EthosU::ethosu_rpmsg_err));

    message->rpmsg.error.type = type;

    for (size_t i = 0; i < sizeof(message->rpmsg.error.msg) && msg[i]; i++) {
        message->rpmsg.error.msg[i] = msg[i];
    }

    responseQueue.send(message);
}

void MessageHandler::sendPong(const uint32_t src, const uint64_t msgId) {
    auto message = new Message(src, EthosU::ETHOSU_RPMSG_PONG, msgId);

    responseQueue.send(message);
}

void MessageHandler::sendVersionRsp(const uint32_t src, const uint64_t msgId) {
    auto message = new Message(src, EthosU::ETHOSU_RPMSG_VERSION_RSP, msgId, sizeof(EthosU::ethosu_rpmsg_version_rsp));

    message->rpmsg.version_rsp = {
        ETHOSU_RPMSG_VERSION_MAJOR,
        ETHOSU_RPMSG_VERSION_MINOR,
        ETHOSU_RPMSG_VERSION_PATCH,
        0,
    };

    responseQueue.send(message);
}

void MessageHandler::sendCapabilitiesRsp(const uint32_t src, const uint64_t msgId) {
    auto message =
        new Message(src, EthosU::ETHOSU_RPMSG_CAPABILITIES_RSP, msgId, sizeof(EthosU::ethosu_rpmsg_capabilities_rsp));

    message->rpmsg.cap_rsp = capabilities;

    responseQueue.send(message);
}

EthosU::ethosu_rpmsg_capabilities_rsp MessageHandler::getCapabilities() const {
    EthosU::ethosu_rpmsg_capabilities_rsp cap = {};

#ifdef ETHOSU
    ethosu_driver_version version;
    ethosu_get_driver_version(&version);

    ethosu_hw_info info;
    ethosu_driver *drv = ethosu_reserve_driver();
    ethosu_get_hw_info(drv, &info);
    ethosu_release_driver(drv);

    cap.version_status     = info.version.version_status;
    cap.version_minor      = info.version.version_minor;
    cap.version_major      = info.version.version_major;
    cap.product_major      = info.version.product_major;
    cap.arch_patch_rev     = info.version.arch_patch_rev;
    cap.arch_minor_rev     = info.version.arch_minor_rev;
    cap.arch_major_rev     = info.version.arch_major_rev;
    cap.driver_patch_rev   = version.patch;
    cap.driver_minor_rev   = version.minor;
    cap.driver_major_rev   = version.major;
    cap.macs_per_cc        = info.cfg.macs_per_cc;
    cap.cmd_stream_version = info.cfg.cmd_stream_version;
    cap.custom_dma         = info.cfg.custom_dma;
#endif

    return cap;
}

void MessageHandler::sendNetworkInfoRsp(const uint32_t src,
                                        const uint64_t msgId,
                                        EthosU::ethosu_rpmsg_network_buffer &network) {
    auto message =
        new Message(src, EthosU::ETHOSU_RPMSG_NETWORK_INFO_RSP, msgId, sizeof(EthosU::ethosu_rpmsg_network_info_rsp));
    auto &rsp = message->rpmsg.net_info_rsp;

    rsp.ifm_count = 0;
    rsp.ofm_count = 0;

    bool failed = getNetwork(network);

    if (!failed) {
        InferenceProcess::InferenceParser parser;

        failed = parser.parseModel(reinterpret_cast<void *>(network.buffer.ptr),
                                   network.buffer.size,
                                   rsp.desc,
                                   InferenceProcess::makeArray(rsp.ifm_size, rsp.ifm_count, ETHOSU_RPMSG_BUFFER_MAX),
                                   InferenceProcess::makeArray(rsp.ofm_size, rsp.ofm_count, ETHOSU_RPMSG_BUFFER_MAX));
    }

    rsp.status = failed ? EthosU::ETHOSU_RPMSG_STATUS_ERROR : EthosU::ETHOSU_RPMSG_STATUS_OK;

    responseQueue.send(message);
}

void MessageHandler::forwardInferenceReq(const uint32_t src,
                                         const uint64_t msgId,
                                         const EthosU::ethosu_rpmsg_inference_req &inference) {
    auto message = new Message(src, EthosU::ETHOSU_RPMSG_INFERENCE_REQ, msgId);
    auto &req    = message->rpmsg.inf_req;

    req = inference;

    getNetwork(req.network);

    inferenceQueue.send(message);
}

void MessageHandler::sendInferenceRsp(const uint32_t src,
                                      const uint64_t msgId,
                                      const EthosU::ethosu_rpmsg_status status) {
    auto message =
        new Message(src, EthosU::ETHOSU_RPMSG_INFERENCE_RSP, msgId, sizeof(EthosU::ethosu_rpmsg_inference_rsp));

    message->rpmsg.inf_rsp.status = status;

    responseQueue.send(message);
}

void MessageHandler::sendCancelInferenceRsp(const uint32_t src,
                                            const uint64_t msgId,
                                            const EthosU::ethosu_rpmsg_status status) {
    auto message = new Message(
        src, EthosU::ETHOSU_RPMSG_CANCEL_INFERENCE_RSP, msgId, sizeof(EthosU::ethosu_rpmsg_cancel_inference_rsp));

    message->rpmsg.cancel_rsp.status = status;

    responseQueue.send(message);
}

bool MessageHandler::getNetwork(EthosU::ethosu_rpmsg_network_buffer &buffer) {
    switch (buffer.type) {
    case EthosU::ETHOSU_RPMSG_NETWORK_BUFFER:
        return false;
    case EthosU::ETHOSU_RPMSG_NETWORK_INDEX: {
        void *ptr;
        size_t size;
        if (getIndexedNetwork(buffer.index, ptr, size)) {
            return true;
        }

        buffer.type        = EthosU::ETHOSU_RPMSG_NETWORK_BUFFER;
        buffer.buffer.ptr  = reinterpret_cast<uint32_t>(ptr);
        buffer.buffer.size = size;

        return false;
    }
    default:
        LOG_WARN("Unsupported network model type. type=%" PRIu32, buffer.type);
        return true;
    }
}

void MessageHandler::responseTask(void *param) {
    auto _this = static_cast<MessageHandler *>(param);

    LOG_DEBUG("Starting message response task");

    while (true) {
        Message *message;
        auto ret = _this->responseQueue.receive(message);
        if (ret) {
            abort();
        }

        LOG_DEBUG("Sending message. type=%" PRIu32, message->rpmsg.header.type);

        _this->send(&message->rpmsg, sizeof(message->rpmsg.header) + message->length, message->src);

        delete message;
    }
}

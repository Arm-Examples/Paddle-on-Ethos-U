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

#include <cinttypes>
#include <memory>
#include <stdio.h>

#include <ethosu_log.h>
#include <mailbox.hpp>

#if defined(MHU_V2)
#include <mhu_v2.hpp>
#elif defined(MHU_JUNO)
#include <mhu_juno.hpp>
#else
#include <mhu_dummy.hpp>
#endif

#include "inference_runner.hpp"
#include "message_handler.hpp"
#include "remoteproc.hpp"

/*****************************************************************************
 * TFLM arena
 *****************************************************************************/

// Number of parallell inference tasks. Typically one per NPU.
#if defined(ETHOSU) && defined(ETHOSU_NPU_COUNT) && ETHOSU_NPU_COUNT > 0
constexpr size_t NUM_PARALLEL_TASKS = ETHOSU_NPU_COUNT;
#else
constexpr size_t NUM_PARALLEL_TASKS = 1;
#endif

#ifndef TENSOR_ARENA_SIZE
#define TENSOR_ARENA_SIZE 0x200000
#endif

// TensorArena static initialisation
constexpr size_t arenaSize = TENSOR_ARENA_SIZE;
static_assert(arenaSize && ((arenaSize & (arenaSize - 1)) == 0), "TENSOR_ARENA_SIZE must be a power of two");

/*****************************************************************************
 * Resource table
 *****************************************************************************/

#if defined(REMOTEPROC_TRACE_BUFFER)
#if defined(__ARMCC_VERSION)
extern uint32_t Image$$trace_buffer$$Base;
extern uint32_t Image$$trace_buffer$$Length;
#define __ethosu_core_trace_buffer_start__ Image$$trace_buffer$$Base
#define __ethosu_core_trace_buffer_size__  Image$$trace_buffer$$Length
#else
extern uint32_t __ethosu_core_trace_buffer_start__;
extern uint32_t __ethosu_core_trace_buffer_size__;
#endif
#endif

extern "C" {
// clang-format off
__attribute__((section(".resource_table"))) ResourceTable resourceTable = {
    // Table
    {
        ResourceTable::VERSION,
        ResourceTable::NUM_RESOURCES,
        {ResourceTable::RESERVED, ResourceTable::RESERVED},
        {}
    },
    // Offset
    {
        offsetof(ResourceTable, mapping),
#if defined(REMOTEPROC_TRACE_BUFFER)
        offsetof(ResourceTable, trace),
#endif
        offsetof(ResourceTable, vdev),
        offsetof(ResourceTable, carveout),
    },
    // Mappings
    {
        RSC_MAPPING,
        ResourceTable::NUM_RANGES,
        {}
    },
    // Ranges
    {
        { 0, 0, 0 },
        { 0, 0, 0 },
    },
    // Trace buffer
#if defined(REMOTEPROC_TRACE_BUFFER)
    {
        RSC_TRACE,
        reinterpret_cast<uint32_t>(&__ethosu_core_trace_buffer_start__),
        reinterpret_cast<uint32_t>(&__ethosu_core_trace_buffer_size__),
        ResourceTable::RESERVED,
        "Trace resource"
    },
#endif
    // VDEV
    {
        RSC_VDEV,
        VIRTIO_ID_RPMSG,
        2, // Notify ID
        1 << VIRTIO_RPMSG_F_NS,
        0,
        0,
        0,
        ResourceTable::NUM_VRINGS,
        {ResourceTable::RESERVED, ResourceTable::RESERVED},
        {}
    },
    // Vrings
    {
        {FW_RSC_U32_ADDR_ANY, ResourceTable::VRING_ALIGN, ResourceTable::VRING_SIZE, 1, ResourceTable::RESERVED},
        {FW_RSC_U32_ADDR_ANY, ResourceTable::VRING_ALIGN, ResourceTable::VRING_SIZE, 2, ResourceTable::RESERVED}
    },
    // Carveout
    {
        RSC_CARVEOUT,
        FW_RSC_U32_ADDR_ANY,
        FW_RSC_U32_ADDR_ANY,
        arenaSize * NUM_PARALLEL_TASKS,
        0,
        ResourceTable::RESERVED,
        "TFLM arena"
    }
};
// clang-format on
}

/*****************************************************************************
 * Mailbox
 *****************************************************************************/

namespace {

#ifdef MHU_V2
Mailbox::MHUv2 mailbox(MHU_TX_BASE_ADDRESS, MHU_RX_BASE_ADDRESS); // txBase, rxBase
#elif defined(MHU_JUNO)
Mailbox::MHUJuno mailbox(MHU_BASE_ADDRESS);
#else
Mailbox::MHUDummy mailbox;
#endif

#ifdef MHU_IRQ
void mailboxIrqHandler() {
    LOG_DEBUG("");
    mailbox.handleMessage();
}
#define MHU_IRQ_PRIORITY 5
#endif

} // namespace

/*****************************************************************************
 * main
 *****************************************************************************/

int main() {
    printf("Ethos-U Message Handler OpenAMP\n");

    auto rproc          = std::make_shared<RProc>(mailbox, resourceTable.table, sizeof(resourceTable));
    auto messageHandler = std::make_shared<MessageHandler>(*rproc, "ethos-u-0.0");

    printf("TFLM arena. pa=%" PRIx32 ", da=%" PRIx32 ", len=%" PRIx32 "\n",
           resourceTable.carveout.pa,
           resourceTable.carveout.da,
           resourceTable.carveout.len);

    std::array<std::shared_ptr<InferenceRunner>, NUM_PARALLEL_TASKS> inferenceRunner;

    for (size_t i = 0; i < NUM_PARALLEL_TASKS; i++) {
        auto tensorArena = reinterpret_cast<uint8_t *>(resourceTable.carveout.da);

        inferenceRunner[i] = std::make_shared<InferenceRunner>(&tensorArena[arenaSize * i],
                                                               arenaSize,
                                                               messageHandler->getInferenceQueue(),
                                                               messageHandler->getResponseQueue());
    }

#ifdef MHU_IRQ
    // Register mailbox interrupt handler
    NVIC_SetVector(static_cast<IRQn_Type>(MHU_IRQ), (uint32_t)&mailboxIrqHandler);
    NVIC_SetPriority(static_cast<IRQn_Type>(MHU_IRQ), MHU_IRQ_PRIORITY);
    NVIC_EnableIRQ(static_cast<IRQn_Type>(MHU_IRQ));
#endif

    // Start Scheduler
    vTaskStartScheduler();

    return 0;
}

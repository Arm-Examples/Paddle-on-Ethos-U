/*
 * SPDX-FileCopyrightText: Copyright 2022-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include <FreeRTOS.h>
#include <queue.h>
#include <semphr.h>

#include <metal/alloc.h>
#include <openamp/open_amp.h>
#include <openamp/remoteproc.h>

#include <mailbox.hpp>

/*****************************************************************************
 * Resource table
 *****************************************************************************/

static constexpr uint32_t RSC_MAPPING = RSC_VENDOR_START + 1;

/**
 * struct fw_rsc_map_range - memory map range
 * @da: Start device address of the memory address range
 * @pa: Start physical address of the memory address range
 * @len: length of memory address range
 *
 * Memory range to translate between physical and device addresses.
 */
METAL_PACKED_BEGIN
struct fw_rsc_map_range {
    uint32_t da;
    uint32_t pa;
    uint32_t len;
} METAL_PACKED_END;

/**
 * struct fw_rsc_mapping - memory map for address translation
 * @type: RSC_MAPPING
 * @num_ranges: Number of ranges in the memory map
 * @range: Array of the ranges in the memory map
 *
 * This resource entry requests the host to provide information for how to
 * translate between physical and device addresses.
 */
METAL_PACKED_BEGIN
struct fw_rsc_mapping {
    uint32_t type;
    uint8_t num_ranges;
    struct fw_rsc_map_range range[0];
} METAL_PACKED_END;

struct ResourceTable {
    static constexpr uint32_t VERSION = 1;
#if defined(REMOTEPROC_TRACE_BUFFER)
    static constexpr uint32_t NUM_RESOURCES = 4;
#else
    static constexpr uint32_t NUM_RESOURCES = 3;
#endif
    static constexpr uint32_t NUM_RANGES  = 2;
    static constexpr uint32_t NUM_VRINGS  = 2;
    static constexpr uint32_t VRING_ALIGN = 0x100;
    static constexpr uint32_t VRING_SIZE  = 0x08;
    static constexpr uint32_t RESERVED    = 0;

    resource_table table;
    uint32_t offset[NUM_RESOURCES];
    fw_rsc_mapping mapping;
    fw_rsc_map_range range[NUM_RANGES];
#if defined(REMOTEPROC_TRACE_BUFFER)
    fw_rsc_trace trace;
#endif
    fw_rsc_vdev vdev;
    fw_rsc_vdev_vring vring[NUM_VRINGS];
    fw_rsc_carveout carveout;
} __attribute__((packed));

/*****************************************************************************
 * RProc
 *****************************************************************************/

class RProc {
public:
    RProc(Mailbox::Mailbox &_mailbox, resource_table &table, size_t tableSize);
    ~RProc();

    remoteproc *getRProc();
    virtio_device *getVDev();

private:
    // IRQ notification callback
    static void mailboxCallback(void *userArg);

    // Notification task handling virtio messages
    static void notifyTask(void *param);

    // Remote proc ops
    static struct remoteproc *init(remoteproc *rproc, const remoteproc_ops *ops, void *arg);
    static void remove(remoteproc *rproc);
    static int handle_rsc(struct remoteproc *rproc, void *rsc, size_t len);
    static int notify(remoteproc *rproc, uint32_t id);

    // IRQ notification
    Mailbox::Mailbox &mailbox;

    // Remoteproc
    remoteproc rproc;
    remoteproc_ops ops;
    virtio_device *vdev;

    // Resource table memory region
    remoteproc_mem rsc_mem;
    metal_io_region rsc_region;

    // Host provided memory regions
    remoteproc_mem mems[ResourceTable::NUM_RANGES];
    metal_io_region regions[ResourceTable::NUM_RANGES];

    // FreeRTOS
    SemaphoreHandle_t notifySemaphore;
    TaskHandle_t notifyHandle;
};

/*****************************************************************************
 * Rpmsg
 *****************************************************************************/

class Rpmsg {
public:
    Rpmsg(RProc &rproc, const char *const name);

    int send(void *data, size_t len, uint32_t dst = 0);
    void *physicalToVirtual(metal_phys_addr_t pa);

protected:
    virtual int handleMessage(void *data, size_t len, uint32_t src);

private:
    // RPMsg ops
    static void rpmsgNsBind(rpmsg_device *rdev, const char *name, uint32_t dest);
    static void nsUnbindCallback(rpmsg_endpoint *ept);
    static int endpointCallback(rpmsg_endpoint *ept, void *data, size_t len, uint32_t src, void *priv);

    // RPMsg
    rpmsg_virtio_device rvdev;
    rpmsg_device *rdev;
    rpmsg_endpoint endpoint;
};

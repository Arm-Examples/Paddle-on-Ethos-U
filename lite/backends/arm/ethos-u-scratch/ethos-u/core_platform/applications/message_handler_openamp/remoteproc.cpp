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

#include "remoteproc.hpp"

#include <cinttypes>

#include <ethosu_log.h>

namespace {
void setupRProcMem(remoteproc &rproc,
                   metal_phys_addr_t pa,
                   metal_phys_addr_t da,
                   size_t size,
                   remoteproc_mem &mem,
                   metal_io_region &region) {

    remoteproc_init_mem(&mem, nullptr, pa, da, size, &region);
    metal_io_init(&region, (void *)da, &mem.pa, size, -1, 0, nullptr);
    remoteproc_add_mem(&rproc, &mem);
}
}; // namespace

/*****************************************************************************
 * RProc
 *****************************************************************************/

RProc::RProc(Mailbox::Mailbox &_mailbox, resource_table &table, size_t tableSize) :
    mailbox(_mailbox), ops({}), vdev(nullptr), rsc_mem(), rsc_region(), mems(), regions(),
    notifySemaphore(xSemaphoreCreateBinary()), notifyHandle() {
    ops.init       = init;       // initialize the remoteproc instance
    ops.remove     = remove;     // remove the remoteproc instance
    ops.handle_rsc = handle_rsc; // handle the vendor specific resource
    ops.notify     = notify;     // notify the remote
    mailbox.registerCallback(mailboxCallback, static_cast<void *>(this));

    if (!remoteproc_init(&rproc, &ops, this)) {
        LOG_ERR("Failed to intialize remoteproc");
        abort();
    }

    // Setup memory region for resource table
    const metal_phys_addr_t rsc_addr = reinterpret_cast<metal_phys_addr_t>(&table);
    // No translation is needed for rsc_addr because it already contains the DA
    // so PA is set to DA
    setupRProcMem(rproc, rsc_addr, rsc_addr, tableSize, rsc_mem, rsc_region);

    int ret = remoteproc_set_rsc_table(&rproc, &table, tableSize);
    if (ret) {
        LOG_ERR("Failed to set resource table. ret=%d", ret);
        abort();
    }

    vdev = remoteproc_create_virtio(&rproc, 0, VIRTIO_DEV_DEVICE, nullptr);
    if (!vdev) {
        LOG_ERR("Failed to create vdev");
        abort();
    }

    BaseType_t taskret = xTaskCreate(notifyTask, "notifyTask", 1024, this, 2, &notifyHandle);
    if (taskret != pdPASS) {
        LOG_ERR("Failed to create remoteproc notify task");
        abort();
    }
}

RProc::~RProc() {
    mailbox.deregisterCallback(mailboxCallback, static_cast<void *>(this));
    vTaskDelete(notifyHandle);
    vSemaphoreDelete(notifySemaphore);
}

remoteproc *RProc::getRProc() {
    return &rproc;
}

virtio_device *RProc::getVDev() {
    return vdev;
}

void RProc::mailboxCallback(void *userArg) {
    auto _this = static_cast<RProc *>(userArg);

    xSemaphoreGiveFromISR(_this->notifySemaphore, nullptr);
}

void RProc::notifyTask(void *param) {
    LOG_DEBUG("Starting message notify task");

    auto _this = static_cast<RProc *>(param);

    while (true) {
        // Wait for event
        xSemaphoreTake(_this->notifySemaphore, portMAX_DELAY);

        // Read virtio queue and notify all rpmsg clients
        rproc_virtio_notified(_this->vdev, RSC_NOTIFY_ID_ANY);
    }
}

struct remoteproc *RProc::init(remoteproc *rproc, const remoteproc_ops *ops, void *arg) {
    LOG_DEBUG("");

    rproc->ops  = ops;
    rproc->priv = arg;

    return rproc;
}

int RProc::handle_rsc(remoteproc *rproc, void *rsc, size_t len) {
    auto _this                     = static_cast<RProc *>(rproc->priv);
    struct fw_rsc_mapping *mapping = static_cast<fw_rsc_mapping *>(rsc);

    if (mapping->type != RSC_MAPPING) {
        LOG_ERR("Unknown resource type %" PRIu32, mapping->type);
        return -RPROC_ERR_RSC_TAB_NS;
    }

    for (uint32_t i = 0; i < mapping->num_ranges; ++i) {
        const fw_rsc_map_range *range = &mapping->range[i];
        if (range->len == 0) {
            LOG_DEBUG("Ignored zero length memory map range[%" PRIu32 "]", i);
            continue;
        }
        setupRProcMem(*rproc, range->pa, range->da, range->len, _this->mems[i], _this->regions[i]);
    }

    return 0;
}

void RProc::remove(remoteproc *rproc) {
    LOG_DEBUG("");
}

int RProc::notify(remoteproc *rproc, uint32_t id) {
    LOG_DEBUG("");

    auto *_this = static_cast<RProc *>(rproc->priv);
    _this->mailbox.sendMessage();
    return 0;
}

/*****************************************************************************
 * Rpmsg
 *****************************************************************************/

Rpmsg::Rpmsg(RProc &rproc, const char *const name) : rvdev({}), rdev(), endpoint({}) {
    struct virtio_device *vdev = rproc.getVDev();

    if (vdev->vrings_num != ResourceTable::NUM_VRINGS) {
        LOG_ERR("Invalid number of vrings");
        abort();
    }

    // Vdev can use the same IO region for translations as the vring
    if (rpmsg_init_vdev(&rvdev, vdev, nullptr, vdev->vrings_info[0].io, nullptr)) {
        LOG_ERR("Failed to initialize rpmsg vdev");
        abort();
    }

    rdev = rpmsg_virtio_get_rpmsg_device(&rvdev);
    if (!rdev) {
        LOG_ERR("Failed to get rpmsg dev");
        abort();
    }

    int ret =
        rpmsg_create_ept(&endpoint, rdev, name, RPMSG_ADDR_ANY, RPMSG_ADDR_ANY, endpointCallback, nsUnbindCallback);
    if (ret != RPMSG_SUCCESS) {
        LOG_ERR("Failed to create rpmsg endpoint. ret=%d", ret);
        abort();
    }

    endpoint.priv = static_cast<void *>(this);
}

int Rpmsg::send(void *data, size_t len, uint32_t dst) {
    LOG_DEBUG("Sending rpmsg. dst=%" PRIu32 ", len=%zu", dst, len);

    int ret = rpmsg_sendto(&endpoint, data, len, dst);
    return ret;
}

void *Rpmsg::physicalToVirtual(metal_phys_addr_t pa) {
    return metal_io_phys_to_virt(rvdev.shbuf_io, pa);
}

void Rpmsg::rpmsgNsBind(rpmsg_device *rdev, const char *name, uint32_t dest) {
    LOG_DEBUG("");
}

void Rpmsg::nsUnbindCallback(rpmsg_endpoint *ept) {
    LOG_DEBUG("");
}

int Rpmsg::endpointCallback(rpmsg_endpoint *ept, void *data, size_t len, uint32_t src, void *priv) {
    LOG_DEBUG("src=%" PRIX32 ", len=%zu", src, len);

    auto _this = static_cast<Rpmsg *>(priv);
    _this->handleMessage(data, len, src);

    return 0;
}

int Rpmsg::handleMessage(void *data, size_t len, uint32_t src) {
    LOG_DEBUG("Receiving rpmsg. src=%" PRIu32 ", len=%zu", src, len);

    auto c = static_cast<char *>(data);
    for (size_t i = 0; i < len; i++) {
        printf("%c", c[i]);
    }
    printf("\n");

    return 0;
}

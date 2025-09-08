/*
 * SPDX-FileCopyrightText: Copyright 2022-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
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

#include <cstdlib>
#include <functional>

#include <ethosu_log.h>

/*****************************************************************************
 * Queue
 *****************************************************************************/

template <typename T>
class Queue {
public:
    using Predicate = std::function<bool(const T &data)>;

    Queue(const size_t size = 10) : queue(xQueueCreate(size, sizeof(T))) {}

    ~Queue() {
        vQueueDelete(queue);
    }

    int send(const T &msg, TickType_t delay = portMAX_DELAY) {
        if (pdPASS != xQueueSend(queue, &msg, delay)) {
            LOG_ERR("Failed to send message");
            return -1;
        }

        return 0;
    }

    int receive(T &msg, TickType_t delay = portMAX_DELAY) {
        if (pdTRUE != xQueueReceive(queue, &msg, delay)) {
            LOG_ERR("Failed to receive message");
            return -1;
        }

        return 0;
    }

    void erase(Predicate pred) {
        const size_t count = uxQueueMessagesWaiting(queue);
        for (size_t i = 0; i < count; i++) {
            T data;

            if (pdPASS != xQueueReceive(queue, &data, 0)) {
                LOG_ERR("Failed to dequeue message");
                abort();
            }

            if (!pred(data)) {
                if (pdPASS != xQueueSend(queue, &data, 0)) {
                    LOG_ERR("Failed to requeue message");
                    abort();
                }
            }
        }
    }

private:
    QueueHandle_t queue;
};

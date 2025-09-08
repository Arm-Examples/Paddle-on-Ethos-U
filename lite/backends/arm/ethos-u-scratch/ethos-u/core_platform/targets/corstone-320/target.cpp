/*
 * SPDX-FileCopyrightText: Copyright 2020-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

/****************************************************************************
 * Includes
 ****************************************************************************/

#include "target.hpp"

#ifdef ETHOSU
#include <ethosu_driver.h>
#include <timing_adapter.h>
#endif

#include "mpu.hpp"
#include "uart_stdout.h"

#include <cstring>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace EthosU;

/****************************************************************************
 * Defines
 ****************************************************************************/

#define ETHOSU_BASE_ADDRESS 0x50004000
#define ETHOSU_IRQ          16
#define ETHOSU_IRQ_PRIORITY 5

#define ETHOSU0_TA0_BASE_ADDRESS 0x50006000 /* AXI SRAM */
#define ETHOSU0_TA1_BASE_ADDRESS 0x50006800 /* AXI EXT */

#define ETHOSU_NUM_TA 2

#define VTABLE_SIZE 496

/****************************************************************************
 * Variables
 ****************************************************************************/

/* CMSIS default vector table. Originally in boot ROM (RO mem) */
extern uint32_t __VECTOR_TABLE[VTABLE_SIZE];

/* Vector table remapping to RW mem required as NVIC_SetVector modifies it */
uint32_t vtable_rw[VTABLE_SIZE] __attribute__((used, section(".vtable_rw")));

#if defined(ETHOSU_FAST_MEMORY_SIZE) && ETHOSU_FAST_MEMORY_SIZE > 0
__attribute__((aligned(16), section(".bss.ethosu_scratch"))) uint8_t ethosu_scratch[ETHOSU_FAST_MEMORY_SIZE];
#else
#define ethosu_scratch          0
#define ETHOSU_FAST_MEMORY_SIZE 0
#endif

#ifdef ETHOSU
struct ethosu_driver ethosu0_driver;

/****************************************************************************
 * Timing Adapters
 ****************************************************************************/

#ifndef ETHOSU_TA_MAXR_0
#define ETHOSU_TA_MAXR_0 0
#endif

#ifndef ETHOSU_TA_MAXW_0
#define ETHOSU_TA_MAXW_0 0
#endif

#ifndef ETHOSU_TA_MAXRW_0
#define ETHOSU_TA_MAXRW_0 0
#endif

#ifndef ETHOSU_TA_RLATENCY_0
#define ETHOSU_TA_RLATENCY_0 0
#endif

#ifndef ETHOSU_TA_WLATENCY_0
#define ETHOSU_TA_WLATENCY_0 0
#endif

#ifndef ETHOSU_TA_PULSE_ON_0
#define ETHOSU_TA_PULSE_ON_0 0
#endif

#ifndef ETHOSU_TA_PULSE_OFF_0
#define ETHOSU_TA_PULSE_OFF_0 0
#endif

#ifndef ETHOSU_TA_BWCAP_0
#define ETHOSU_TA_BWCAP_0 0
#endif

#ifndef ETHOSU_TA_PERFCTRL_0
#define ETHOSU_TA_PERFCTRL_0 0
#endif

#ifndef ETHOSU_TA_PERFCNT_0
#define ETHOSU_TA_PERFCNT_0 0
#endif

#ifndef ETHOSU_TA_MODE_0
#define ETHOSU_TA_MODE_0 1
#endif

#ifndef ETHOSU_TA_HISTBIN_0
#define ETHOSU_TA_HISTBIN_0 0
#endif

#ifndef ETHOSU_TA_HISTCNT_0
#define ETHOSU_TA_HISTCNT_0 0
#endif

#ifndef ETHOSU_TA_MAXR_1
#define ETHOSU_TA_MAXR_1 0
#endif

#ifndef ETHOSU_TA_MAXW_1
#define ETHOSU_TA_MAXW_1 0
#endif

#ifndef ETHOSU_TA_MAXRW_1
#define ETHOSU_TA_MAXRW_1 0
#endif

#ifndef ETHOSU_TA_RLATENCY_1
#define ETHOSU_TA_RLATENCY_1 0
#endif

#ifndef ETHOSU_TA_WLATENCY_1
#define ETHOSU_TA_WLATENCY_1 0
#endif

#ifndef ETHOSU_TA_PULSE_ON_1
#define ETHOSU_TA_PULSE_ON_1 0
#endif

#ifndef ETHOSU_TA_PULSE_OFF_1
#define ETHOSU_TA_PULSE_OFF_1 0
#endif

#ifndef ETHOSU_TA_BWCAP_1
#define ETHOSU_TA_BWCAP_1 0
#endif

#ifndef ETHOSU_TA_PERFCTRL_1
#define ETHOSU_TA_PERFCTRL_1 0
#endif

#ifndef ETHOSU_TA_PERFCNT_1
#define ETHOSU_TA_PERFCNT_1 0
#endif

#ifndef ETHOSU_TA_MODE_1
#define ETHOSU_TA_MODE_1 1
#endif

#ifndef ETHOSU_TA_HISTBIN_1
#define ETHOSU_TA_HISTBIN_1 0
#endif

#ifndef ETHOSU_TA_HISTCNT_1
#define ETHOSU_TA_HISTCNT_1 0
#endif

#ifdef ETHOSU_HAS_TA
static uintptr_t ethosu_ta_base_addrs[ETHOSU_NUM_TA] = {ETHOSU0_TA0_BASE_ADDRESS, ETHOSU0_TA1_BASE_ADDRESS};
struct timing_adapter ethosu_ta[ETHOSU_NUM_TA];
struct timing_adapter_settings ethosu_ta_settings[ETHOSU_NUM_TA] = {{ETHOSU_TA_MAXR_0,
                                                                     ETHOSU_TA_MAXW_0,
                                                                     ETHOSU_TA_MAXRW_0,
                                                                     ETHOSU_TA_RLATENCY_0,
                                                                     ETHOSU_TA_WLATENCY_0,
                                                                     ETHOSU_TA_PULSE_ON_0,
                                                                     ETHOSU_TA_PULSE_OFF_0,
                                                                     ETHOSU_TA_BWCAP_0,
                                                                     ETHOSU_TA_PERFCTRL_0,
                                                                     ETHOSU_TA_PERFCNT_0,
                                                                     ETHOSU_TA_MODE_0,
                                                                     0, // Read only register
                                                                     ETHOSU_TA_HISTBIN_0,
                                                                     ETHOSU_TA_HISTCNT_0},
                                                                    {ETHOSU_TA_MAXR_1,
                                                                     ETHOSU_TA_MAXW_1,
                                                                     ETHOSU_TA_MAXRW_1,
                                                                     ETHOSU_TA_RLATENCY_1,
                                                                     ETHOSU_TA_WLATENCY_1,
                                                                     ETHOSU_TA_PULSE_ON_1,
                                                                     ETHOSU_TA_PULSE_OFF_1,
                                                                     ETHOSU_TA_BWCAP_1,
                                                                     ETHOSU_TA_PERFCTRL_1,
                                                                     ETHOSU_TA_PERFCNT_1,
                                                                     ETHOSU_TA_MODE_1,
                                                                     0, // Read only register
                                                                     ETHOSU_TA_HISTBIN_1,
                                                                     ETHOSU_TA_HISTCNT_1}};
#endif // ETHOSU_HAS_TA
#endif // ETHOSU

/****************************************************************************
 * Cache maintenance
 ****************************************************************************/

#if defined(CPU_CACHE_ENABLE) && defined(__DCACHE_PRESENT) && (__DCACHE_PRESENT == 1U)
extern "C" {
void ethosu_flush_dcache(uint32_t *p, size_t bytes) {
    if (p)
        SCB_CleanDCache_by_Addr(p, bytes);
    else
        SCB_CleanDCache();
}

void ethosu_invalidate_dcache(uint32_t *p, size_t bytes) {
    if (p)
        SCB_InvalidateDCache_by_Addr(p, bytes);
    else
        SCB_InvalidateDCache();
}
}
#endif

/****************************************************************************
 * Init
 ****************************************************************************/

namespace {

extern "C" {
struct ExcContext {
    uint32_t r0;
    uint32_t r1;
    uint32_t r2;
    uint32_t r3;
    uint32_t r12;
    uint32_t lr;
    uint32_t pc;
    uint32_t xPsr;
};

void HardFault_Handler() {
    int irq;
    struct ExcContext *e;
    uint32_t sp;

    asm volatile("mrs %0, ipsr            \n" // Read IPSR (Exception number)
                 "sub %0, #16             \n" // Get it into IRQn_Type range
                 "tst lr, #4              \n" // Select the stack which was in use
                 "ite eq                  \n"
                 "mrseq %1, msp           \n"
                 "mrsne %1, psp           \n"
                 "mov %2, sp              \n"
                 : "=r"(irq), "=r"(e), "=r"(sp));

    printf("Hard fault. irq=%d, pc=0x%08" PRIx32 ", lr=0x%08" PRIx32 ", xpsr=0x%08" PRIx32 ", sp=0x%08" PRIx32 "\n",
           irq,
           e->pc,
           e->lr,
           e->xPsr,
           sp);
    printf(
        "%11s cfsr=0x%08" PRIx32 " bfar=0x%08" PRIx32 " mmfar=0x%08" PRIx32 "\n", "", SCB->CFSR, SCB->BFAR, SCB->MMFAR);
    exit(1);
}
}

#ifdef ETHOSU
void ethosuIrqHandler() {
    ethosu_irq_handler(&ethosu0_driver);
}
#endif

} // namespace

namespace EthosU {

void targetSetup() {
    // Initialize UART driver
    UartStdOutInit();

    // Relocate vector table to RW ITCM memory
    memcpy(vtable_rw, __VECTOR_TABLE, sizeof(VECTOR_TABLE_Type) * VTABLE_SIZE);
    __disable_irq();
    SCB->VTOR = (uint32_t)&vtable_rw;
    __DSB();
    __ISB();
    __enable_irq();

#ifdef ETHOSU
#ifdef ETHOSU_HAS_TA
    // Initialize timing adapter(s)
    for (int i = 0; i < ETHOSU_NUM_TA; i++) {
        if (ta_init(&ethosu_ta[i], ethosu_ta_base_addrs[i])) {
            printf("Failed to initialize timing-adapter #%d\n", i);
        } else {
            // Set the updated configuration
            ta_set_all(&ethosu_ta[i], &ethosu_ta_settings[i]);
        }
    }
#endif

    // Initialize Ethos-U NPU driver
    if (ethosu_init(&ethosu0_driver,
                    reinterpret_cast<void *>(ETHOSU_BASE_ADDRESS),
                    ethosu_scratch,
                    ETHOSU_FAST_MEMORY_SIZE,
                    1,
                    1)) {
        printf("Failed to initialize NPU.\n");
        return;
    }

    // Assumes SCB->VTOR point to RW memory
    NVIC_SetVector(static_cast<IRQn_Type>(ETHOSU_IRQ), (uint32_t)&ethosuIrqHandler);
    NVIC_SetPriority(static_cast<IRQn_Type>(ETHOSU_IRQ), ETHOSU_IRQ_PRIORITY);
    NVIC_EnableIRQ(static_cast<IRQn_Type>(ETHOSU_IRQ));
#endif

    // MPU example setup
    const std::vector<ARM_MPU_Region_t> mpuConfig = {
        {
            // ITCM (S)
            ARM_MPU_RBAR(0x10000000,      // Base
                         ARM_MPU_SH_NON,  // Non-shareable
                         0,               // Read-Write
                         1,               // Non-Privileged
                         0),              // eXecute Never disabled
            ARM_MPU_RLAR(0x10007fff,      // Limit
                         Mpu::WTRA_index) // Attribute index - Write-Through, Read-allocate
        },
        {
            // FPGA DATA SRAM; BRAM (S)
            ARM_MPU_RBAR(0x12000000,        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-Write
                         1,                 // Non-Privileged
                         0),                // eXecute Never disabled
            ARM_MPU_RLAR(0x121FFFFF,        // Limit
                         Mpu::WBWARA_index) // Attribute index - Write-Back, Write-Allocate, Read-allocate
        },
        {
            // DTCM (S)
            ARM_MPU_RBAR(0x30000000,        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-Write
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR(0x30007fff,        // Limit
                         Mpu::WBWARA_index) // Attribute index - Write-Back, Write-Allocate, Read-allocate
        },
        {
            // Internal SRAM (S)
            ARM_MPU_RBAR(0x31000000,        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-Write
                         1,                 // Non-Privileged
                         0),                // eXecute Never disabled
            ARM_MPU_RLAR(0x313fffff,        // Limit
                         Mpu::WTWARA_index) // Attribute index - Write-Through, Write-Allocate, Read-allocate
        },
        {
            // QSPI Flash (S)
            ARM_MPU_RBAR(0x38000000,        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-Write
                         1,                 // Non-Privileged
                         0),                // eXecute Never disabled
            ARM_MPU_RLAR(0x3fffffff,        // Limit
                         Mpu::WTWARA_index) // Attribute index - Write-Through, Write-Allocate, Read-allocate
        },
        {
            // DDR1 (S)
            ARM_MPU_RBAR(0x70000000,        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-Write
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR(0x7fffffff,        // Limit
                         Mpu::WBWARA_index) // Attribute index - Write-Back, Write-Allocate, Read-allocate
        }};

    // Setup MPU configuration
    Mpu::loadAndEnableConfig(&mpuConfig[0], mpuConfig.size());

#if defined(CPU_CACHE_ENABLE) && defined(__DCACHE_PRESENT) && (__DCACHE_PRESENT == 1U)
    SCB_EnableICache();
    SCB_EnableDCache();
#endif
}

} // namespace EthosU

/* Copyright 2024 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cinttypes>
#include <vector>
#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <memory>
#include <vector>

#include "arm_perf_monitor.h"

#ifdef ETHOSU
#include <ethosu_driver.h>
#include <pmu_ethosu.h>

static uint32_t ethosu_inference_count = 0;
static uint64_t ethosu_ArmBackendExecuteCycleCountStart = 0;
static uint64_t ethosu_ArmBackendExecuteCycleCount = 0;
static uint64_t ethosu_ArmWhenNPURunCycleCountStart = 0;
static uint64_t ethosu_ArmWhenNPURunCycleCount = 0;
static uint64_t ethosu_pmuCycleCount = 0;
static std::vector<uint64_t> ethosu_pmuEventCounts(
    ETHOSU_PMU_Get_NumEventCounters(),
    0);

#if defined(ETHOSU55) || defined(ETHOSU65)
static const uint32_t ethosu_pmuCountersUsed = 4;
#elif defined(ETHOSU85)
static const uint32_t ethosu_pmuCountersUsed = 5;
#else
#error No NPU target defined
#endif

// ethosu_pmuCountersUsed should match numbers of counters setup in
// ethosu_inference_begin() and not be more then the HW supports
static_assert(ETHOSU_PMU_NCOUNTERS >= ethosu_pmuCountersUsed);

extern "C" {

// Callback invoked at start of NPU execution
void ethosu_inference_begin(struct ethosu_driver* drv, void*) {
  // Enable PMU
  ETHOSU_PMU_Enable(drv);
  ETHOSU_PMU_PMCCNTR_CFG_Set_Stop_Event(drv, ETHOSU_PMU_NPU_IDLE);
  ETHOSU_PMU_PMCCNTR_CFG_Set_Start_Event(drv, ETHOSU_PMU_NPU_ACTIVE);

  // Setup 4 counters
#if defined(ETHOSU55) || defined(ETHOSU65)
  ETHOSU_PMU_Set_EVTYPER(drv, 0, ETHOSU_PMU_AXI0_RD_DATA_BEAT_RECEIVED);
  ETHOSU_PMU_Set_EVTYPER(drv, 1, ETHOSU_PMU_AXI1_RD_DATA_BEAT_RECEIVED);
  ETHOSU_PMU_Set_EVTYPER(drv, 2, ETHOSU_PMU_AXI0_WR_DATA_BEAT_WRITTEN);
  ETHOSU_PMU_Set_EVTYPER(drv, 3, ETHOSU_PMU_NPU_IDLE);
  // Enable the 4 counters
  ETHOSU_PMU_CNTR_Enable(
      drv,
      ETHOSU_PMU_CNT1_Msk | ETHOSU_PMU_CNT2_Msk | ETHOSU_PMU_CNT3_Msk |
          ETHOSU_PMU_CNT4_Msk);
#elif defined(ETHOSU85)
  ETHOSU_PMU_Set_EVTYPER(drv, 0, ETHOSU_PMU_SRAM_RD_DATA_BEAT_RECEIVED);
  ETHOSU_PMU_Set_EVTYPER(drv, 1, ETHOSU_PMU_SRAM_WR_DATA_BEAT_WRITTEN);
  ETHOSU_PMU_Set_EVTYPER(drv, 2, ETHOSU_PMU_EXT_RD_DATA_BEAT_RECEIVED);
  ETHOSU_PMU_Set_EVTYPER(drv, 3, ETHOSU_PMU_EXT_WR_DATA_BEAT_WRITTEN);
  ETHOSU_PMU_Set_EVTYPER(drv, 4, ETHOSU_PMU_NPU_IDLE);
  // Enable the 5 counters
  ETHOSU_PMU_CNTR_Enable(
      drv,
      ETHOSU_PMU_CNT1_Msk | ETHOSU_PMU_CNT2_Msk | ETHOSU_PMU_CNT3_Msk |
          ETHOSU_PMU_CNT4_Msk | ETHOSU_PMU_CNT5_Msk);
#else
#error No NPU target defined
#endif

  ETHOSU_PMU_CNTR_Enable(drv, ETHOSU_PMU_CCNT_Msk);
  ETHOSU_PMU_CYCCNT_Reset(drv);

  // Reset all counters
  ETHOSU_PMU_EVCNTR_ALL_Reset(drv);

  // Save Cortex-M cycle clock to calculate total CPU cycles used in
  // ethosu_inference_end()
  ethosu_ArmWhenNPURunCycleCountStart = ARM_PMU_Get_CCNTR();
}

// Callback invoked at end of NPU execution
void ethosu_inference_end(struct ethosu_driver* drv, void*) {
  ethosu_inference_count++;
  ethosu_pmuCycleCount += ETHOSU_PMU_Get_CCNTR(drv);

  for (size_t i = 0; i < ethosu_pmuCountersUsed; i++) {
    ethosu_pmuEventCounts[i] += ETHOSU_PMU_Get_EVCNTR(drv, i);
  }
  ETHOSU_PMU_Disable(drv);
  // Add Cortex-M cycle clock used during this NPU execution
  ethosu_ArmWhenNPURunCycleCount +=
      (ARM_PMU_Get_CCNTR() - ethosu_ArmWhenNPURunCycleCountStart);
}

// Callback invoked at start of ArmBackend::execute()
void ArmBackend_execute_begin() {
  // Save Cortex-M cycle clock to calculate total CPU cycles used in
  // ArmBackend_execute_end()
  ethosu_ArmBackendExecuteCycleCountStart = ARM_PMU_Get_CCNTR();
}

// Callback invoked at end of ArmBackend::execute()
void ArmBackend_execute_end() {
  // Add Cortex-M cycle clock used during this ArmBackend::execute()
  ethosu_ArmBackendExecuteCycleCount +=
      (ARM_PMU_Get_CCNTR() - ethosu_ArmBackendExecuteCycleCountStart);
}
}

void StartMeasurements() {
  ethosu_ArmBackendExecuteCycleCount = 0;
  ethosu_ArmWhenNPURunCycleCount = 0;
  ethosu_pmuCycleCount = 0;

  for (size_t i = 0; i < ethosu_pmuCountersUsed; i++) {
    ethosu_pmuEventCounts[i] = 0;
  }
  ARM_PMU_Enable();
  DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk; // Trace enable
  ARM_PMU_CYCCNT_Reset();
  ARM_PMU_CNTR_Enable(PMU_CNTENSET_CCNTR_ENABLE_Msk);
}

void StopMeasurements() {
  ARM_PMU_CNTR_Disable(
      PMU_CNTENCLR_CCNTR_ENABLE_Msk | PMU_CNTENCLR_CNT0_ENABLE_Msk |
      PMU_CNTENCLR_CNT1_ENABLE_Msk);
  uint32_t cycle_count = ARM_PMU_Get_CCNTR();

  // Number of comand streams handled by the NPU
  printf("============ NPU Inferences : %d ============\n", ethosu_inference_count);
  printf("Profiler report, CPU cycles per operator:\n");
  // This is number of CPU cycles for the ethos-u operator from start to finish
  // in the framework If there is more then one commandstream the time is added
  // together
  printf(
      "ethos-u : cycle_cnt : %d cycles\n",
      ethosu_ArmBackendExecuteCycleCount);
  // We could print a list of the cycles used by the other delegates here in the
  // future but now we only print ethos-u: this means that "Operator(s) total:
  // ..." will be the same number as ethos-u : cycle_cnt and not the sum of all
  printf(
      "Operator(s) total: %d CPU cycles\n",
      ethosu_ArmBackendExecuteCycleCount);
  // Total CPU cycles used in the executorch method->execute()
  // Other delegates and no delegates are counted in this
  printf("Inference runtime: %d CPU cycles total\n", cycle_count);

  printf(
      "NOTE: CPU cycle values and ratio calculations require FPGA and identical CPU/NPU frequency\n");

  // Avoid division with zero if ARM_PMU_Get_CCNTR() is not enabled properly.
  if (cycle_count == 0) {
    printf("Inference CPU ratio: ?.??\n");
    printf("Inference NPU ratio: ?.??\n");
  } else {
    printf(
        "Inference CPU ratio: %.2f\n",
        100.0 * (cycle_count - ethosu_ArmWhenNPURunCycleCount) / cycle_count);
    printf(
        "Inference NPU ratio: %.2f\n",
        100.0 * ethosu_ArmWhenNPURunCycleCount / cycle_count);
  }

  // CPU cycles used by NPU, e.g. number of CPU cycles used between
  // ethosu_inference_begin() and ethosu_inference_end()
  // If there is more then one commandstream the time is added together
  printf(
         "cpu_wait_for_npu_cntr : %ld CPU cycles\n", ethosu_ArmWhenNPURunCycleCount);

  printf("Ethos-U PMU report:\n");
  printf("ethosu_pmu_cycle_cntr : %ld\n", ethosu_pmuCycleCount);

  for (size_t i = 0; i < ethosu_pmuCountersUsed; i++) {
    printf("ethosu_pmu_cntr%zd : %ld \n", i, ethosu_pmuEventCounts[i]);
  }
#if defined(ETHOSU55) || defined(ETHOSU65)
  printf("Ethos-U PMU Events:[ETHOSU_PMU_AXI0_RD_DATA_BEAT_RECEIVED, ");
  printf("ETHOSU_PMU_AXI1_RD_DATA_BEAT_RECEIVED, ETHOSU_PMU_AXI0_WR_DATA_BEAT_WRITTEN, ");
  printf("ETHOSU_PMU_NPU_IDLE]\n");
#elif defined(ETHOSU85)
  printf("Ethos-U PMU Events:[ETHOSU_PMU_SRAM_RD_DATA_BEAT_RECEIVED, ");
  printf("ETHOSU_PMU_SRAM_WR_DATA_BEAT_WRITTEN, ETHOSU_PMU_EXT_RD_DATA_BEAT_RECEIVED, ");
  printf("ETHOSU_PMU_EXT_WR_DATA_BEAT_WRITTEN, ETHOSU_PMU_NPU_IDLE]\n");
#else
#error No NPU target defined
#endif
  printf("============ Measurements end ============\n");
}

#else
void StartMeasurements() {}

void StopMeasurements() {}

#endif

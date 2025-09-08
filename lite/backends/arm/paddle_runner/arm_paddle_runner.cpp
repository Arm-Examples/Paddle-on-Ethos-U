/* Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2023-2024 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <ethosu_driver.h>

#include <memory>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstring>
#include "arm_perf_monitor.h"
#include "paddle_api.h"
#include "mobilenet_v1_opt.h"

#include "VelaBinStream.h"
#include "vela.h"
#include "input_tensor.h"

#if MODEL_FLAG == 7
#include "vela2.h"
#include "vela3.h"
#include "vela4.h"
#endif

#if MODEL_FLAG == 6
// ppocr_rec
#include "rec_postprocess.h"
#endif

#if MODEL_FLAG == 3
extern void int82float(int8_t * input, float * output, int size, float scale);
extern std::vector<std::unique_ptr<float[]>> process_conv_outputs(
    float *conv_output_1x52x52x1,
    float *conv_output_1x52x52x32,
    float *conv_output_1x52x52x80,
    float *conv_output_1x26x26x1,
    float *conv_output_1x26x26x32,
    float *conv_output_1x26x26x80,
    float *conv_output_1x13x13x1,
    float *conv_output_1x13x13x32,
    float *conv_output_1x13x13x80,
    float *conv_output_1x7x7x1,
    float *conv_output_1x7x7x32,
    float *conv_output_1x7x7x80);
#endif

#if MODEL_FLAG==5
extern std::vector<int8_t> layout_conv_preprocess(
    const std::vector<float>& normalized_input,
    int batch_size = 1,
    int height = 800,
    int width = 608);
#endif

float g_output[1000];
using namespace paddle::lite_api;  // NOLINT
void softmax(float* input, float* output, int length) {
    float max_val = input[0];
    float sum = 0.0f;
    for (int i = 1; i < length; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    for (int i = 0; i < length; i++) {
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < length; i++) {
        output[i] = output[i]/sum;
    }
}

int64_t ShapeProduction(const shape_t& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

std::string ShapePrint(const std::vector<shape_t>& shapes) {
  std::string shapes_str{""};
  for (size_t shape_idx = 0; shape_idx < shapes.size(); ++shape_idx) {
    auto shape = shapes[shape_idx];
    std::string shape_str;
    for (auto i : shape) {
      shape_str += std::to_string(i) + ",";
    }
    shapes_str += shape_str;
    shapes_str +=
        (shape_idx != 0 && shape_idx == shapes.size() - 1) ? "" : " : ";
  }
  return shapes_str;
}

std::string ShapePrint(const shape_t& shape) {
  std::string shape_str{""};
  for (auto i : shape) {
    shape_str += std::to_string(i) + " ";
  }
  return shape_str;
}

std::vector<std::string> split_string(const std::string& str_in) {
  std::vector<std::string> str_out;
  std::string tmp_str = str_in;
  while (!tmp_str.empty()) {
    size_t next_offset = tmp_str.find(":");
    str_out.push_back(tmp_str.substr(0, next_offset));
    if (next_offset == std::string::npos) {
      break;
    } else {
      tmp_str = tmp_str.substr(next_offset + 1);
    }
  }
  return str_out;
}

template <typename T>
double compute_mean(const T* in, const size_t length) {
  double sum = 0.;
  for (size_t i = 0; i < length; ++i) {
    sum += in[i];
  }
  return sum / length;
}

template <typename T>
double compute_standard_deviation(const T* in,
                                  const size_t length,
                                  bool has_mean = false,
                                  double mean = 10000) {
  if (!has_mean) {
    mean = compute_mean<T>(in, length);
  }

  double variance = 0.;
  for (size_t i = 0; i < length; ++i) {
    variance += pow((in[i] - mean), 2);
  }
  variance /= length;
  return sqrt(variance);
}

#if MODEL_FLAG == 7
void RunModel(const std::vector<shape_t>& input_shapes,
              size_t repeats,
              size_t warmup,
              size_t power_mode,
              size_t thread_num,
              size_t print_output_elem) {
  std::vector<VelaHandles> handles_vec(4);
  size_t alignment = 16;
  char* align_data1 = reinterpret_cast<char*>(aligned_alloc(alignment, verify_vela_bin_len));
  memcpy(align_data1, reinterpret_cast<char*>(verify_vela_bin), verify_vela_bin_len);
  char* align_data2 = reinterpret_cast<char*>(aligned_alloc(alignment, verify_vela2_bin_len));
  memcpy(align_data2, reinterpret_cast<char*>(verify_vela2_bin), verify_vela2_bin_len);
  char* align_data3 = reinterpret_cast<char*>(aligned_alloc(alignment, verify_vela3_bin_len));
  memcpy(align_data3, reinterpret_cast<char*>(verify_vela3_bin), verify_vela3_bin_len);
  char* align_data4 = reinterpret_cast<char*>(aligned_alloc(alignment, verify_vela4_bin_len));
  memcpy(align_data4, reinterpret_cast<char*>(verify_vela4_bin), verify_vela4_bin_len);

  if (vela_bin_validate(align_data1, verify_vela_bin_len) == false) {
    std::cout << "Malformed vela_bin_stream found" << std::endl;
  }
  if (vela_bin_read(align_data1, &(handles_vec[0]), verify_vela_bin_len) == false) {
    std::cout << " \n vela bin read error " << std::endl;
  }

  if (vela_bin_validate(align_data2, verify_vela2_bin_len) == false) {
    std::cout << "Malformed vela_bin_stream found" << std::endl;
  }
  if (vela_bin_read(align_data2, &(handles_vec[1]), verify_vela2_bin_len) == false) {
    std::cout << " \n vela bin read error " << std::endl;
  }

  if (vela_bin_validate(align_data3, verify_vela3_bin_len) == false) {
    std::cout << "Malformed vela_bin_stream found" << std::endl;
  }
  if (vela_bin_read(align_data3, &(handles_vec[2]), verify_vela3_bin_len) == false) {
    std::cout << " \n vela bin read error " << std::endl;
  }

  if (vela_bin_validate(align_data4, verify_vela4_bin_len) == false) {
    std::cout << "Malformed vela_bin_stream found" << std::endl;
  }
  if (vela_bin_read(align_data4, &(handles_vec[3]), verify_vela4_bin_len) == false) {
    std::cout << " \n vela bin read error " << std::endl;
  }

  auto driver =
  std::unique_ptr<ethosu_driver, decltype(&ethosu_release_driver)>(
      ethosu_reserve_driver(), ethosu_release_driver);

  if (driver == NULL) {
    std::cout <<  "ArmBackend::execute: ethosu_reserve_driver failed" << std::endl;
    return;
  }

  // init 3 inputs buffer for part2
  std::vector<int8_t> part2_input1(30720);
  std::vector<int8_t> part2_input2(15360);
  std::vector<int8_t> part2_input3(7680);

  for (int vela_index = 0; vela_index < 4; vela_index++) {
    VelaHandles handles = handles_vec[vela_index];
    std::cout << "handles.inputs->count is " << handles.inputs->count << std::endl;
    for (int i = 0; i < handles.inputs->count; i++) {
      char* scratch_addr = handles.scratch_data + handles.inputs->io[i].offset;
      size_t shapes = handles.inputs->io[i].shape[0] * handles.inputs->io[i].shape[1] *
                handles.inputs->io[i].shape[2] * handles.inputs->io[i].shape[3];

      printf("input tensor scratch_addr address  %p\n", scratch_addr);
      std::cout << "input shapes " << shapes << std::endl;

      if (vela_index < 3) {
        // tinypose part1
        if (verify_input_tensor_bin_len == shapes) {
          printf("copy input data into  scratch_addr \n");
          memcpy(scratch_addr, verify_input_tensor_bin, verify_input_tensor_bin_len);
        }
      } else {
        // tinypose part2
        if (part2_input1.size() == shapes) {
          printf("copy input data into  scratch_addr \n");
          memcpy(scratch_addr, part2_input1.data(), part2_input1.size());
        } else if (part2_input2.size() == shapes) {
          printf("copy input data2 into  scratch_addr \n");
          memcpy(scratch_addr, part2_input2.data(), part2_input2.size());
        } else if (part2_input3.size() == shapes) {
          printf("copy input data3 into  scratch_addr \n");
          memcpy(scratch_addr, part2_input3.data(), part2_input3.size());
        }
      }
    }

    uint64_t bases[2] = {
        (uint64_t)handles.weight_data, (uint64_t)handles.scratch_data};
    size_t bases_size[2] = {
        handles.weight_data_size, handles.scratch_data_size};
    int result = 0;

    for (int i = 0; i < handles.outputs->count; i++) {
      int tensor_count = 1, io_count = 1;
      const char* output_addr =
          handles.scratch_data + handles.outputs->io[i].offset;

      size_t shapes = handles.outputs->io[i].shape[0] * handles.outputs->io[i].shape[1] *
                handles.outputs->io[i].shape[2] * handles.outputs->io[i].shape[3];

      if (vela_index == 3) {
        std::cout << "handles.outputs->io[x] shapes is " << shapes << std::endl;
        printf("output tensor output_addr address  %p\n", output_addr);
        std::cout << "output shapes " << shapes *  handles.outputs->io[i].elem_size << std::endl;
        printf("output bin  [%p %zu] \n", output_addr, shapes);
      }
    }
    std::cout << "handles.outputs->count is " << handles.outputs->count << std::endl;

    result = ethosu_invoke_v3(
        driver.get(),
        const_cast<void*>(reinterpret_cast<const void*>(handles.cmd_data)),
        handles.cmd_data_size,
        bases,
        bases_size,
        2, /* fixed array of pointers to binary interface*/
        nullptr);

    if (result != 0) {
        std::cout << "result is not 0" << std::endl;
    }

    if (vela_index < 3) {
      for (int i = 0; i < handles.outputs->count; i++) {
        const int8_t * output_addr =
            reinterpret_cast<int8_t *>(handles.scratch_data) + handles.outputs->io[i].offset;

        size_t shapes = handles.outputs->io[i].shape[0] * handles.outputs->io[i].shape[1] *
                  handles.outputs->io[i].shape[2] * handles.outputs->io[i].shape[3];
        if (shapes == part2_input1.size())
          memcpy(part2_input1.data(), output_addr, shapes);
        else if (shapes == part2_input2.size())
          memcpy(part2_input2.data(), output_addr, shapes);
        else if (shapes == part2_input3.size())
          memcpy(part2_input3.data(), output_addr, shapes);
      }
    }
  }
}
#else
void RunModel(const std::vector<shape_t>& input_shapes,
              size_t repeats,
              size_t warmup,
              size_t power_mode,
              size_t thread_num,
              size_t print_output_elem) {
  VelaHandles handles;
  size_t alignment = 16;
#if MODEL_FLAG == 3
  float *tmp0 = reinterpret_cast<float*>(aligned_alloc(alignment, 2704*4));
  float *tmp1 = reinterpret_cast<float*>(aligned_alloc(alignment, 86528*4));
  float *tmp2 = reinterpret_cast<float*>(aligned_alloc(alignment, 216320*4));
  float *tmp3 = reinterpret_cast<float*>(aligned_alloc(alignment, 676*4));
  float *tmp4 = reinterpret_cast<float*>(aligned_alloc(alignment, 21632*4));
  float *tmp5 = reinterpret_cast<float*>(aligned_alloc(alignment, 54080*4));
  float *tmp6 = reinterpret_cast<float*>(aligned_alloc(alignment, 169*4));
  float *tmp7 = reinterpret_cast<float*>(aligned_alloc(alignment, 5408*4));
  float *tmp8 = reinterpret_cast<float*>(aligned_alloc(alignment, 13520*4));
  float *tmp9 = reinterpret_cast<float*>(aligned_alloc(alignment, 49*4));
  float *tmp10 = reinterpret_cast<float*>(aligned_alloc(alignment, 1568*4));
  float *tmp11 = reinterpret_cast<float*>(aligned_alloc(alignment, 3920*4));
#endif
  char* align_data = reinterpret_cast<char*>(aligned_alloc(alignment, verify_vela_bin_len));
  memcpy(align_data, reinterpret_cast<char*>(verify_vela_bin), verify_vela_bin_len);

  if (vela_bin_validate(align_data, verify_vela_bin_len) == false) {
    std::cout << "Malformed vela_bin_stream found" << std::endl;
  }

  if (vela_bin_read(align_data, &handles, verify_vela_bin_len) == false) {
    std::cout << " \n vela bin read error " << std::endl;
  }

  auto driver =
  std::unique_ptr<ethosu_driver, decltype(&ethosu_release_driver)>(
      ethosu_reserve_driver(), ethosu_release_driver);

  if (driver == NULL) {
    std::cout <<  "ArmBackend::execute: ethosu_reserve_driver failed" << std::endl;
    return;
  }

  std::cout << "handles.inputs->count is " << handles.inputs->count << std::endl;
  for (int i = 0; i < handles.inputs->count; i++) {
    char* scratch_addr = handles.scratch_data + handles.inputs->io[i].offset;
    size_t shapes = handles.inputs->io[i].shape[0] * handles.inputs->io[i].shape[1] *
              handles.inputs->io[i].shape[2] * handles.inputs->io[i].shape[3];

    printf("input tensor scratch_addr address  %p\n", scratch_addr);
    std::cout << "input shapes " << shapes << std::endl;

    if (verify_input_tensor_bin_len == shapes) {
      printf("copy input data into  scratch_addr \n");
      memcpy(scratch_addr, verify_input_tensor_bin, verify_input_tensor_bin_len);
    }
  }

  uint64_t bases[2] = {
      (uint64_t)handles.weight_data, (uint64_t)handles.scratch_data};
  size_t bases_size[2] = {
      handles.weight_data_size, handles.scratch_data_size};
  int result = 0;

  for (int i = 0; i < handles.outputs->count; i++) {
    int tensor_count = 1, io_count = 1;
    const char* output_addr =
        handles.scratch_data + handles.outputs->io[i].offset;

    size_t shapes = handles.outputs->io[i].shape[0] * handles.outputs->io[i].shape[1] *
              handles.outputs->io[i].shape[2] * handles.outputs->io[i].shape[3];

    std::cout << "handles.outputs->io[x] shapes is " << shapes << std::endl;
#if MODEL_FLAG == 2 or MODEL_FLAG == 3 or MODEL_FLAG == 5 or MODEL_FLAG == 6
    printf("output tensor output_addr address  %p\n", output_addr);
    std::cout << "output shapes " << shapes *  handles.outputs->io[i].elem_size << std::endl;
#elif MODEL_FLAG == 4
    printf("output tensor output_addr address  %p\n", output_addr);
    std::cout << "output shapes " << shapes << std::endl;
#else
    printf("output tensor output_addr address  %p\n", g_output);
    std::cout << "output shapes " << sizeof(g_output) << std::endl;
#endif
    printf("output bin  [%p %zu] \n", output_addr, shapes);
  }
  std::cout << "handles.outputs->count is " << handles.outputs->count << std::endl;

#if MODEL_FLAG == 1
  // PPLCnetv2
  float output_scale = 0.074288541;
#else
  float output_scale = 1;
#endif
  result = ethosu_invoke_v3(
      driver.get(),
      const_cast<void*>(reinterpret_cast<const void*>(handles.cmd_data)),
      handles.cmd_data_size,
      bases,
      bases_size,
      2, /* fixed array of pointers to binary interface*/
      nullptr);

  if (result != 0) {
      std::cout << "result is not 0" << std::endl;
  }
#if MODEL_FLAG == 3 //picodet
  for (int i = 0; i < handles.outputs->count; i++) {
    int tensor_count = 1, io_count = 1;
    int8_t * output_addr =
        reinterpret_cast<int8_t *>(handles.scratch_data) + handles.outputs->io[i].offset;

    size_t shapes = handles.outputs->io[i].shape[0] * handles.outputs->io[i].shape[1] *
              handles.outputs->io[i].shape[2] * handles.outputs->io[i].shape[3];
    if(shapes == 49)
       int82float(output_addr, tmp9, shapes, 0.057094354);
    else if(shapes == 1568)
       int82float(output_addr, tmp10, shapes, 0.051765633);
    else if(shapes == 3920)
       int82float(output_addr, tmp11, shapes, 0.087895138);
    else if(shapes == 169)
       int82float(output_addr, tmp6, shapes, 0.063905528);
    else if(shapes == 5408)
       int82float(output_addr, tmp7, shapes, 0.053765744);
    else if(shapes == 13520)
       int82float(output_addr, tmp8, shapes, 0.10870742);
    else if(shapes == 676)
       int82float(output_addr, tmp3, shapes, 0.051181102);
    else if(shapes == 21632)
       int82float(output_addr, tmp4, shapes, 0.056824466);
    else if(shapes == 54080)
       int82float(output_addr, tmp5, shapes, 0.120927052);
    else if(shapes == 2704)
       int82float(output_addr, tmp0, shapes, 0.028682091);
    else if(shapes == 86528)
       int82float(output_addr, tmp1, shapes, 0.063936984);
    else if(shapes == 216320)
       int82float(output_addr, tmp2, shapes, 0.14924572);
  }
  process_conv_outputs(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11);
#elif MODEL_FLAG == 6
  // ppocr_rec
  RecPostprocess processor;
  std::string text;
  float confidence;

  for (int i = 0; i < handles.outputs->count; i++) {
    int8_t *output_addr = reinterpret_cast<int8_t *>(handles.scratch_data) + handles.outputs->io[i].offset;
    size_t shapes = handles.outputs->io[i].shape[0] * handles.outputs->io[i].shape[1] *
                    handles.outputs->io[i].shape[2] * handles.outputs->io[i].shape[3];
    std::cout << "\nShape : " << shapes << std::endl;

    std::vector<int8_t> output_buffer(output_addr, output_addr + shapes);

    std::tie(text, confidence) = processor.process(output_buffer);
    std::cout << "\nRec Reuslut: " << text << std::endl;
    std::cout << "Confidence: " << confidence << std::endl;
  }

#else
  for (int i = 0; i < handles.outputs->count; i++) {
    int tensor_count = 1, io_count = 1;
    const int8_t * output_addr =
        reinterpret_cast<int8_t *>(handles.scratch_data) + handles.outputs->io[i].offset;

    size_t shapes = handles.outputs->io[i].shape[0] * handles.outputs->io[i].shape[1] *
              handles.outputs->io[i].shape[2] * handles.outputs->io[i].shape[3];
    float input_r[1000];

    // TODO: scale varies according to the model.
    for (int i = 0; i < 1000; i++) {
      input_r[i]  = static_cast<float>(output_addr[i]) * output_scale;
    }
    softmax(input_r, g_output, 1000);
  }
#endif
}
#endif

int main(int argc, const char* argv[]) {
  std::vector<std::string> str_input_shapes;
  std::vector<shape_t> input_shapes{
      {1, 3, 224, 224}};  // shape_t ==> std::vector<int64_t>


  int repeats = 10;
  int warmup = 10;
  // set arm power mode:
  // 0 for big cluster, high performance
  // 1 for little cluster
  // 2 for all cores
  // 3 for no bind
  size_t power_mode = 0;
  size_t thread_num = 1;
  int print_output_elem = 1;

  if (argc > 2 && argc < 9) {
    std::cout
        << "usage: ./" << argv[0] << "\n"
        << "  <naive_buffer_model_dir>\n"
        << "  <raw_input_shapes>, eg: 1,3,224,224 for 1 input; "
           "1,3,224,224:1,5 for 2 inputs\n"
        << "  <repeats>, eg: 100\n"
        << "  <warmup>, eg: 10\n"
        << "  <power_mode>, 0: big cluster, high performance\n"
           "                1: little cluster\n"
           "                2: all cores\n"
           "                3: no bind\n"
        << "  <thread_num>, eg: 1 for single thread \n"
        << "  <accelerate_opencl>, this option takes effect only when model "
           "can be running on opencl backend.\n"
           "                       0: disable opencl kernel cache & tuning\n"
           "                       1: enable opencl kernel cache & tuning\n"
        << "  <print_output>, 0: disable print outputs to stdout\n"
           "                  1: enable print outputs to stdout\n"
        << std::endl;
    return 0;
  }
#if MODEL_FLAG == 5
  std::vector<float> in_data(verify_input_tensor_bin_len/4);
  memcpy(reinterpret_cast<char*>(in_data.data()), verify_input_tensor_bin, verify_input_tensor_bin_len);
  std::vector<int8_t> tmp  = layout_conv_preprocess(in_data);
  memcpy(verify_input_tensor_bin, reinterpret_cast<char*>(tmp.data()), tmp.size());
  verify_input_tensor_bin_len = tmp.size();
#endif

  StartMeasurements();
  RunModel(input_shapes,
           repeats,
           warmup,
           power_mode,
           thread_num,
           print_output_elem);
  StopMeasurements();
  std::cout << "Running Model Exit Successfully\n";
  return 0;
}

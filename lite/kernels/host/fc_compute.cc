// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/kernels/host/fc_compute.h"
#include <string>
#include <vector>
#include "lite/api/paddle_place.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {


///  for fp32 kernel
template <>
void FcCompute<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
}


/// for int8 kernel with fp32 output
template <>
void FcCompute<PRECISION(kInt8), PRECISION(kFloat)>::PrepareForRun() {
}

// for int8 kernel with int8 output
template <>
void FcCompute<PRECISION(kInt8), PRECISION(kInt8)>::PrepareForRun() {
}

template <>
void FcCompute<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
    LOG(INFO) << "Run in FcCompute\n"; 
}

template <>
void FcCompute<PRECISION(kInt8), PRECISION(kFloat)>::Run() {
    LOG(INFO) << "Run in FcCompute\n"; 
}

template <>
void FcCompute<PRECISION(kInt8), PRECISION(kInt8)>::Run() {
    LOG(INFO) << "Run in FcCompute\n"; 
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::host::FcCompute<PRECISION(kFloat),
                                              PRECISION(kFloat)>
    FcCompute_FP32;
typedef paddle::lite::kernels::host::FcCompute<PRECISION(kInt8),
                                              PRECISION(kFloat)>
    FcCompute_int8_fp32;
typedef paddle::lite::kernels::host::FcCompute<PRECISION(kInt8),
                                              PRECISION(kInt8)>
    FcCompute_int8_int8;


REGISTER_LITE_KERNEL(fc, kHost, kFloat, kNCHW, FcCompute_FP32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Alpha", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();

REGISTER_LITE_KERNEL(fc, kHost, kInt8, kNCHW, FcCompute_int8_int8, int8out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt8))})
    .Finalize();

REGISTER_LITE_KERNEL(fc, kHost, kInt8, kNCHW, FcCompute_int8_fp32, fp32out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .Finalize();

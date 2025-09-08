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

#include "lite/kernels/host/pool_compute.h"
#include <string>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

#define POOL_IN_PARAM                                                        \
  din, dout, out_dims[0], out_dims[1], out_dims[2], out_dims[3], in_dims[1], \
      in_dims[2], in_dims[3]
template <>
void PoolCompute<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
    LOG(INFO) << "Run in PoolCompute\n"; 
}
#undef POOL_IN_PARAM
}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::host::PoolCompute<PRECISION(kFloat),
                                                PRECISION(kFloat)>
    PoolFp32;
REGISTER_LITE_KERNEL(pool2d, kHost, kFloat, kNCHW, PoolFp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindPaddleOpVersion("pool2d", 1)
    .Finalize();

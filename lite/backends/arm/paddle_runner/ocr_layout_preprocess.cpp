#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <memory>
#include <string>
#if MODEL_FLAG == 5
#include "b1.h"
#include "b2.h"
#include "b3.h"
#include "b4.h"
#include "b5.h"
#include "b6.h"
#include "w1.h"
#include "w2.h"
#include "w3.h"
#include "w4.h"
#include "w5.h"
#include "w6.h"
#include <cstring>
class Conv2d {
 private:
  int in_channels;
  int out_channels;
  std::vector<int> kernel_size;
  std::vector<int> stride;
  std::vector<int> padding;
  int groups;
  bool use_bias;

  std::vector<float> weight; // format: [out_channels, in_channels/groups, kH, kW]
  std::vector<float> bias;  // format: [out_channels]

  std::vector<int> weight_shape;

 public:
  Conv2d(int in_channels,
         int out_channels,
         std::vector<int> kernel_size,
         std::vector<int> stride = {1, 1},
         std::vector<int> padding = {0, 0},
         int groups = 1,
         bool has_bias = true) 
      : in_channels(in_channels),
        out_channels(out_channels),
        kernel_size(kernel_size),
        stride(stride),
        padding(padding),
        groups(groups),
        use_bias(has_bias) {
    weight_shape = {
        out_channels, in_channels / groups, kernel_size[0], kernel_size[1]};

    int weight_size =
        out_channels * (in_channels / groups) * kernel_size[0] * kernel_size[1];
    weight.resize(weight_size, 0.0f);

    if (use_bias) {
      bias.resize(out_channels, 0.0f); 
    }
  }

  void load_weight_from_header(unsigned char * buffer, int len) {
    std::vector<float> data(len);
    memcpy(reinterpret_cast<char*>(data.data()), buffer, len);
    weight = data;
  }

  void load_bias_from_header(unsigned char * buffer, int len) {
    if (!use_bias) {
        std::cout<<"not using bias\n";
    }
    std::vector<float> data(len);
    memcpy(reinterpret_cast<char*>(data.data()), buffer, len);
    bias = data;
  }

  const std::vector<float>& get_weight() const { return weight; }

  const std::vector<float>& get_bias() const {
    if (!use_bias) {
        std::cout<<"not using bias\n";
    }
    return bias;
  }
  std::vector<int> get_output_shape(const std::vector<int>& input_shape) const {
    if (input_shape.size() != 4) {
        std::cout<<"input shape not correct\n";
    }

    int batch_size = input_shape[0];
    int height = input_shape[2];
    int width = input_shape[3];

    int out_height = (height + 2 * padding[0] - kernel_size[0]) / stride[0] + 1;
    int out_width = (width + 2 * padding[1] - kernel_size[1]) / stride[1] + 1;

    return {batch_size, out_channels, out_height, out_width};
  }

  std::vector<float> forward(const std::vector<float>& input,
                             const std::vector<int>& input_shape) const {
    // 检查输入shape
    if (input_shape.size() != 4 || input_shape[1] != in_channels) {
        std::cout<<"input shape not correct\n";
    }

    int batch_size = input_shape[0];
    int height = input_shape[2];
    int width = input_shape[3];

    // 检查输入数据大小
    if (input.size() !=
        static_cast<size_t>(batch_size * in_channels * height * width)) {
        std::cout<<"input size not correct\n";
    }
    // 计算输出尺寸
    std::vector<int> output_shape = get_output_shape(input_shape);
    int out_height = output_shape[2];
    int out_width = output_shape[3];

    // 输出数据
    std::vector<float> output(
        batch_size * out_channels * out_height * out_width, 0.0f);

    // 对每个样本进行卷积操作
    for (int b = 0; b < batch_size; ++b) {
      for (int g = 0; g < groups; ++g) {
        // 计算当前组的通道范围
        int in_channels_per_group = in_channels / groups;
        int out_channels_per_group = out_channels / groups;

        for (int oc = 0; oc < out_channels_per_group; ++oc) {
          int out_c = g * out_channels_per_group + oc;  // 全局输出通道索引

          for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
              // 输出索引
              int out_idx =
                  ((b * out_channels + out_c) * out_height + oh) * out_width +
                  ow;

              // 初始化为bias（如果有）
              if (use_bias) {
                output[out_idx] = bias[out_c];
              } else {
                output[out_idx] = 0.0f;
              }

              // 对应的输入区域的左上角
              int ih_start = oh * stride[0] - padding[0];
              int iw_start = ow * stride[1] - padding[1];

              // 卷积操作
              for (int ic = 0; ic < in_channels_per_group; ++ic) {
                int in_c = g * in_channels_per_group + ic;  // 全局输入通道索引

                for (int kh = 0; kh < kernel_size[0]; ++kh) {
                  for (int kw = 0; kw < kernel_size[1]; ++kw) {
                    int ih = ih_start + kh;
                    int iw = iw_start + kw;

                    // 检查边界
                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                      // 输入索引
                      int in_idx =
                          ((b * in_channels + in_c) * height + ih) * width + iw;

                      // 权重索引 - 这里使用PyTorch的权重格式 [out_channels,
                      // in_channels/groups, kH, kW]
                      int w_idx = (((out_c) * (in_channels / groups) + ic) *
                                       kernel_size[0] +
                                   kh) *
                                      kernel_size[1] +
                                  kw;

                      output[out_idx] += input[in_idx] * weight[w_idx];
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    return output;
  }

  void print_info() const {
    std::cout << "Conv2d setting:" << std::endl;
    std::cout << "  in channels: " << in_channels << std::endl;
    std::cout << "  out channels: " << out_channels << std::endl;
    std::cout << "  kernel: [" << kernel_size[0] << ", " << kernel_size[1]
              << "]" << std::endl;
    std::cout << "  stride: [" << stride[0] << ", " << stride[1] << "]"
              << std::endl;
    std::cout << "  padding: [" << padding[0] << ", " << padding[1] << "]"
              << std::endl;
    std::cout << "  groups: " << groups << std::endl;
    std::cout << "  using bias: " << (use_bias ? "yes" : "no") << std::endl;
    std::cout << "  weight shape: [" << weight_shape[0] << ", " << weight_shape[1]
              << ", " << weight_shape[2] << ", " << weight_shape[3] << "]"
              << std::endl;
  }
};

std::vector<float> hardswish(const std::vector<float>& input) {
  std::vector<float> output(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    float x = input[i];
    float relu6 = std::min(std::max(0.0f, x + 3.0f), 6.0f);
    output[i] = x * relu6 / 6.0f;
  }
  return output;
}



void transpose_tensor(int8_t* input, int8_t* output, int* input_shape, int* perm) {
    // 计算各个维度的步长
    int input_strides[4];
    input_strides[3] = 1;  // 最后一个维度步长为1
    for (int i = 2; i >= 0; --i) {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }

    // 输出tensor的形状
    int output_shape[4];
    for (int i = 0; i < 4; ++i) {
        output_shape[i] = input_shape[perm[i]];
    }

    // 计算输出tensor的步长
    int output_strides[4];
    output_strides[3] = 1;
    for (int i = 2; i >= 0; --i) {
        output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
    }

    // 遍历输入tensor的每个元素
    for (int b = 0; b < input_shape[0]; ++b) {
        for (int c = 0; c < input_shape[1]; ++c) {
            for (int h = 0; h < input_shape[2]; ++h) {
                for (int w = 0; w < input_shape[3]; ++w) {
                    // 输入tensor的索引
                    int input_idx = b * input_strides[0] + c * input_strides[1] +
                                   h * input_strides[2] + w * input_strides[3];

                    // 输出tensor的索引（根据perm规则）
                    int output_idx = b * output_strides[0] + h * output_strides[1] +
                                    w * output_strides[2] + c * output_strides[3];

                    // 执行转置
                    output[output_idx] = input[input_idx];
                }
            }
        }
    }
}

// 卷积网络处理函数 - 只返回int8_t数组
std::vector<int8_t> layout_conv_preprocess(
    const std::vector<float>& normalized_input,
    int batch_size = 1,
    int height = 800,
    int width = 608) {
    // 定义输入shape
    std::vector<int> input_shape = {batch_size, 3, height, width};


    // 创建卷积层并加载权重
    Conv2d conv1(3, 16, {3, 3}, {2, 2}, {1, 1}, 1, true);
    conv1.load_weight_from_header(verify_w1_bin, verify_w1_bin_len);
    conv1.load_bias_from_header(verify_b1_bin, verify_b1_bin_len);

    Conv2d conv2(16, 16, {3, 3}, {1, 1}, {1, 1}, 16, true);
    conv2.load_weight_from_header(verify_w2_bin, verify_w2_bin_len);
    conv2.load_bias_from_header(verify_b2_bin, verify_b2_bin_len);

    Conv2d conv3(16, 32, {1, 1}, {1, 1}, {0, 0}, 1, true);
    conv3.load_weight_from_header(verify_w3_bin, verify_w3_bin_len);
    conv3.load_bias_from_header(verify_b3_bin, verify_b3_bin_len);

    Conv2d conv4(32, 32, {3, 3}, {2, 2}, {1, 1}, 32, true);
    conv4.load_weight_from_header(verify_w4_bin, verify_w4_bin_len);
    conv4.load_bias_from_header(verify_b4_bin, verify_b4_bin_len);

    Conv2d conv5(32, 64, {1, 1}, {1, 1}, {0, 0}, 1, true);
    conv5.load_weight_from_header(verify_w5_bin, verify_w5_bin_len);
    conv5.load_bias_from_header(verify_b5_bin, verify_b5_bin_len);

    Conv2d conv6(64, 64, {3, 3}, {1, 1}, {1, 1}, 64, true);
    conv6.load_weight_from_header(verify_w6_bin, verify_w6_bin_len);
    conv6.load_bias_from_header(verify_b6_bin, verify_b6_bin_len);

    // 执行前向传播
    std::vector<float> x = normalized_input;
    std::vector<int> x_shape = input_shape;

    // 第一层卷积 + hardswish
    x = conv1.forward(x, x_shape);
    x_shape = conv1.get_output_shape(x_shape);
    x = hardswish(x);

    // 第二层卷积 + hardswish
    x = conv2.forward(x, x_shape);
    x_shape = conv2.get_output_shape(x_shape);
    x = hardswish(x);

    // 第三层卷积 + hardswish
    x = conv3.forward(x, x_shape);
    x_shape = conv3.get_output_shape(x_shape);
    x = hardswish(x);

    // 第四层卷积 + hardswish
    x = conv4.forward(x, x_shape);
    x_shape = conv4.get_output_shape(x_shape);
    x = hardswish(x);

    // 第五层卷积 + hardswish
    x = conv5.forward(x, x_shape);
    x_shape = conv5.get_output_shape(x_shape);
    x = hardswish(x);

    // 第六层卷积 + hardswish
    x = conv6.forward(x, x_shape);
    x_shape = conv6.get_output_shape(x_shape);
    x = hardswish(x);

    // 除以缩放因子
    const float scale_factor = 0.3498304486274719f;
    for (auto& val : x) {
      val /= scale_factor;
    }

    // 转换为int8类型
    std::vector<int8_t> output(x.size());
    int8_t *tmp = (int8_t*)malloc(x.size());
    std::vector<int8_t> transpose_output(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      float clamped = std::min(std::max(x[i], -128.0f), 127.0f);
      output[i] = static_cast<int8_t>(std::round(clamped));
    }
        
    int in_shape[4] = {1, 64, 200, 152};
    // 转置后的形状 [1, 200, 152, 64]
    int perm[4] = {0, 2, 3, 1};  // 指定转置的维度顺序
    transpose_tensor(output.data(), tmp, in_shape, perm);
    memcpy((char*)transpose_output.data(), tmp, x.size());
    // 返回int8结果
    return transpose_output;
} 
#endif




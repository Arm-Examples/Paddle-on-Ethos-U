// Copyright (c) 2020-2025, ARM Limited.
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//         http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#ifndef REC_POSTPROCESS_H
#define REC_POSTPROCESS_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <memory>
#include <numeric>
#include <cstring>
#include <unordered_map>

#include "weights.h"
#include "label_dict.h"

// Matrix: Implements a multi-dimensional array with 
// various operations like indexing, reshaping, and mathematical operations.
class Matrix {
private:
    std::vector<float> data;
    std::vector<int> shape;
    int stride_size;
    int total_size;

    // Calculate total size and strides
    void computeStrides() {
        stride_size = 1;
        total_size = 1;
        for (int i = shape.size() - 1; i >= 0; i--) {
            total_size *= shape[i];
        }
        data.resize(total_size, 0.0f);
    }

public:
    // default constructor
    Matrix() : shape({0}), stride_size(1), total_size(0) {}

    // Constructor that initializes the matrix with given dimensions
    explicit Matrix(const std::vector<int> &dimensions) : shape(dimensions) {
        computeStrides();
    }

    // Constructor that initializes the matrix with given dimensions and initial value
    Matrix(const std::vector<int> &dimensions, float init_value) : shape(dimensions) {
        computeStrides();
        std::fill(data.begin(), data.end(), init_value);
    }

    // Copy constructor
    Matrix(const Matrix &other) : data(other.data), shape(other.shape),
                                  stride_size(other.stride_size), total_size(other.total_size) {}

    // Getters for shape and size
    int dim() const { return shape.size(); }
    const std::vector<int> &getShape() const { return shape; }
    int size(int dim) const { return shape[dim]; }
    int totalSize() const { return total_size; }

    // Indexing function to convert multi-dimensional indices to a flat index
    int flattenIndex(const std::vector<int> &indices) const {
        // if (indices.size() != shape.size())
        // {
        //     throw std::runtime_error("Unmatching number of indices and dimensions");
        // }

        int flat_index = 0;
        int multiplier = 1;
        for (int i = shape.size() - 1; i >= 0; i--) {
            // if (indices[i] < 0 || indices[i] >= shape[i])
            // {
            //     throw std::runtime_error("Out of bounds index at dimension " + std::to_string(i));
            // }
            flat_index += indices[i] * multiplier;
            multiplier *= shape[i];
        }
        return flat_index;
    }
    // Single index access for 1D matrices
    float &at(int index) {
        // if (shape.size() != 1)
        // {
        //     throw std::runtime_error("at() only works for 1D matrices");
        // }
        // if (index < 0 || index >= shape[0])
        // {
        //     throw std::runtime_error("Out of bounds index");
        // }
        return data[index];
    }

    const float &at(int index) const {
        // if (shape.size() != 1)
        // {
        //     throw std::runtime_error("at() only works for 1D matrices");
        // }
        // if (index < 0 || index >= shape[0])
        // {
        //     throw std::runtime_error("Out of bounds index");
        // }
        return data[index];
    }

    // Multi-dimensional indexing
    float &at(const std::vector<int> &indices) {
        return data[flattenIndex(indices)];
    }

    const float &at(const std::vector<int> &indices) const {
        return data[flattenIndex(indices)];
    }

    // Indexing for 2D matrices
    float &at(int i, int j) {
        // if (shape.size() != 2)
        // {
        //     throw std::runtime_error("Only works for 2D matrices");
        // }
        return data[i * shape[1] + j];
    }

    const float &at(int i, int j) const {
        // if (shape.size() != 2)
        // {
        //     throw std::runtime_error("Only works for 2D matrices");
        // }
        return data[i * shape[1] + j];
    }

    // Indexing for 3D matrices
    float &at(int i, int j, int k) {
        // if (shape.size() != 3)
        // {
        //     throw std::runtime_error("Only works for 3D matrices");
        // }
        return data[(i * shape[1] + j) * shape[2] + k];
    }

    const float &at(int i, int j, int k) const {
        // if (shape.size() != 3)
        // {
        //     throw std::runtime_error("Only works for 3D matrices");
        // }
        return data[(i * shape[1] + j) * shape[2] + k];
    }

    // Indexing for 4D matrices
    float &at(int i, int j, int k, int l) {
        // if (shape.size() != 4)
        // {
        //     throw std::runtime_error("Only works for 4D matrices");
        // }
        return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
    }

    const float &at(int i, int j, int k, int l) const {
        // if (shape.size() != 4)
        // {
        //     throw std::runtime_error("Only works for 4D matrices");
        // }
        return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
    }

    float *getData() { return data.data(); }
    const float *getData() const { return data.data(); }

    Matrix reshape(const std::vector<int> &new_shape) const {
        int new_total = 1;
        for (int dim : new_shape) {
            new_total *= dim;
        }

        // if (new_total != total_size)
        // {
        //     throw std::runtime_error("Shape have to match total size");
        // }

        Matrix result(new_shape);
        std::copy(data.begin(), data.end(), result.data.begin());
        return result;
    }

    Matrix transpose(const std::vector<int> &dims) const {
        // if (dims.size() != shape.size())
        // {
        //     throw std::runtime_error("Dims have to match shape size");
        // }

        // Check if dims is a permutation of 0 to shape.size()-1
        std::vector<int> check(dims);
        std::sort(check.begin(), check.end());
        // for (size_t i = 0; i < check.size(); i++)
        // {
        //     if (check[i] != static_cast<int>(i))
        //     {
        //         throw std::runtime_error("Transposed dimensions must be a permutation of 0 to n-1");
        //     }
        // }

        std::vector<int> new_shape(shape.size());
        for (size_t i = 0; i < dims.size(); i++) {
            new_shape[i] = shape[dims[i]];
        }

        Matrix result(new_shape);

        std::vector<int> old_indices(shape.size());
        std::vector<int> new_indices(shape.size());

        for (int i = 0; i < total_size; i++) {
            int temp = i;
            for (int j = shape.size() - 1; j >= 0; j--) {
                old_indices[j] = temp % shape[j];
                temp /= shape[j];
            }

            for (size_t j = 0; j < dims.size(); j++) {
                new_indices[j] = old_indices[dims[j]];
            }

            result.at(new_indices) = data[i];
        }

        return result;
    }

    Matrix permute(int dim0, int dim1, int dim2) const {
        return transpose({dim0, dim1, dim2});
    }

    Matrix permute(int dim0, int dim1, int dim2, int dim3) const {
        return transpose({dim0, dim1, dim2, dim3});
    }

    Matrix squeeze(int dim) const {
        // if (dim < 0 || dim >= static_cast<int>(shape.size()))
        // {
        //     throw std::runtime_error("Unvalid dimension for squeeze");
        // }

        // if (shape[dim] != 1)
        // {
        //     throw std::runtime_error("Only dimensions with size 1 can be squeezed");
        // }

        std::vector<int> new_shape;
        for (size_t i = 0; i < shape.size(); i++) {
            if (i != static_cast<size_t>(dim)) {
                new_shape.push_back(shape[i]);
            }
        }

        if (new_shape.empty()) {
            new_shape.push_back(1);
        }

        return reshape(new_shape);
    }

    Matrix operator+(const Matrix &other) const {
        if (shape == other.shape) {
            Matrix result(*this);
            for (int i = 0; i < total_size; i++) {
                result.data[i] += other.data[i];
            }
            return result;
        }

        // Vector Add for 1D matrices(Boardcasting)
        if (other.dim() == 1) {
            // if (shape.back() != other.size(0))
            // {
            //     throw std::runtime_error("Last dimension of matrix must match size of vector for addition");
            // }

            Matrix result(*this);
            int bias_size = other.size(0);

            for (int i = 0; i < total_size / bias_size; i++) {
                for (int j = 0; j < bias_size; j++) {
                    result.data[i * bias_size + j] += other.at(j);
                }
            }

            return result;
        }
        return Matrix();
        // throw std::runtime_error("Shape not match for boardcasting addition");
    }

    Matrix operator-(const Matrix &other) const {
        // if (shape != other.shape)
        // {
        //     throw std::runtime_error("Shape not match");
        // }

        Matrix result(*this);
        for (int i = 0; i < total_size; i++) {
            result.data[i] -= other.data[i];
        }

        return result;
    }

    Matrix operator*(const Matrix &other) const {
        // if (shape != other.shape)
        // {
        //     throw std::runtime_error("Unmatch shape for multiplication");
        // }

        Matrix result(*this);
        for (int i = 0; i < total_size; i++) {
            result.data[i] *= other.data[i];
        }

        return result;
    }

    Matrix operator*(float scalar) const {
        Matrix result(*this);
        for (int i = 0; i < total_size; i++) {
            result.data[i] *= scalar;
        }

        return result;
    }

    Matrix matmul(const Matrix &other) const {
        if (shape.size() == 2 && other.shape.size() == 2) {
            // if (shape[1] != other.shape[0])
            // {
            //     throw std::runtime_error("Unmatch shape for multiplication");
            // }

            int m = shape[0];
            int n = other.shape[1];
            int k = shape[1];

            Matrix result({m, n}, 0.0f);

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    float sum = 0.0f;
                    for (int l = 0; l < k; l++) {
                        sum += at(i, l) * other.at(l, j);
                    }
                    result.at(i, j) = sum;
                }
            }

            return result;
        } else if (shape.size() == 3 && other.shape.size() == 2) {
            // Support for 3D matrix multiplied by 2D matrix
            // if (shape[2] != other.shape[0])
            // {
            //     throw std::runtime_error("Unmatch shape for multiplication");
            // }

            int batch = shape[0];
            int m = shape[1];
            int n = other.shape[1];
            int k = shape[2];

            Matrix result({batch, m, n}, 0.0f);

            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        float sum = 0.0f;
                        for (int l = 0; l < k; l++) {
                            sum += at(b, i, l) * other.at(l, j);
                        }
                        result.at(b, i, j) = sum;
                    }
                }
            }

            return result;
        } else {
            printf("matmul only support 2D matrix multiplication or 3D matrix multiplied by 2D matrix\n");
            return Matrix();
        }
    }

    Matrix t() const {
        // if (shape.size() != 2)
        // {
        //     throw std::runtime_error("t()only support 2D matrices");
        // }

        Matrix result({shape[1], shape[0]});

        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                result.at(j, i) = at(i, j);
            }
        }

        return result;
    }

    Matrix sigmoid() const {
        Matrix result(*this);
        for (int i = 0; i < total_size; i++) {
            result.data[i] = 1.0f / (1.0f + std::exp(-data[i]));
        }
        return result;
    }

    Matrix tanh() const {
        Matrix result(*this);
        for (int i = 0; i < total_size; i++) {
            result.data[i] = std::tanh(data[i]);
        }
        return result;
    }

    Matrix hardswish() const {
        Matrix result(*this);
        for (int i = 0; i < total_size; i++) {
            float x = data[i];
            if (x <= -3) {
                result.data[i] = 0;
            } else if (x >= 3) {
                result.data[i] = x;
            } else {
                result.data[i] = x * (x + 3) / 6;
            }
        }
        return result;
    }

    Matrix softmax(int dim) const {
        // if (dim < 0 || dim >= static_cast<int>(shape.size()))
        // {
        //     throw std::runtime_error("Unvalid dimension for softmax");
        // }

        Matrix result(*this);

        int dim_size = shape[dim];

        int pre_stride = 1;
        for (int i = shape.size() - 1; i > dim; i--) {
            pre_stride *= shape[i];
        }

        int post_stride = pre_stride * dim_size;

        for (int i = 0; i < total_size / post_stride; i++) {
            for (int j = 0; j < pre_stride; j++) {
                float max_val = -std::numeric_limits<float>::max();
                for (int k = 0; k < dim_size; k++) {
                    int idx = i * post_stride + k * pre_stride + j;
                    max_val = std::max(max_val, data[idx]);
                }

                float sum_exp = 0.0f;
                for (int k = 0; k < dim_size; k++) {
                    int idx = i * post_stride + k * pre_stride + j;
                    result.data[idx] = std::exp(data[idx] - max_val);
                    sum_exp += result.data[idx];
                }

                for (int k = 0; k < dim_size; k++) {
                    int idx = i * post_stride + k * pre_stride + j;
                    result.data[idx] /= sum_exp;
                }
            }
        }

        return result;
    }

    Matrix max_pool2d(int kernel_size, int stride, int padding = 0) const {
        // if (shape.size() != 4)
        // {
        //     throw std::runtime_error("max_pool2d only support 4D tensor [batch, channels, height, width]");
        // }

        int batch_size = shape[0];
        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];

        int output_height = (height + 2 * padding - kernel_size) / stride + 1;
        int output_width = (width + 2 * padding - kernel_size) / stride + 1;

        Matrix result({batch_size, channels, output_height, output_width}, -std::numeric_limits<float>::max());

        for (int b = 0; b < batch_size; b++) {
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < output_height; h++) {
                    for (int w = 0; w < output_width; w++) {
                        float max_val = -std::numeric_limits<float>::max();

                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int h_in = h * stride + kh - padding;
                                int w_in = w * stride + kw - padding;

                                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                                    max_val = std::max(max_val, at(b, c, h_in, w_in));
                                }
                            }
                        }

                        result.at(b, c, h, w) = max_val;
                    }
                }
            }
        }

        return result;
    }

    static Matrix loadWeight(const std::string type) {
        std::unordered_map<std::string, std::vector<int>> shape_info = {
            {"w1", {2, 256, 512}},
            {"r1", {2, 256, 64}},
            {"b1", {2, 512}},
            {"w2", {2, 256, 128}},
            {"r2", {2, 256, 64}},
            {"b2", {2, 512}},
            {"linear1_w", {128, 96}},
            {"linear1_b", {96}},
            {"linear2_w", {96, 6625}},
            {"linear2_b", {6625}}};

        Matrix w(shape_info[type]);

        auto conver2fp32 = [](unsigned char *data, unsigned int len) {
            int float_len = len / 4;
            std::vector<float> result(float_len);
            for (unsigned int i = 0; i < float_len; i++) {
                result[i] = *(float*)(&data[i*4]);
            }
            return result;
        };

        if ("w1" == type) {
            auto data = conver2fp32(w1_bin, w1_bin_len);
            w.data.assign(data.begin(), data.end());

        } else if ("r1" == type) {
            auto data = conver2fp32(r1_bin, r1_bin_len);
            w.data.assign(data.begin(), data.end());

        } else if ("b1" == type) {
            auto data = conver2fp32(b1_bin, b1_bin_len);
            w.data.assign(data.begin(), data.end());

        } else if ("w2" == type) {
            auto data = conver2fp32(w2_bin, w2_bin_len);
            w.data.assign(data.begin(), data.end());

        } else if ("r2" == type) {
            auto data = conver2fp32(r2_bin, r2_bin_len);
            w.data.assign(data.begin(), data.end());

        } else if ("b2" == type) {
            auto data = conver2fp32(b2_bin, b2_bin_len);
            w.data.assign(data.begin(), data.end());

        } else if ("linear1_w" == type) {
            auto data = conver2fp32(linear1_w_bin, linear1_w_bin_len);
            w.data.assign(data.begin(), data.end());

        } else if ("linear1_b" == type) {
            auto data = conver2fp32(linear1_b_bin, linear1_b_bin_len);
            w.data.assign(data.begin(), data.end());

        } else if ("linear2_w" == type) {
            auto data = conver2fp32(linear2_w_bin, linear2_w_bin_len);
            w.data.assign(data.begin(), data.end());

        } else if ("linear2_b" == type) {
            auto data = conver2fp32(linear2_b_bin, linear2_b_bin_len);
            w.data.assign(data.begin(), data.end());
        }
        return w;
    }

    static Matrix loadFromNpy(const std::string &filename) {
        std::ifstream file(filename, std::ios::binary);
        // if (!file)
        // {
        //     throw std::runtime_error("Can not open file: " + filename);
        // }

        char magic[6];
        file.read(magic, 6);
        // if (std::string(magic, 6) != "\x93NUMPY")
        // {
        //     throw std::runtime_error("Not vaild numpy file: " + filename);
        // }

        uint8_t major_version, minor_version;
        file.read(reinterpret_cast<char *>(&major_version), 1);
        file.read(reinterpret_cast<char *>(&minor_version), 1);

        uint16_t header_len;
        file.read(reinterpret_cast<char *>(&header_len), 2);

        std::string header(header_len, ' ');
        file.read(&header[0], header_len);

        size_t shape_start = header.find("'shape': (") + 10;
        size_t shape_end = header.find(")", shape_start);
        std::string shape_str = header.substr(shape_start, shape_end - shape_start);

        std::vector<int> shape;
        size_t pos = 0;
        size_t comma_pos;
        while ((comma_pos = shape_str.find(",", pos)) != std::string::npos) {
            std::string dim_str = shape_str.substr(pos, comma_pos - pos);
            shape.push_back(std::stoi(dim_str));
            pos = comma_pos + 1;
        }
        if (pos < shape_str.length()) {
            shape.push_back(std::stoi(shape_str.substr(pos)));
        }

        Matrix result(shape);

        file.read(reinterpret_cast<char *>(result.data.data()), result.total_size * sizeof(float));

        return result;
    }

    static Matrix fromArray(const float *data, const std::vector<int> &shape) {
        Matrix result(shape);
        int total_size = result.total_size;
        std::memcpy(result.data.data(), data, total_size * sizeof(float));
        return result;
    }

    static Matrix dequantize(const int8_t *data, int size, float scale) {
        std::vector<float> float_data(size);
        for (int i = 0; i < size; i++) {
            float_data[i] = static_cast<float>(data[i]) * scale;
        }

        return Matrix::fromArray(float_data.data(), {1, size});
    }
};

class LSTMImplementation {
private:
    int hidden_size;
    bool bidirectional;
    int num_directions;
    int num_layers;

public:
    LSTMImplementation(int hidden_size = 64, bool bidirectional = true, int num_layers = 1)
        : hidden_size(hidden_size), bidirectional(bidirectional), num_layers(num_layers) {
        num_directions = bidirectional ? 2 : 1;
    }

    // forward
    std::tuple<Matrix, Matrix, Matrix> forward(
        const Matrix &X,                                 // Input sequence [seq_length, batch_size, input_size]
        const Matrix &W,                                 // Weight [num_directions, 4*hidden_size, input_size]
        const Matrix &R,                                 // Recurrent weight [num_directions, 4*hidden_size, hidden_size]
        const Matrix *B = nullptr,                       // Bias [num_directions, 8*hidden_size]
        const std::vector<int> *sequence_lens = nullptr, // Sequence length
        const Matrix *initial_h = nullptr,               // Initial hidden state [num_directions, batch_size, hidden_size]
        const Matrix *initial_c = nullptr,               // Initial cell state [num_directions, batch_size, hidden_size]
        const Matrix *P = nullptr                        // Peephole connection weights [num_directions, 3*hidden_size]
    ) {
        // get input size
        int seq_length = X.size(0);
        int batch_size = X.size(1);
        int input_size = X.size(2);

        // Initialize hidden state and cell state
        Matrix h_init, c_init;
        if (initial_h) {
            h_init = *initial_h;
        } else {
            h_init = Matrix({num_directions, batch_size, hidden_size}, 0.0f);
        }

        if (initial_c) {
            c_init = *initial_c;
        } else {
            c_init = Matrix({num_directions, batch_size, hidden_size}, 0.0f);
        }

        // Extract bias
        Matrix W_bias, R_bias;
        if (B) {
            W_bias = Matrix({num_directions, 4 * hidden_size});
            R_bias = Matrix({num_directions, 4 * hidden_size});

            for (int dir = 0; dir < num_directions; dir++) {
                for (int i = 0; i < 4 * hidden_size; i++) {
                    W_bias.at(dir, i) = B->at(dir, i);
                    R_bias.at(dir, i) = B->at(dir, i + 4 * hidden_size);
                }
            }
        } else {
            W_bias = Matrix({num_directions, 4 * hidden_size}, 0.0f);
            R_bias = Matrix({num_directions, 4 * hidden_size}, 0.0f);
        }

        Matrix P_weights;
        if (P) {
            P_weights = *P;
        } else {
            P_weights = Matrix({num_directions, 3 * hidden_size}, 0.0f);
        }

        Matrix Y({seq_length, num_directions, batch_size, hidden_size}, 0.0f);
        Matrix Y_h({num_directions, batch_size, hidden_size}, 0.0f);
        Matrix Y_c({num_directions, batch_size, hidden_size}, 0.0f);

        // Compute LSTM for each direction
        for (int direction = 0; direction < num_directions; direction++) {
            // Extract initial state for current direction
            Matrix h_t({batch_size, hidden_size});
            Matrix c_t({batch_size, hidden_size});

            for (int b = 0; b < batch_size; b++) {
                for (int h = 0; h < hidden_size; h++) {
                    h_t.at(b, h) = h_init.at(direction, b, h);
                    c_t.at(b, h) = c_init.at(direction, b, h);
                }
            }

            // Process sequence
            std::vector<int> seq_indices(seq_length);
            std::iota(seq_indices.begin(), seq_indices.end(), 0);

            if (direction == 1 && bidirectional) {
                std::reverse(seq_indices.begin(), seq_indices.end());
            }

            // Extract weights and bias for current direction
            Matrix W_dir = Matrix({4 * hidden_size, input_size});
            Matrix R_dir = Matrix({4 * hidden_size, hidden_size});
            Matrix W_bias_dir = Matrix({4 * hidden_size});
            Matrix R_bias_dir = Matrix({4 * hidden_size});

            for (int i = 0; i < 4 * hidden_size; i++) {
                for (int j = 0; j < input_size; j++) {
                    W_dir.at(i, j) = W.at(direction, i, j);
                }
                for (int j = 0; j < hidden_size; j++) {
                    R_dir.at(i, j) = R.at(direction, i, j);
                }
                W_bias_dir.at(i) = W_bias.at(direction, i);
                R_bias_dir.at(i) = R_bias.at(direction, i);
            }

            // Extract peephole parameters
            Matrix P_i, P_f, P_o;
            if (P) {
                P_i = Matrix({hidden_size});
                P_f = Matrix({hidden_size});
                P_o = Matrix({hidden_size});

                for (int i = 0; i < hidden_size; i++) {
                    P_i.at(i) = P_weights.at(direction, i);
                    P_f.at(i) = P_weights.at(direction, i + hidden_size);
                    P_o.at(i) = P_weights.at(direction, i + 2 * hidden_size);
                }
            }

            // Process each time step in the sequence
            for (int idx : seq_indices) {
                // Extract input for current time step
                Matrix x_t({batch_size, input_size});
                for (int b = 0; b < batch_size; b++) {
                    for (int i = 0; i < input_size; i++) {
                        x_t.at(b, i) = X.at(idx, b, i);
                    }
                }

                // Extract gate weights
                Matrix W_i({hidden_size, input_size});
                Matrix W_o({hidden_size, input_size});
                Matrix W_f({hidden_size, input_size});
                Matrix W_c({hidden_size, input_size});

                Matrix R_i({hidden_size, hidden_size});
                Matrix R_o({hidden_size, hidden_size});
                Matrix R_f({hidden_size, hidden_size});
                Matrix R_c({hidden_size, hidden_size});

                for (int i = 0; i < hidden_size; i++) {
                    for (int j = 0; j < input_size; j++) {
                        W_i.at(i, j) = W_dir.at(i, j);
                        W_o.at(i, j) = W_dir.at(i + hidden_size, j);
                        W_f.at(i, j) = W_dir.at(i + 2 * hidden_size, j);
                        W_c.at(i, j) = W_dir.at(i + 3 * hidden_size, j);
                    }

                    for (int j = 0; j < hidden_size; j++) {
                        R_i.at(i, j) = R_dir.at(i, j);
                        R_o.at(i, j) = R_dir.at(i + hidden_size, j);
                        R_f.at(i, j) = R_dir.at(i + 2 * hidden_size, j);
                        R_c.at(i, j) = R_dir.at(i + 3 * hidden_size, j);
                    }
                }

                Matrix Wb_i({batch_size, hidden_size});
                Matrix Wb_o({batch_size, hidden_size});
                Matrix Wb_f({batch_size, hidden_size});
                Matrix Wb_c({batch_size, hidden_size});

                Matrix Rb_i({batch_size, hidden_size});
                Matrix Rb_o({batch_size, hidden_size});
                Matrix Rb_f({batch_size, hidden_size});
                Matrix Rb_c({batch_size, hidden_size});

                for (int b = 0; b < batch_size; b++) {
                    for (int i = 0; i < hidden_size; i++) {
                        Wb_i.at(b, i) = W_bias_dir.at(i);
                        Wb_o.at(b, i) = W_bias_dir.at(i + hidden_size);
                        Wb_f.at(b, i) = W_bias_dir.at(i + 2 * hidden_size);
                        Wb_c.at(b, i) = W_bias_dir.at(i + 3 * hidden_size);

                        Rb_i.at(b, i) = R_bias_dir.at(i);
                        Rb_o.at(b, i) = R_bias_dir.at(i + hidden_size);
                        Rb_f.at(b, i) = R_bias_dir.at(i + 2 * hidden_size);
                        Rb_c.at(b, i) = R_bias_dir.at(i + 3 * hidden_size);
                    }
                }

                // Compute input gate
                Matrix i_t = x_t.matmul(W_i.t()) + Wb_i;
                i_t = i_t + h_t.matmul(R_i.t()) + Rb_i;

                if (P) {
                    // Compute peephole connection effect on input gate
                    Matrix P_i_c_t({batch_size, hidden_size});
                    for (int b = 0; b < batch_size; b++) {
                        for (int h = 0; h < hidden_size; h++) {
                            P_i_c_t.at(b, h) = P_i.at(h) * c_t.at(b, h);
                        }
                    }
                    i_t = i_t + P_i_c_t;
                }
                i_t = i_t.sigmoid();

                // Compute forget gate
                Matrix f_t = x_t.matmul(W_f.t()) + Wb_f;
                f_t = f_t + h_t.matmul(R_f.t()) + Rb_f;

                if (P) {
                    Matrix P_f_c_t({batch_size, hidden_size});
                    for (int b = 0; b < batch_size; b++) {
                        for (int h = 0; h < hidden_size; h++) {
                            P_f_c_t.at(b, h) = P_f.at(h) * c_t.at(b, h);
                        }
                    }
                    f_t = f_t + P_f_c_t;
                }
                f_t = f_t.sigmoid();

                // Compute candidate cell state
                Matrix c_tilde = x_t.matmul(W_c.t()) + Wb_c;
                c_tilde = c_tilde + h_t.matmul(R_c.t()) + Rb_c;
                c_tilde = c_tilde.tanh();

                // Update cell state
                Matrix new_c_t({batch_size, hidden_size});
                for (int b = 0; b < batch_size; b++) {
                    for (int h = 0; h < hidden_size; h++) {
                        new_c_t.at(b, h) = f_t.at(b, h) * c_t.at(b, h) + i_t.at(b, h) * c_tilde.at(b, h);
                    }
                }
                c_t = new_c_t;

                // Compute output gate
                Matrix o_t = x_t.matmul(W_o.t()) + Wb_o;
                o_t = o_t + h_t.matmul(R_o.t()) + Rb_o;

                if (P) {
                    Matrix P_o_c_t({batch_size, hidden_size});
                    for (int b = 0; b < batch_size; b++) {
                        for (int h = 0; h < hidden_size; h++) {
                            P_o_c_t.at(b, h) = P_o.at(h) * c_t.at(b, h);
                        }
                    }
                    o_t = o_t + P_o_c_t;
                }
                o_t = o_t.sigmoid();

                // Compute hidden state
                Matrix tanh_c_t = c_t.tanh();
                Matrix new_h_t({batch_size, hidden_size});
                for (int b = 0; b < batch_size; b++) {
                    for (int h = 0; h < hidden_size; h++) {
                        new_h_t.at(b, h) = o_t.at(b, h) * tanh_c_t.at(b, h);
                    }
                }
                h_t = new_h_t;

                // Handle variable length sequences
                if (sequence_lens) {
                    for (int b = 0; b < batch_size; b++) {
                        if (sequence_lens->size() > b && idx >= (*sequence_lens)[b]) {
                            for (int h = 0; h < hidden_size; h++) {
                                h_t.at(b, h) = h_init.at(direction, b, h);
                                c_t.at(b, h) = c_init.at(direction, b, h);
                            }
                        }
                    }
                }

                for (int b = 0; b < batch_size; b++) {
                    for (int h = 0; h < hidden_size; h++) {
                        Y.at(idx, direction, b, h) = h_t.at(b, h);
                    }
                }
            }

            for (int b = 0; b < batch_size; b++) {
                for (int h = 0; h < hidden_size; h++) {
                    Y_h.at(direction, b, h) = h_t.at(b, h);
                    Y_c.at(direction, b, h) = c_t.at(b, h);
                }
            }
        }

        return std::make_tuple(Y, Y_h, Y_c);
    }
};

// Text recognition post-processing class
class RecPostprocess {
private:
    LSTMImplementation lstm;
    float scale = 0.018496015879112905f; // Quantization scale factor
    std::vector<std::string> characters; // Character dictionary
    std::unordered_map<std::string, Matrix> weight_vec;

public:
    // RecPostprocess(const std::string &dict_path, const std::string &weight_path,
    //                int hidden_size = 64, bool bidirectional = true)
    RecPostprocess(int hidden_size = 64, bool bidirectional = true)
        : lstm(hidden_size, bidirectional) {
            loadDict();

            weight_vec["w1"] = Matrix::loadWeight("w1");
            weight_vec["r1"] = Matrix::loadWeight("r1");
            weight_vec["b1"] = Matrix::loadWeight("b1");
            weight_vec["w2"] = Matrix::loadWeight("w2");
            weight_vec["r2"] = Matrix::loadWeight("r2");
            weight_vec["b2"] = Matrix::loadWeight("b2");
            weight_vec["linear1_w"] = Matrix::loadWeight("linear1_w");
            weight_vec["linear1_b"] = Matrix::loadWeight("linear1_b");
            weight_vec["linear2_w"] = Matrix::loadWeight("linear2_w");
            weight_vec["linear2_b"] = Matrix::loadWeight("linear2_b");
        }

    bool loadDict() {
        characters.clear();
        characters.assign(dict_list.begin(), dict_list.end());
        return true;
    }
    // Load character dictionary
    // bool loadDictionary(const std::string &dict_path)
    // {
    //     characters.clear();
    //     std::ifstream file(dict_path);
    //     if (!file.is_open())
    //     {
    //         std::cerr << "Unable to open dictionary file: " << dict_path << std::endl;
    //         return false;
    //     }

    //     std::string line;
    //     while (std::getline(file, line))
    //     {
    //         line.erase(line.find_last_not_of(" \n\r\t") + 1);
    //         if (!line.empty())
    //         {
    //             characters.push_back(line);
    //         }
    //     }

    //     // Add CTC blank token and space
    //     characters.insert(characters.begin(), "#");

    //     // If dictionary doesn't have space, add space
    //     if (std::find(characters.begin(), characters.end(), " ") == characters.end())
    //     {
    //         characters.push_back(" ");
    //     }

    //     // Add empty string
    //     characters.push_back("");

    //     std::cout << "Loaded " << characters.size() << " characters" << std::endl;
    //     printf("{\n");
    //     // for (auto &d : characters) {
    //     for (int i = 0; i < characters.size(); i++) {
    //         auto &d = characters[i];
    //         printf("\"%s\",", d.c_str());
    //         if (i % 24 == 0 ) {
    //             printf("\n");
    //         }
    //     }
    //     printf("}\n");

    //     return true;
    // }

    // Dequantize data
    Matrix dequantize(const std::vector<int8_t> &data, const std::vector<int> &shape) {
        int total_size = 1;
        for (int dim : shape) {
            total_size *= dim;
        }

        std::vector<float> float_data(total_size);
        for (int i = 0; i < total_size; i++) {
            float_data[i] = static_cast<float>(data[i]) * scale;
        }

        return Matrix::fromArray(float_data.data(), shape);
    }

    // Compute recognition result from convolution output
    Matrix conv2realoutput(const Matrix &conv_output) {
        // Apply hardswish activation function
        Matrix x = conv_output.hardswish();

        // Load first LSTM layer weights
        Matrix &W1 = weight_vec["w1"];
        Matrix &R1 = weight_vec["r1"];
        Matrix &B1 = weight_vec["b1"];

        // Load second LSTM layer weights
        Matrix &W2 = weight_vec["w2"];
        Matrix &R2 = weight_vec["r2"];
        Matrix &B2 = weight_vec["b2"];

        // Load first fully connected layer weights
        Matrix &w1 = weight_vec["linear1_w"];
        Matrix &b1 = weight_vec["linear1_b"];
        // Load second fully connected layer weights
        Matrix &w2 = weight_vec["linear2_w"];
        Matrix &b2 = weight_vec["linear2_b"];

        // Apply max pooling
        x = x.max_pool2d(2, 2, 0);

        // Shape transformation
        x = x.squeeze(2);
        x = x.permute(2, 0, 1);

        Matrix initial_h1({2, 1, 64}, 0.0f);
        Matrix initial_c1({2, 1, 64}, 0.0f);

        // First LSTM layer forward propagation
        Matrix lstm_out1, Y_h1, Y_c1;
        std::tie(lstm_out1, Y_h1, Y_c1) = lstm.forward(x, W1, R1, &B1, nullptr, &initial_h1, &initial_c1);

        // Shape transformation
        x = lstm_out1.permute(0, 2, 1, 3).reshape({320, 1, 128});

        Matrix initial_h2({2, 1, 64}, 0.0f);
        Matrix initial_c2({2, 1, 64}, 0.0f);

        // Second LSTM layer forward propagation
        Matrix lstm_out2, Y_h2, Y_c2;
        std::tie(lstm_out2, Y_h2, Y_c2) = lstm.forward(x, W2, R2, &B2, nullptr, &initial_h2, &initial_c2);

        // Shape transformation
        x = lstm_out2.permute(0, 2, 1, 3).reshape({320, 1, 128}).permute(1, 0, 2);

        // First fully connected layer forward propagation
        x = x.matmul(w1) + b1;

        // Second fully connected layer forward propagation
        x = x.matmul(w2) + b2;

        x = x.softmax(2);

        return x;
    }

    // CTC Decoding
    std::pair<std::string, float> ctcDecode(const Matrix &probs) {
        // if (probs.dim() != 3 || probs.size(0) != 1)
        // {
        //     throw std::runtime_error("Expected input dimension [1, seq_len, num_classes]");
        // }

        int seq_len = probs.size(1);
        int num_classes = probs.size(2);

        std::string text;
        std::vector<float> char_scores;
        int last_index = 0;

        for (int t = 0; t < seq_len; t++)
        {
            // Find index with maximum probability
            float max_prob = -1.0f;
            int max_idx = 0;

            for (int c = 0; c < num_classes; c++)
            {
                float prob = probs.at(0, t, c);
                if (prob > max_prob)
                {
                    max_prob = prob;
                    max_idx = c;
                }
            }

            // If index > 0 and not equal to previous index (remove duplicate characters)
            if (max_idx > 0 && (max_idx != last_index || last_index == 0))
            {
                if (max_idx < static_cast<int>(characters.size()))
                {
                    text += characters[max_idx];
                    char_scores.push_back(max_prob);
                } else {
                    std::cerr << "Warning: Index " << max_idx << " exceeds character dictionary range " << characters.size() << std::endl;
                }
            }

            last_index = max_idx;
        }

        // Calculate average confidence
        float score = 0.0f;
        if (!char_scores.empty())
        {
            score = std::accumulate(char_scores.begin(), char_scores.end(), 0.0f) / char_scores.size();
        }

        return {text, score};
    }

    std::pair<std::string, float> process(const std::string &conv_output_path) {
        // if (!loadDictionary(dict_path))
        // {
        //     return {"", 0.0f};
        // }

        std::ifstream file(conv_output_path, std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "Unable to open convolution output file: " << conv_output_path << std::endl;
            return {"", 0.0f};
        }

        // Get file size
        file.seekg(0, std::ios::end);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        // Read data
        std::vector<int8_t> buffer(size);
        if (!file.read(reinterpret_cast<char *>(buffer.data()), size))
        {
            std::cerr << "Read file fail: " << conv_output_path << std::endl;
            return {"", 0.0f};
        }

        // Dequantize and reshape data
        Matrix conv_output = dequantize(buffer, {1, 2, 640, 512});
        conv_output = conv_output.permute(0, 3, 1, 2);

        // Compute recognition result probability from convolution output
        Matrix model_output = conv2realoutput(conv_output);

        // Perform CTC decoding
        std::string text;
        float confidence;
        std::tie(text, confidence) = ctcDecode(model_output);

        return {text, confidence};
    }

    // Version for processing directly from vector data
    std::pair<std::string, float> process(const std::vector<int8_t> &buffer) {
        // if (!loadDictionary(dict_path))
        // {
        //     return {"", 0.0f};
        // }

        // Dequantize and reshape data
        Matrix conv_output = dequantize(buffer, {1, 2, 640, 512});
        conv_output = conv_output.permute(0, 3, 1, 2);

        // Compute recognition result probability from convolution output
        Matrix model_output = conv2realoutput(conv_output);

        std::string text;
        float confidence;
        std::tie(text, confidence) = ctcDecode(model_output);

        return {text, confidence};
    }
};

#endif // REC_POSTPROCESS_H

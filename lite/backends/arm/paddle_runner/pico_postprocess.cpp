
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <memory>
#include <string>

void int82float(int8_t * input, float * output, int size, float scale)
{
    for(int i = 0; i < size; i ++) {
        output[i] = (float)input[i] * scale; 
    }
}
float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

std::vector<std::unique_ptr<float[]>> process_conv_outputs(
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
    float *conv_output_1x7x7x80)
{

    std::vector<std::unique_ptr<float[]>> outputs;
 
    
    {
        const int height = 52;
        const int width = 52;
        const int size_hw = height * width;
        const int output_size = 1 * size_hw * 80;
        auto output_0 = std::make_unique<float[]>(output_size);

        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                int hw_idx = h * width + w;
                
                
                float sigmoid_val1 = sigmoid(conv_output_1x52x52x1[hw_idx]);
                
                for (int c = 0; c < 80; ++c)
                {
                    
                    int input_idx_80 = hw_idx * 80 + c;
                    float sigmoid_val2 = sigmoid(conv_output_1x52x52x80[input_idx_80]);
                    
                    
                    int output_idx = hw_idx * 80 + c;
                    output_0[output_idx] = std::sqrt(sigmoid_val1 * sigmoid_val2);
                }
            }
        }
        outputs.push_back(std::move(output_0));
    }

    
    {
        const int height = 26;
        const int width = 26;
        const int size_hw = height * width;
        const int output_size = 1 * size_hw * 80;
        auto output_1 = std::make_unique<float[]>(output_size);

        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                int hw_idx = h * width + w;
                float sigmoid_val1 = sigmoid(conv_output_1x26x26x1[hw_idx]);
                
                for (int c = 0; c < 80; ++c)
                {
                    int input_idx_80 = hw_idx * 80 + c;
                    float sigmoid_val2 = sigmoid(conv_output_1x26x26x80[input_idx_80]);
                    
                    int output_idx = hw_idx * 80 + c;
                    output_1[output_idx] = std::sqrt(sigmoid_val1 * sigmoid_val2);
                }
            }
        }
        outputs.push_back(std::move(output_1));
    }

    
    {
        const int height = 13;
        const int width = 13;
        const int size_hw = height * width;
        const int output_size = 1 * size_hw * 80;
        auto output_2 = std::make_unique<float[]>(output_size);

        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                int hw_idx = h * width + w;
                float sigmoid_val1 = sigmoid(conv_output_1x13x13x1[hw_idx]);
                
                for (int c = 0; c < 80; ++c)
                {
                    int input_idx_80 = hw_idx * 80 + c;
                    float sigmoid_val2 = sigmoid(conv_output_1x13x13x80[input_idx_80]);
                    
                    int output_idx = hw_idx * 80 + c;
                    output_2[output_idx] = std::sqrt(sigmoid_val1 * sigmoid_val2);
                }
            }
        }
        outputs.push_back(std::move(output_2));
    }

    
    {
        const int height = 7;
        const int width = 7;
        const int size_hw = height * width;
        const int output_size = 1 * size_hw * 80;
        auto output_3 = std::make_unique<float[]>(output_size);

        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                int hw_idx = h * width + w;
                float sigmoid_val1 = sigmoid(conv_output_1x7x7x1[hw_idx]);
                
                for (int c = 0; c < 80; ++c)
                {
                    int input_idx_80 = hw_idx * 80 + c;
                    float sigmoid_val2 = sigmoid(conv_output_1x7x7x80[input_idx_80]);
                    
                    int output_idx = hw_idx * 80 + c;
                    output_3[output_idx] = std::sqrt(sigmoid_val1 * sigmoid_val2);
                }
            }
        }
        outputs.push_back(std::move(output_3));
    }

    
    
    {
        const int height = 52;
        const int width = 52;
        const int size_hw = height * width;
        const int output_size = 1 * 32 * size_hw;
        auto output_4 = std::make_unique<float[]>(output_size);

        
        for (int c = 0; c < 32; ++c)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    int hw_idx = h * width + w;
                    
                    
                    int input_idx = c * size_hw + hw_idx;
                    
                    
                    int output_idx = c * size_hw + hw_idx;
                    output_4[output_idx] = sigmoid(conv_output_1x52x52x32[input_idx]);
                }
            }
        }
        outputs.push_back(std::move(output_4));
    }

    
    {
        const int height = 26;
        const int width = 26;
        const int size_hw = height * width;
        const int output_size = 1 * 32 * size_hw;
        auto output_5 = std::make_unique<float[]>(output_size);

        for (int c = 0; c < 32; ++c)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    int hw_idx = h * width + w;
                    
                    
                    int input_idx = c * size_hw + hw_idx;
                    
                    int output_idx = c * size_hw + hw_idx;
                    output_5[output_idx] = sigmoid(conv_output_1x26x26x32[input_idx]);
                }
            }
        }
        outputs.push_back(std::move(output_5));
    }

    
    {
        const int height = 13;
        const int width = 13;
        const int size_hw = height * width;
        const int output_size = 1 * 32 * size_hw;
        auto output_6 = std::make_unique<float[]>(output_size);

        for (int c = 0; c < 32; ++c)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    int hw_idx = h * width + w;
                    
                    
                    int input_idx = c * size_hw + hw_idx;
                    
                    int output_idx = c * size_hw + hw_idx;
                    output_6[output_idx] = sigmoid(conv_output_1x13x13x32[input_idx]);
                }
            }
        }
        outputs.push_back(std::move(output_6));
    }

    
    {
        const int height = 7;
        const int width = 7;
        const int size_hw = height * width;
        const int output_size = 1 * 32 * size_hw;
        auto output_7 = std::make_unique<float[]>(output_size);

        for (int c = 0; c < 32; ++c)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    int hw_idx = h * width + w;
                    
                    
                    int input_idx = c * size_hw + hw_idx;
                    
                    int output_idx = c * size_hw + hw_idx;
                    output_7[output_idx] = sigmoid(conv_output_1x7x7x32[input_idx]);
                }
            }
        }
        outputs.push_back(std::move(output_7));
    }
    return outputs;
}


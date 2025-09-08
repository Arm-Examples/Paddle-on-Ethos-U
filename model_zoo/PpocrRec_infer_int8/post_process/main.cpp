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

#include <iostream>
#include <string>
#include <chrono>
#include "rec_postprocess.h"

void printUsage(const char *programName) {
    std::cout << "Usage: " << programName << " --output_path <output path> --dict_path <dict paty>" << std::endl;
}

int main(int argc, char *argv[]) {
    std::string output_path;
    std::string dict_path;
    std::string weight_path;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--output_path" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--dict_path" && i + 1 < argc) {
            dict_path = argv[++i];
        } else {
            std::cerr << "Unknown Params: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    if (output_path.empty() || dict_path.empty()) {
        std::cerr << "Missing necessary params" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    std::cout << "Processing output: " << output_path << std::endl;
    std::cout << "Use dict: " << dict_path << std::endl;

    RecPostprocess processor;

    // Record starting time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Postprocess
    std::string text;
    float confidence;
    std::tie(text, confidence) = processor.process(output_path);

    // Record ending time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "\nRec Result: " << text << std::endl;
    std::cout << "Confidence: " << confidence << std::endl;
    std::cout << "Cost: " << duration << " ms" << std::endl;

    return 0;
}

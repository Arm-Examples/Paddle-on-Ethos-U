/*
 * Copyright (c) 2022 Arm Limited.
 *
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

#ifndef TEST_ASSERTIONS_H
#define TEST_ASSERTIONS_H

#include <stddef.h>
#include <stdio.h>

namespace {
template <typename... Args>
std::string string_format(std::ostringstream &stringStream) {
    return stringStream.str();
}

template <typename T, typename... Args>
std::string string_format(std::ostringstream &stringStream, T t, Args... args) {
    stringStream << t;
    return string_format(stringStream, args...);
}

class TestFailureException : public std::exception {
public:
    template <typename... Args>
    TestFailureException(const char *msg, Args... args) {
        std::ostringstream stringStream;
        this->msg = string_format(stringStream, msg, args...);
    }
    const char *what() const throw() {
        return msg.c_str();
    }

private:
    std::string msg;
};
} // namespace

#define TEST_ASSERT(v)                                                                             \
    do {                                                                                           \
        if (!(v)) {                                                                                \
            throw TestFailureException(__FILE__, ":", __LINE__, " ERROR test failed: '", #v, "'"); \
        }                                                                                          \
    } while (0)

#define FAIL() TEST_ASSERT(false)

#endif

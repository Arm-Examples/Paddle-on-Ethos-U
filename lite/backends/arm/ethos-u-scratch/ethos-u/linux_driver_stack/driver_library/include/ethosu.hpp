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

#pragma once

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

/*
 *The following undef are necessary to avoid clash with macros in GNU C Library
 * if removed the following warning/error are produced:
 *
 *  In the GNU C Library, "major" ("minor") is defined
 *  by <sys/sysmacros.h>. For historical compatibility, it is
 *  currently defined by <sys/types.h> as well, but we plan to
 *  remove this soon. To use "major" ("minor"), include <sys/sysmacros.h>
 *  directly. If you did not intend to use a system-defined macro
 *  "major" ("minor"), you should undefine it after including <sys/types.h>.
 */
#undef major
#undef minor

namespace EthosU {

constexpr uint32_t DRIVER_LIBRARY_VERSION_MAJOR = 4;
constexpr uint32_t DRIVER_LIBRARY_VERSION_MINOR = 0;
constexpr uint32_t DRIVER_LIBRARY_VERSION_PATCH = 0;

constexpr uint32_t MAX_SUPPORTED_KERNEL_DRIVER_MAJOR_VERSION = 4;
constexpr uint32_t MIN_SUPPORTED_KERNEL_DRIVER_MAJOR_VERSION = 4;

class Exception : public std::exception {
public:
    Exception(const char *msg);
    virtual ~Exception() throw();
    virtual const char *what() const throw();

private:
    std::string msg;
};

/**
 * Sematic Version : major.minor.patch
 */
class SemanticVersion {
public:
    SemanticVersion(uint32_t _major = 0, uint32_t _minor = 0, uint32_t _patch = 0) :
        major(_major), minor(_minor), patch(_patch){};

    bool operator==(const SemanticVersion &other);
    bool operator<(const SemanticVersion &other);
    bool operator<=(const SemanticVersion &other);
    bool operator!=(const SemanticVersion &other);
    bool operator>(const SemanticVersion &other);
    bool operator>=(const SemanticVersion &other);

    uint32_t major;
    uint32_t minor;
    uint32_t patch;
};

std::ostream &operator<<(std::ostream &out, const SemanticVersion &v);

const SemanticVersion getLibraryVersion();

/*
 * Hardware Identifier
 * @versionStatus:             Version status
 * @version:                   Version revision
 * @product:                   Product revision
 * @architecture:              Architecture revison
 */
struct HardwareId {
public:
    HardwareId(uint32_t _versionStatus              = 0,
               const SemanticVersion &_version      = SemanticVersion(),
               const SemanticVersion &_product      = SemanticVersion(),
               const SemanticVersion &_architecture = SemanticVersion()) :
        versionStatus(_versionStatus),
        version(_version), product(_product), architecture(_architecture) {}

    uint32_t versionStatus;
    SemanticVersion version;
    SemanticVersion product;
    SemanticVersion architecture;
};

/*
 * Hardware Configuration
 * @macsPerClockCycle:         MACs per clock cycle
 * @cmdStreamVersion:          NPU command stream version
 * @type:                      NPU device type
 * @customDma:                 Custom DMA enabled
 */
struct HardwareConfiguration {
public:
    HardwareConfiguration(uint32_t _macsPerClockCycle = 0,
                          uint32_t _cmdStreamVersion  = 0,
                          uint32_t _type              = static_cast<uint32_t>(DeviceType::UNKNOWN),
                          bool _customDma             = false) :
        macsPerClockCycle(_macsPerClockCycle),
        cmdStreamVersion(_cmdStreamVersion), customDma(_customDma) {

        if (_type > static_cast<uint32_t>(DeviceType::DIRECT)) {
            throw EthosU::Exception(std::string("Invalid device type: ").append(std::to_string(_type)).c_str());
        }

        type = static_cast<DeviceType>(_type);
    }

    enum class DeviceType {
        UNKNOWN = 0,
        SUBSYSTEM,
        DIRECT,
    };

    uint32_t macsPerClockCycle;
    uint32_t cmdStreamVersion;
    DeviceType type;
    bool customDma;
};

std::ostream &operator<<(std::ostream &out, const HardwareConfiguration::DeviceType &deviceType);

/**
 * Device capabilities
 * @hwId:                      Hardware
 * @driver:                    Driver revision
 * @hwCfg                      Hardware configuration
 */
class Capabilities {
public:
    Capabilities() {}
    Capabilities(const HardwareId &_hwId, const HardwareConfiguration &_hwCfg, const SemanticVersion &_driver) :
        hwId(_hwId), hwCfg(_hwCfg), driver(_driver) {}

    HardwareId hwId;
    HardwareConfiguration hwCfg;
    SemanticVersion driver;
};

class Device {
public:
    Device(const char *device = "/dev/ethosu0");
    virtual ~Device() noexcept(false);

    int ioctl(unsigned long cmd, void *data = nullptr) const;
    Capabilities capabilities() const;
    const SemanticVersion &getDriverVersion() const;

private:
    int fd;
    SemanticVersion driverVersion;
};

class Buffer {
public:
    Buffer(const Device &device, const size_t size);
    virtual ~Buffer() noexcept(false);

    void clear() const;
    char *data() const;
    size_t size() const;

    int getFd() const;

private:
    int fd;
    char *dataPtr;
    const size_t dataSize;
};

class Network {
public:
    Network(const Device &device, const unsigned char *networkData, size_t networkSize);
    Network(const Device &device, const unsigned index);
    virtual ~Network() noexcept(false);

    int ioctl(unsigned long cmd, void *data = nullptr);
    const std::vector<size_t> &getIfmDims() const;
    size_t getIfmSize() const;
    const std::vector<size_t> &getOfmDims() const;
    size_t getOfmSize() const;

private:
    void collectNetworkInfo();

    int fd;
    std::vector<size_t> ifmDims;
    std::vector<size_t> ofmDims;
};

enum class InferenceStatus {
    OK,
    ERROR,
    RUNNING,
    REJECTED,
    ABORTED,
    ABORTING,
};

std::ostream &operator<<(std::ostream &out, const InferenceStatus &v);

class Inference {
public:
    template <typename T>
    Inference(const std::shared_ptr<Network> &network,
              const T &ifmBegin,
              const T &ifmEnd,
              const T &ofmBegin,
              const T &ofmEnd) :
        network(network) {
        std::copy(ifmBegin, ifmEnd, std::back_inserter(ifmBuffers));
        std::copy(ofmBegin, ofmEnd, std::back_inserter(ofmBuffers));
        std::vector<uint32_t> counterConfigs = initializeCounterConfig();

        create(counterConfigs, false);
    }
    template <typename T, typename U>
    Inference(const std::shared_ptr<Network> &network,
              const T &ifmBegin,
              const T &ifmEnd,
              const T &ofmBegin,
              const T &ofmEnd,
              const U &counters,
              bool enableCycleCounter) :
        network(network) {
        std::copy(ifmBegin, ifmEnd, std::back_inserter(ifmBuffers));
        std::copy(ofmBegin, ofmEnd, std::back_inserter(ofmBuffers));
        std::vector<uint32_t> counterConfigs = initializeCounterConfig();

        if (counters.size() > counterConfigs.size())
            throw EthosU::Exception("PMU Counters argument to large.");

        std::copy(counters.begin(), counters.end(), counterConfigs.begin());
        create(counterConfigs, enableCycleCounter);
    }

    virtual ~Inference() noexcept(false);

    bool wait(int64_t timeoutNanos = -1) const;
    const std::vector<uint64_t> getPmuCounters() const;
    uint64_t getCycleCounter() const;
    bool cancel() const;
    InferenceStatus status() const;
    int getFd() const;
    const std::shared_ptr<Network> getNetwork() const;
    std::vector<std::shared_ptr<Buffer>> &getIfmBuffers();
    std::vector<std::shared_ptr<Buffer>> &getOfmBuffers();

    static uint32_t getMaxPmuEventCounters();

private:
    void create(std::vector<uint32_t> &counterConfigs, bool enableCycleCounter);
    std::vector<uint32_t> initializeCounterConfig();

    int fd;
    const std::shared_ptr<Network> network;
    std::vector<std::shared_ptr<Buffer>> ifmBuffers;
    std::vector<std::shared_ptr<Buffer>> ofmBuffers;
};

} // namespace EthosU


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

#include <ethosu.hpp>
#include <uapi/ethosu.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <utility>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

using namespace std;

namespace {
std::string driverVersionToString(const EthosU::ethosu_uapi_kernel_driver_version &version) {
    return std::to_string(version.major) + "." + std::to_string(version.minor) + "." + std::to_string(version.patch);
}

std::string cmdToString(unsigned long cmd) {
    // Needed for the struct types in the macro expansions
    using namespace EthosU;

    switch (cmd) {
    case ETHOSU_IOCTL_PING:
        return "PING";
    case ETHOSU_IOCTL_CAPABILITIES_REQ:
        return "CAPABILITIES_REQ";
    case ETHOSU_IOCTL_DRIVER_VERSION_GET:
        return "DRIVER_VERSION_GET";
    case ETHOSU_IOCTL_BUFFER_CREATE:
        return "BUFFER_CREATE";
    case ETHOSU_IOCTL_NETWORK_CREATE:
        return "NETWORK_CREATE";
    case ETHOSU_IOCTL_NETWORK_INFO:
        return "NETWORK_INFO";
    case ETHOSU_IOCTL_INFERENCE_CREATE:
        return "INFERENCE_CREATE";
    case ETHOSU_IOCTL_INFERENCE_STATUS:
        return "INFERENCE_STATUS";
    case ETHOSU_IOCTL_INFERENCE_CANCEL:
        return "INFERENCE_CANCEL";
    default:
        return std::string("Unknown command: ").append(to_string(cmd));
    }
}

enum class Severity { Error, Warning, Info, Debug };

class Log {
public:
    Log(const Severity _severity = Severity::Error) : severity(_severity) {}

    ~Log() = default;

    template <typename T>
    const Log &operator<<(const T &d) const {
        if (level >= severity) {
            cout << d;
        }

        return *this;
    }

    const Log &operator<<(ostream &(*manip)(ostream &)) const {
        if (level >= severity) {
            manip(cout);
        }

        return *this;
    }

private:
    static Severity getLogLevel() {
        if (const char *e = getenv("ETHOSU_LOG_LEVEL")) {
            const string env(e);

            if (env == "Error") {
                return Severity::Error;
            } else if (env == "Warning") {
                return Severity::Warning;
            } else if (env == "Info") {
                return Severity::Info;
            } else if (env == "Debug") {
                return Severity::Debug;
            } else {
                cerr << "Unsupported log level '" << env << "'" << endl;
            }
        }

        return Severity::Warning;
    }

    static const Severity level;
    const Severity severity;
};

const Severity Log::level = Log::getLogLevel();

} // namespace

namespace EthosU {

const SemanticVersion getLibraryVersion() {
    return {DRIVER_LIBRARY_VERSION_MAJOR, DRIVER_LIBRARY_VERSION_MINOR, DRIVER_LIBRARY_VERSION_PATCH};
}

__attribute__((weak)) int eioctl(int fd, unsigned long cmd, void *data = nullptr) {
    int ret = ::ioctl(fd, cmd, data);
    if (ret < 0) {
        throw EthosU::Exception(string("IOCTL cmd=").append(cmdToString(cmd) + " failed: " + strerror(errno)).c_str());
    }

    Log(Severity::Debug) << "ioctl. fd=" << fd << ", cmd=" << cmdToString(cmd) << ", ret=" << ret << endl;

    return ret;
}

__attribute__((weak)) int eopen(const char *pathname, int flags) {
    int fd = ::open(pathname, flags);
    if (fd < 0) {
        throw Exception("Failed to open device");
    }

    Log(Severity::Debug) << "open. fd=" << fd << ", path='" << pathname << "', flags=" << flags << endl;

    return fd;
}

__attribute__((weak)) int
eppoll(struct pollfd *fds, nfds_t nfds, const struct timespec *tmo_p, const sigset_t *sigmask) {
    int result = ::ppoll(fds, nfds, tmo_p, sigmask);
    if (result < 0) {
        throw Exception("Failed to wait for ppoll event or signal");
    }

    return result;
}

__attribute__((weak)) int eclose(int fd) {
    Log(Severity::Debug) << "close. fd=" << fd << endl;

    int result = ::close(fd);
    if (result < 0) {
        throw Exception("Failed to close file");
    }

    return result;
}
__attribute((weak)) void *emmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
    void *ptr = ::mmap(addr, length, prot, flags, fd, offset);
    if (ptr == MAP_FAILED) {
        throw Exception("Failed to mmap file");
    }

    Log(Severity::Debug) << "map. fd=" << fd << ", addr=" << setfill('0') << addr << ", length=" << dec << length
                         << ", ptr=" << hex << ptr << endl;

    return ptr;
}

__attribute__((weak)) int emunmap(void *addr, size_t length) {
    Log(Severity::Debug) << "unmap. addr=" << setfill('0') << addr << ", length=" << dec << length << endl;

    int result = ::munmap(addr, length);
    if (result < 0) {
        throw Exception("Failed to munmap file");
    }

    return result;
}

} // namespace EthosU

namespace EthosU {

/****************************************************************************
 * Exception
 ****************************************************************************/

Exception::Exception(const char *msg) : msg(msg) {}

Exception::~Exception() throw() {}

const char *Exception::what() const throw() {
    return msg.c_str();
}

/****************************************************************************
 * Semantic Version
 ****************************************************************************/

bool SemanticVersion::operator==(const SemanticVersion &other) {
    return other.major == major && other.minor == minor && other.patch == patch;
}

bool SemanticVersion::operator<(const SemanticVersion &other) {
    if (other.major > major)
        return true;
    if (other.minor > minor)
        return true;
    return other.patch > patch;
}

bool SemanticVersion::operator<=(const SemanticVersion &other) {
    return *this < other || *this == other;
}

bool SemanticVersion::operator!=(const SemanticVersion &other) {
    return !(*this == other);
}

bool SemanticVersion::operator>(const SemanticVersion &other) {
    return !(*this <= other);
}

bool SemanticVersion::operator>=(const SemanticVersion &other) {
    return !(*this < other);
}

ostream &operator<<(ostream &out, const SemanticVersion &v) {
    return out << "{ major=" << unsigned(v.major) << ", minor=" << unsigned(v.minor) << ", patch=" << unsigned(v.patch)
               << " }";
}

/****************************************************************************
 * Device
 ****************************************************************************/
Device::Device(const char *device) : fd(eopen(device, O_RDWR | O_NONBLOCK)) {
    ethosu_uapi_kernel_driver_version version = {};

    Log(Severity::Info) << "Device(\"" << device << "\"). this=" << this << ", fd=" << fd << endl;

    try {
        eioctl(fd, ETHOSU_IOCTL_DRIVER_VERSION_GET, &version);

        if (MAX_SUPPORTED_KERNEL_DRIVER_MAJOR_VERSION < version.major ||
            MIN_SUPPORTED_KERNEL_DRIVER_MAJOR_VERSION > version.major) {
            throw Exception(
                std::string("Unsupported kernel driver version: ").append(driverVersionToString(version)).c_str());
        }
    } catch (std::exception &e) {
        try {
            eclose(fd);
        } catch (...) { std::throw_with_nested(e); }
        throw;
    }

    driverVersion = {version.major, version.minor, version.patch};
}

Device::~Device() noexcept(false) {
    eclose(fd);
    Log(Severity::Info) << "~Device(). this=" << this << endl;
}

int Device::ioctl(unsigned long cmd, void *data) const {
    return eioctl(fd, cmd, data);
}

const SemanticVersion &Device::getDriverVersion() const {
    return driverVersion;
}

Capabilities Device::capabilities() const {
    ethosu_uapi_device_capabilities uapi;
    (void)eioctl(fd, ETHOSU_IOCTL_CAPABILITIES_REQ, static_cast<void *>(&uapi));

    Capabilities capabilities(
        HardwareId(uapi.hw_id.version_status,
                   SemanticVersion(uapi.hw_id.version_major, uapi.hw_id.version_minor),
                   SemanticVersion(uapi.hw_id.product_major),
                   SemanticVersion(uapi.hw_id.arch_major_rev, uapi.hw_id.arch_minor_rev, uapi.hw_id.arch_patch_rev)),
        HardwareConfiguration(
            uapi.hw_cfg.macs_per_cc, uapi.hw_cfg.cmd_stream_version, uapi.hw_cfg.type, bool(uapi.hw_cfg.custom_dma)),
        SemanticVersion(uapi.driver_major_rev, uapi.driver_minor_rev, uapi.driver_patch_rev));
    return capabilities;
}

ostream &operator<<(ostream &out, const HardwareConfiguration::DeviceType &deviceType) {
    switch (deviceType) {
    case HardwareConfiguration::DeviceType::UNKNOWN:
        return out << "unknown";
    case HardwareConfiguration::DeviceType::SUBSYSTEM:
        return out << "subsystem";
    case HardwareConfiguration::DeviceType::DIRECT:
        return out << "direct";
    default:
        throw Exception(string("Invalid device type: ").append(to_string(static_cast<uint32_t>(deviceType))).c_str());
    }
}

static_assert(static_cast<uint32_t>(HardwareConfiguration::DeviceType::UNKNOWN) == ETHOSU_UAPI_DEVICE_UNKNOWN &&
                  static_cast<uint32_t>(HardwareConfiguration::DeviceType::SUBSYSTEM) == ETHOSU_UAPI_DEVICE_SUBSYSTEM &&
                  static_cast<uint32_t>(HardwareConfiguration::DeviceType::DIRECT) == ETHOSU_UAPI_DEVICE_DIRECT,
              "DeviceType enums values doesn't match UAPI");

/****************************************************************************
 * Buffer
 ****************************************************************************/

Buffer::Buffer(const Device &device, const size_t size) : fd(-1), dataPtr(nullptr), dataSize(size) {
    ethosu_uapi_buffer_create uapi = {static_cast<uint32_t>(dataSize)};
    fd                             = device.ioctl(ETHOSU_IOCTL_BUFFER_CREATE, static_cast<void *>(&uapi));

    void *d;
    try {
        d = emmap(nullptr, dataSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    } catch (std::exception &e) {
        try {
            eclose(fd);
        } catch (...) { std::throw_with_nested(e); }
        throw;
    }

    dataPtr = reinterpret_cast<char *>(d);

    Log(Severity::Info) << "Buffer(" << &device << ", " << dec << size << "), this=" << this << ", fd=" << fd
                        << ", dataPtr=" << static_cast<void *>(dataPtr) << endl;
}

Buffer::~Buffer() noexcept(false) {
    try {
        emunmap(dataPtr, dataSize);
    } catch (std::exception &e) {
        try {
            eclose(fd);
        } catch (...) { std::throw_with_nested(e); }
        throw;
    }

    eclose(fd);

    Log(Severity::Info) << "~Buffer(). this=" << this << endl;
}

void Buffer::clear() const {
    memset(dataPtr, 0, dataSize);
}

char *Buffer::data() const {
    return dataPtr;
}

size_t Buffer::size() const {
    return dataSize;
}

int Buffer::getFd() const {
    return fd;
}

/****************************************************************************
 * Network
 ****************************************************************************/

Network::Network(const Device &device, const unsigned char *networkData, size_t networkSize) : fd(-1) {
    // Create buffer handle
    ethosu_uapi_network_create uapi;
    uapi.type             = ETHOSU_UAPI_NETWORK_USER_BUFFER;
    uapi.network.data_ptr = reinterpret_cast<uintptr_t>(networkData);
    uapi.network.size     = networkSize;
    fd                    = device.ioctl(ETHOSU_IOCTL_NETWORK_CREATE, static_cast<void *>(&uapi));
    try {
        collectNetworkInfo();
    } catch (std::exception &e) {
        try {
            eclose(fd);
        } catch (...) { std::throw_with_nested(e); }
        throw;
    }

    Log(Severity::Info) << "Network(" << &device << "), this=" << this << ", fd=" << fd << endl;
}

Network::Network(const Device &device, const unsigned index) : fd(-1) {
    // Create buffer handle
    ethosu_uapi_network_create uapi;
    uapi.type  = ETHOSU_UAPI_NETWORK_INDEX;
    uapi.index = index;
    fd         = device.ioctl(ETHOSU_IOCTL_NETWORK_CREATE, static_cast<void *>(&uapi));
    try {
        collectNetworkInfo();
    } catch (std::exception &e) {
        try {
            eclose(fd);
        } catch (...) { std::throw_with_nested(e); }
        throw;
    }

    Log(Severity::Info) << "Network(" << &device << ", " << index << "), this=" << this << ", fd=" << fd << endl;
}

void Network::collectNetworkInfo() {
    ethosu_uapi_network_info info;
    ioctl(ETHOSU_IOCTL_NETWORK_INFO, static_cast<void *>(&info));

    for (uint32_t i = 0; i < info.ifm_count; i++) {
        ifmDims.push_back(info.ifm_size[i]);
    }

    for (uint32_t i = 0; i < info.ofm_count; i++) {
        ofmDims.push_back(info.ofm_size[i]);
    }
}

Network::~Network() noexcept(false) {
    eclose(fd);
    Log(Severity::Info) << "~Network(). this=" << this << endl;
}

int Network::ioctl(unsigned long cmd, void *data) {
    return eioctl(fd, cmd, data);
}

const std::vector<size_t> &Network::getIfmDims() const {
    return ifmDims;
}

size_t Network::getIfmSize() const {
    size_t size = 0;

    for (auto s : ifmDims) {
        size += s;
    }

    return size;
}

const std::vector<size_t> &Network::getOfmDims() const {
    return ofmDims;
}

size_t Network::getOfmSize() const {
    size_t size = 0;

    for (auto s : ofmDims) {
        size += s;
    }

    return size;
}

/****************************************************************************
 * Inference
 ****************************************************************************/

ostream &operator<<(ostream &out, const InferenceStatus &status) {
    switch (status) {
    case InferenceStatus::OK:
        return out << "ok";
    case InferenceStatus::ERROR:
        return out << "error";
    case InferenceStatus::RUNNING:
        return out << "running";
    case InferenceStatus::REJECTED:
        return out << "rejected";
    case InferenceStatus::ABORTED:
        return out << "aborted";
    case InferenceStatus::ABORTING:
        return out << "aborting";
    }
    throw Exception("Unknown inference status");
}

Inference::~Inference() noexcept(false) {
    eclose(fd);
    Log(Severity::Info) << "~Inference(). this=" << this << endl;
}

void Inference::create(std::vector<uint32_t> &counterConfigs, bool cycleCounterEnable = false) {
    ethosu_uapi_inference_create uapi;

    if (ifmBuffers.size() > ETHOSU_FD_MAX) {
        throw Exception("IFM buffer overflow");
    }

    if (ofmBuffers.size() > ETHOSU_FD_MAX) {
        throw Exception("OFM buffer overflow");
    }

    if (counterConfigs.size() != ETHOSU_PMU_EVENT_MAX) {
        throw Exception("Wrong size of counter configurations");
    }

    uapi.ifm_count = 0;
    for (auto it : ifmBuffers) {
        uapi.ifm_fd[uapi.ifm_count++] = it->getFd();
    }

    uapi.ofm_count = 0;
    for (auto it : ofmBuffers) {
        uapi.ofm_fd[uapi.ofm_count++] = it->getFd();
    }

    for (int i = 0; i < ETHOSU_PMU_EVENT_MAX; i++) {
        uapi.pmu_config.events[i] = counterConfigs[i];
    }

    uapi.pmu_config.cycle_count = cycleCounterEnable;

    fd = network->ioctl(ETHOSU_IOCTL_INFERENCE_CREATE, static_cast<void *>(&uapi));

    Log(Severity::Info) << "Inference(" << &*network << "), this=" << this << ", fd=" << fd << endl;
}

std::vector<uint32_t> Inference::initializeCounterConfig() {
    return std::vector<uint32_t>(ETHOSU_PMU_EVENT_MAX, 0);
}

uint32_t Inference::getMaxPmuEventCounters() {
    return ETHOSU_PMU_EVENT_MAX;
}

bool Inference::wait(int64_t timeoutNanos) const {
    struct pollfd pfd;
    pfd.fd      = fd;
    pfd.events  = POLLIN | POLLERR;
    pfd.revents = 0;

    // if timeout negative wait forever
    if (timeoutNanos < 0) {
        return eppoll(&pfd, 1, NULL, NULL) == 0;
    }

    struct timespec tmo_p;
    int64_t nanosec = 1000000000;
    tmo_p.tv_sec    = timeoutNanos / nanosec;
    tmo_p.tv_nsec   = timeoutNanos % nanosec;

    return eppoll(&pfd, 1, &tmo_p, NULL) == 0;
}

bool Inference::cancel() const {
    ethosu_uapi_cancel_inference_status uapi;
    eioctl(fd, ETHOSU_IOCTL_INFERENCE_CANCEL, static_cast<void *>(&uapi));
    return uapi.status == ETHOSU_UAPI_STATUS_OK;
}

InferenceStatus Inference::status() const {
    ethosu_uapi_result_status uapi;

    eioctl(fd, ETHOSU_IOCTL_INFERENCE_STATUS, static_cast<void *>(&uapi));

    switch (uapi.status) {
    case ETHOSU_UAPI_STATUS_OK:
        return InferenceStatus::OK;
    case ETHOSU_UAPI_STATUS_ERROR:
        return InferenceStatus::ERROR;
    case ETHOSU_UAPI_STATUS_RUNNING:
        return InferenceStatus::RUNNING;
    case ETHOSU_UAPI_STATUS_REJECTED:
        return InferenceStatus::REJECTED;
    case ETHOSU_UAPI_STATUS_ABORTED:
        return InferenceStatus::ABORTED;
    case ETHOSU_UAPI_STATUS_ABORTING:
        return InferenceStatus::ABORTING;
    }

    throw Exception("Unknown inference status");
}

const std::vector<uint64_t> Inference::getPmuCounters() const {
    ethosu_uapi_result_status uapi;
    std::vector<uint64_t> counterValues = std::vector<uint64_t>(ETHOSU_PMU_EVENT_MAX, 0);

    eioctl(fd, ETHOSU_IOCTL_INFERENCE_STATUS, static_cast<void *>(&uapi));

    for (int i = 0; i < ETHOSU_PMU_EVENT_MAX; i++) {
        if (uapi.pmu_config.events[i]) {
            counterValues.at(i) = uapi.pmu_count.events[i];
        }
    }

    return counterValues;
}

uint64_t Inference::getCycleCounter() const {
    ethosu_uapi_result_status uapi;

    eioctl(fd, ETHOSU_IOCTL_INFERENCE_STATUS, static_cast<void *>(&uapi));

    return uapi.pmu_count.cycle_count;
}

int Inference::getFd() const {
    return fd;
}

const shared_ptr<Network> Inference::getNetwork() const {
    return network;
}

vector<shared_ptr<Buffer>> &Inference::getIfmBuffers() {
    return ifmBuffers;
}

vector<shared_ptr<Buffer>> &Inference::getOfmBuffers() {
    return ofmBuffers;
}

static_assert(MAX_SUPPORTED_KERNEL_DRIVER_MAJOR_VERSION >= ETHOSU_KERNEL_DRIVER_VERSION_MAJOR &&
                  MIN_SUPPORTED_KERNEL_DRIVER_MAJOR_VERSION <= ETHOSU_KERNEL_DRIVER_VERSION_MAJOR,
              "Unsupported major kernel driver version in UAPI");
} // namespace EthosU

//
// SPDX-FileCopyrightText: Copyright 2020, 2022-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
//
%module driver
%{
#define SWIG_FILE_WITH_INIT
%}

//typemap definitions and other common stuff
%include "standard_header.i"

%{
#include "ethosu.hpp"
#include <fstream>
#include <list>
#include <string>
#include <cstring>
#include <sstream>
#include <linux/ioctl.h>

#define ETHOSU_IOCTL_BASE               0x01
#define ETHOSU_IO(nr)                   _IO(ETHOSU_IOCTL_BASE, nr)
#define ETHOSU_IOCTL_PING               ETHOSU_IO(0x00)

%}
%include <typemaps/buffer.i>

%shared_ptr(EthosU::Buffer);
%shared_ptr(EthosU::Network);

%typemap(out) (std::vector<uint64_t>) {
    PyObject *list = PyList_New($1.size());
    for (size_t i=0; i < $1.size(); ++i) {
        PyList_SET_ITEM(list, i, PyLong_FromUnsignedLong($1.at(i)));
    }
    $result = list;
}

namespace std {
   %template(UintVector) vector<unsigned int>;
   %template(SizeTVector) vector<size_t>;
   %template(SharedBufferVector) vector<shared_ptr<EthosU::Buffer>>;
}

namespace EthosU
{

constexpr uint32_t DRIVER_LIBRARY_VERSION_MAJOR;
constexpr uint32_t DRIVER_LIBRARY_VERSION_MINOR;
constexpr uint32_t DRIVER_LIBRARY_VERSION_PATCH;

constexpr uint32_t MAX_SUPPORTED_KERNEL_DRIVER_MAJOR_VERSION;
constexpr uint32_t MIN_SUPPORTED_KERNEL_DRIVER_MAJOR_VERSION;

%feature("docstring",
"
Semantic Version : major.minor.patch
") SemanticVersion;
%nodefaultctor SemanticVersion;
class SemanticVersion {
public:
    SemanticVersion(uint32_t major = 0, uint32_t minor = 0, uint32_t patch = 0);

    uint32_t major;
    uint32_t minor;
    uint32_t patch;
};

%extend SemanticVersion {
    std::string __str__() const {
        std::ostringstream out;
        out << *$self;
        return out.str();
    }
}

%feature("docstring",
"
    Return driver library version information.

    Returns:
        SemanticVersion: driver library version.
") getLibraryVersion;
const SemanticVersion getLibraryVersion();

%feature("docstring",
"
Hardware Identifier which consists of version status, version revision, product revision and architecture revision.
") HardwareId;
class HardwareId {
public:
    HardwareId(uint32_t versionStatus, SemanticVersion& version, SemanticVersion& product, SemanticVersion& arch);

    uint32_t versionStatus{0};
    SemanticVersion version{};
    SemanticVersion product{};
    SemanticVersion architecture{};
};

%extend HardwareId {
    std::string __str__() const {
        std::ostringstream out;
        out << "{versionStatus=" << $self->versionStatus <<
        ", version=" << EthosU_SemanticVersion___str__(&$self->version) <<
        ", product=" << EthosU_SemanticVersion___str__(&$self->product) <<
        ", architecture=" << EthosU_SemanticVersion___str__(&$self->architecture) << "}";
        return out.str();
    }
}

%feature("docstring",
"
Hardware Configuration object defines specific configuration including MACs per clock cycle and NPU command stream
version. This also specifies is custom DMA is enabled or not.
") HardwareConfiguration;
%nodefaultctor HardwareConfiguration;
class HardwareConfiguration {
    public:
    HardwareConfiguration(uint32_t macs = 0, uint32_t cmdStreamVersion = 0, uint32_t type = static_cast<uint32_t>(DeviceType::UNKNOWN), bool customDma = false);

    %feature("docstring",
    "
    DeviceType enumeration
    ") DeviceType;
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

%extend HardwareConfiguration {
    std::string __str__() const {
        std::ostringstream out;
        out << "{macsPerClockCycle=" << $self->macsPerClockCycle <<
        ", cmdStreamVersion=" << $self->cmdStreamVersion <<
        ", type=" << $self->type <<
        ", customDma=" <<  ($self->customDma? "True": "False") << "}";
        return out.str();
    }
}

%feature("docstring",
"
Device capabilities object which specifies capabilities based on hardware ID, configuration and semantic version.
") Capabilities;
class Capabilities {
    public:
    Capabilities() {}
    Capabilities(const HardwareId& hwId, const HardwareConfiguration& hwCfg, const SemanticVersion& driverVersion);

    HardwareId hwId;
    HardwareConfiguration hwCfg;
    SemanticVersion driver;
};

%extend Capabilities {
    std::string __str__() const {
        std::ostringstream out;
        out << "{hwId=" << EthosU_HardwareId___str__(&$self->hwId) <<
        ", hwCfg=" << EthosU_HardwareConfiguration___str__(&$self->hwCfg) <<
        ", driver=" <<  EthosU_SemanticVersion___str__(&$self->driver) << "}";
        return out.str();
    }
}

%feature("docstring",
"
Device object represents Ethos-U device file descriptor and manages Ethos-U device lifecycle.
Constructor accepts device name and opens file descriptor with O_RDWR | O_NONBLOCK flags.
When the object is destroyed - device file descriptor is closed.
") Device;
%nodefaultctor Device;
class Device {
public:
    Device(const char *device);

    %feature("docstring",
    "
    Performs the I/O control operation on the Ethos-U device.

    Args:
        cmd: Command code
        data: Command data
    Returns:
        int: Return value depends on command. Usually -1 indicates error.
    ") ioctl;
    int ioctl(unsigned long cmd, void *data = nullptr) const;

    %feature("docstring",
    "
    Returns the capabilities of the Ethos-U device.

    Returns:
    Capabilities: Return capabilities of device.
    ") capabilities;
    Capabilities capabilities() const;

    %feature("docstring",
    "
    Returns kernel driver version information.

    Returns:
        SemanticVersion: kernel driver version.
    ") getDriverVersion;
    const SemanticVersion &getDriverVersion() const;
};

%extend Device {

    %feature("docstring",
    "
    Sends ping command to the Ethos-U device.

    See ETHOSU_IOCTL_PING from kernel module uapi/ethosu.h
    ") ping;
    void ping() {
        $self->ioctl(ETHOSU_IOCTL_PING);
    }
}

%feature("docstring",
    "
    Buffer object represents a RW mapping in the virtual address space of the caller.

    Created mapping is shareable, updates to the mapping are visible to other processes mapping the same region.
    Issues ETHOSU_IOCTL_BUFFER_CREATE I/O request to the device with given size.

    Buffer could be created for a device with given size or instantiated directly from
    a file containing binary data.

    Examples:
        >>> import ethosu_driver as driver
        >>> # from file:
        >>> buf = driver.Buffer(device, '/path/to/file')
        >>> # Empty, with size:
        >>> buf = driver.Buffer(device, 1024)
    ") Buffer;
%nodefaultctor Buffer;
class Buffer {
public:
    Buffer(const Device &device, const size_t size);

    %feature("docstring",
    "
    Sets the size of the device buffer to 0.
    ") clear;
    void clear() const;

    %feature("docstring",
    "
    Returns a readonly view to the mapped memory.

    Returns:
        memoryview: readonly memory data.
    ") data;
    %driver_buffer_out;
    char* data() const;
    %clear_driver_buffer_out;

    %feature("docstring",
    "
    Queries device and returns buffer data size.

    Issues ETHOSU_IOCTL_BUFFER_GET I/O request.

    Returns:
        int: current device buffer size.
    ") size;
    size_t size() const;

    %feature("docstring",
    "
    Returns buffer file descriptor id.

    Returns:
        int: file descriptor id.
    ") getFd;
    int getFd() const;
};

%extend Buffer {

    Buffer(const Device& device, const std::string& filename) {
        std::ifstream stream(filename, std::ios::binary);
        if (!stream.is_open()) {
            throw EthosU::Exception(std::string("Failed to open file: ").append(filename).c_str());
        }

        stream.seekg(0, std::ios_base::end);
        size_t size = stream.tellg();
        stream.seekg(0, std::ios_base::beg);

        auto buffer = new EthosU::Buffer(device, size);
        stream.read(buffer->data(), size);

        return buffer;
    }

    %feature("docstring",
    "
    Fills the buffer from python buffer.

    Copies python buffer data to the mapped memory region.

    Args:
        buffer: data to be copied to the mapped memory.

    ") from_buffer;
    %buffer_in(char* buffer, size_t size, BUFFER_FLAG_RW);
    void from_buffer(char* buffer, size_t size) {
        char* data = $self->data();
        std::memcpy(data, buffer, size);
    }
    %clear_buffer_in(char* buffer, size_t size);
}

%feature("docstring",
    "
    Represents the neural network file descriptor received from the Ethos-U device.

    `Network` is created providing `Device` object and a `Buffer` containing tflite file data.
    Network creation issues ETHOSU_IOCTL_NETWORK_CREATE I/O request with buffer file descriptor id.
    Provided `Buffer` data is parsed into tflite Model object and input/output feature maps sizes are saved.

    Destruction of the object closes network file descriptor.
    ") Network;
%nodefaultctor Network;
class Network {
public:

    %feature("docstring",
    "
    Performs the I/O control operation with network buffer device.

    Args:
        cmd: Command code
        data: Command data
    Returns:
        int: Return value depends on command. Usually -1 indicates error.
    ") ioctl;
    int ioctl(unsigned long cmd, void *data);

    %feature("docstring",
    "
    Returns saved sizes of the neural network model input feature maps.

    Returns:
        list: sizes of all input feature maps
    ") getIfmDims;
    const std::vector<size_t> &getIfmDims() const;

    %feature("docstring",
    "
    Returns total size of all input feature maps.

    Returns:
        int: total size of all input feature maps
    ") getIfmSize;
    size_t getIfmSize() const;

    %feature("docstring",
    "
    Returns saved sizes of the neural network model output feature maps.

    Returns:
        list: sizes of all output feature maps
    ") getOfmDims;
    const std::vector<size_t> &getOfmDims() const;

    %feature("docstring",
    "
    Returns total size of all output feature maps.

    Returns:
        int: total size of all output feature maps
    ") getOfmSize;
    size_t getOfmSize() const;
};

%extend Network {

    Network(const Device &device, const std::string& filename)
    {
        std::ifstream stream(filename, std::ios::binary);
        if (!stream.is_open()) {
            throw EthosU::Exception(std::string("Failed to open file: ").append(filename).c_str());
        }

        stream.seekg(0, std::ios_base::end);
        size_t size = stream.tellg();
        stream.seekg(0, std::ios_base::beg);

        std::unique_ptr<unsigned char[]> buffer = std::make_unique<unsigned char[]>(size);
        stream.read(reinterpret_cast<char*>(buffer.get()), size);
        return new EthosU::Network(device, buffer.get(), size);
    }

    %buffer_in(const unsigned char* networkData, size_t networkSize, BUFFER_FLAG_RO);
    Network(const Device &device, const unsigned char* networkData, size_t networkSize)
    {
        if(networkData == nullptr){
            throw EthosU::Exception(std::string("Failed to create the network, networkData is nullptr.").c_str());
        }

        if(networkSize == 0U){
            throw EthosU::Exception(std::string("Failed to create the network, networkSize is zero.").c_str());
        }

        return new EthosU::Network(device, networkData, networkSize);
    }
    %clear_buffer_in(const unsigned char* networkData, size_t networkSize);

    Network(const Device &device, const unsigned int index)
    {
        return new EthosU::Network(device, index);
    }
}

%feature("docstring",
    "
    InferenceStatus enumeration
    ") InferenceStatus;
enum class InferenceStatus {
    OK,
    ERROR,
    RUNNING,
    REJECTED,
    ABORTED,
    ABORTING
    };

%feature("docstring",
    "
    Represents the inference file descriptor received from the Ethos-U device.

    `Inference` is created providing `Network` object and lists of input and output feature maps buffers.
    Feature map buffers are copied.

    Inference creation issues ETHOSU_IOCTL_INFERENCE_CREATE I/O request with
    file descriptor ids for all input and output buffers.

    The number of input/output buffers must not exceed ETHOSU_FD_MAX value defined in the kernel module
    uapi/ethosu.h.

    Destruction of the object closes inference file descriptor.
    ") Inference;
%nodefaultctor Inference;
class Inference {
public:

    %feature("docstring",
    "
    Polls inference file descriptor for events.

    Args:
        timeoutNanos (int64_t): polling timeout in nanoseconds.

    Returns:
        bool: True for success, False otherwise.
    ") wait;
    void wait(int64_t timeoutNanos = -1) const;

    %feature("docstring",
    "
    Aborts the current inference job.

    Returns:
        bool: True if gracefully stopped, False otherwise.
    ") cancel;
    bool cancel() const;

    %feature("docstring",
    "
    Gets the current inference job status.

    Returns:
        InferenceStatus.
    ") status;
    EthosU::InferenceStatus status() const;

    %feature("docstring",
    "
    Returns inference file descriptor.

    Returns:
        int: file descriptor id
    ") getFd;
    int getFd() const;

    %feature("docstring",
    "
    Returns associated `Network` object.

    Returns:
        `Network`: network used during initialisation
    ") getNetwork;
    std::shared_ptr<Network> getNetwork() const;

    %feature("docstring",
    "
    Returns copied input feature maps buffers.

    Returns:
       list: input feature map buffers
    ") getIfmBuffers;
    std::vector<std::shared_ptr<Buffer>> &getIfmBuffers();

    %feature("docstring",
    "
    Returns copied output feature maps buffers.

    Returns:
       list: output feature map buffers
    ") getOfmBuffers;
    std::vector<std::shared_ptr<Buffer>> &getOfmBuffers();

    %feature("docstring",
    "
    Returns PMU event data.

    Returns:
        list: PMU event data
    ") getPmuCounters;
    const std::vector<uint64_t> getPmuCounters();

    %feature("docstring",
    "
    Returns the total cycle count, including idle cycles, as reported by the PMU.

    Returns:
        int: total cycle count
    ") getCycleCounter;
    uint64_t getCycleCounter();

    %feature("docstring",
    "
    Returns maximum supported number of PMU events.

    Returns:
        int: PMU event max
    ") getMaxPmuEventCounters;
    static uint32_t getMaxPmuEventCounters();
};

%extend Inference {
    Inference(const std::shared_ptr<Network> &network,
            const std::vector<std::shared_ptr<Buffer>> &ifm,
            const std::vector<std::shared_ptr<Buffer>> &ofm)
   {
        return new EthosU::Inference(network, ifm.begin(), ifm.end(), ofm.begin(), ofm.end());
   }
   Inference(const std::shared_ptr<Network> & network,
             const std::vector<std::shared_ptr<Buffer>> &ifm,
             const std::vector<std::shared_ptr<Buffer>> &ofm,
             const std::vector<unsigned int> &enabledCounters,
             bool enableCycleCounter)
   {
       return new EthosU::Inference(network, ifm.begin(), ifm.end(), ofm.begin(), ofm.end(), enabledCounters, enableCycleCounter);
   }
}

}
// Clear exception typemap.
%exception;

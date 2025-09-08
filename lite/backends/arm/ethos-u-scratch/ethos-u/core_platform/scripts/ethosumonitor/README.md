# About

The Arm(R) Ethos(TM)-U Monitor is a Python application that reads performance
data from an Ethos-U (sub)system and outputs the captured data to a file on the
host machine in either binary or JSON format.

# Ethos-U Monitor porting guidelines

## New input source

The Ethos-U Monitor currently supports three input sources: DAPLink, Linux
shared memory and file. Adding support for additional input sources requires the
user to modify [inputs.py](inputs.py) and implement the `InputRingBuffer`
interface. The `InputDAPLink` class can be used for reference.

```python
class InputMyCustomInputSource(InputRingBuffer):
    def __init__(self, elfFile):
        # Call base class constructor
        super().__init__(elfFile)

    def read(self, address, size):
        # Read 'size' bytes from 'address' and return a bytearray
```

## Linux shared memory

The memory input source (please see `ethosu_monitor.py memory --help`) is
recommended for Linux Driver Stack, where a Linux host communicates with an
Ethos-U subsystem using shared memory. Please note that the EventRecorder ring
buffer must be stored in shared memory that can be accessed by both Linux and
the Ethos-U subsystem.

The `InputMem` class uses the `/dev/mem` device to read physical memory. The
user must provide a configuration file that describes the memory mapping between
Linux and the subsystem. Please see [sample.json](sample.json) for reference.

If for example the Cortex(R)-M in the Ethos-U subsystem access 4MB of shared
memory at address `0x6000'0000`, and Linux access the same memory region at
physical address `0x8'8000'0000`, then the entry in the configuration file would
look like this:

```json
{
    "memoryMap": [
        {
            "host": "0x880000000",
            "device": "0x60000000",
            "size": "0x00400000"
        }
    ]
}
```

# Firmware

The [baremetal](../../applications/baremetal) application demonstrates how to
periodically poll the Ethos-U PMU device while an inference is running. PMU
samples are written to a ring buffer in memory using the
[EventRecorder](https://github.com/ARM-software/CMSIS-View) library. For users
that develop their own firmware this application and the libraries it depends on
serves as reference.

The ring buffer holding the EventRecorder data is recommended to be allocated
large enough to store all samples for an inference. This is to prevent buffer
overflows and loss of profiling data.

The ring buffer size is configured by `EventRecorderConf.h`. Please refer to
[targets/corstone-300/CMakeLists.txt](../../targets/corstone-300/CMakeLists.txt)
and
[targets/corstone-300/event_recorder/EventRecorderConf.h](../../targets/corstone-300/event_recorder/EventRecorderConf.h)
for an example how to override the default EventRecorder settings.

For Linux systems that use `/dev/mem` to read profiling data, the EventRecorder
BSS segment must be stored in shared memory. For a scatter file it would look
something like this:

```
SHARED_MEMORY 0x60000000 0x20000000
{
    DDR 0x60000000
    {
        EventRecorder.o (+ZI)
    }
}
```
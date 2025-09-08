# About Python ethosu_driver

Python ethosu_driver is an extension for
[Arm Ethos-U driver library](https://review.mlplatform.org/plugins/gitiles/ml/ethos-u/ethos-u-linux-driver-stack/).
Python ethosu_driver provides interface similar to Ethos-u Linux driver C++
Api.

The Python package is built with public headers from the
[driver_library/include](https://review.mlplatform.org/plugins/gitiles/ml/ethos-u/ethos-u-linux-driver-stack/+/refs/heads/master/driver_library/include/)
folder.

The [SWIG](http://www.swig.org/) tool is used to generate the Ethos-U driver
library Python shadow classes and C wrapper.

## Python ethosu_driver library installation

Python ethosu_driver library can be packaged as a source package or a binary
package (wheel). Binary package is platform dependent, the name of the
package will indicate the platform it was built for, e.g.:

* Linux Aarch 64 bit machine: ethosu_driver-1.0.0-cp37-cp37m-linux_aarch64.whl

The source package is platform independent but installation will involve
Ethos-U driver library C wrapper compilation on a target machine.
You will need to have g++ compatible with C++ 14 standard and a Python
development library installed on the target machine.

Python driver binary package is linked statically with C++ Ethos-U driver
library and can operate independently from it when built.
Python driver source package requires static Ethos-U driver library -
libethosu.a - and public header during installation, thus they must be
present on the target machine.

### Installing from wheel

Install ethosu_driver from binary by pointing to the wheel file:

1) If corresponding wheel is available for your platform architecture in the
public repository.
    ```
    pip install ethosu_driver
    ```
2) If you have local wheel file. Example:
    ```
    pip install /path/to/ethosu_driver-X.X.X-cp37-cp37m-linux_aarch64.whl
    ```

### Installing from source package

While installing from sources, you can choose Ethos-U driver library to
be used. By default, library will be searched in standard for your system
locations. You can check them by running:
```
gcc --print-search-dirs
```
Headers will be searched in standard include directories for your system.
If Ethos-U driver library has custom location, set environment variables
*ETHOS_U_DRIVER_LIB* and *ETHOS_U_DRIVER_INCLUDE* to point to Ethos-U driver
library (libethosu.a) and header (ethosu.hpp):
```
export  ETHOS_U_DRIVER_LIB=/path/to/lib
export  ETHOS_U_DRIVER_INCLUDE=/path/to/headers
```

Installing from the public repository.
```
pip install ethosu_driver
```
Installing from local file.
```
pip install /path/to/ethosu_driver-X.X.X.tar.gz
```

If ethosu_driver installation script fails to find Ethos-U driver libraries it
will raise an error like this

`RuntimeError: Ethos-U driver library was not found in
('/usr/lib/gcc/aarch64-linux-gnu/8/', <...> ,'/lib/', '/usr/lib/').
Please install driver to one of the standard locations or set correct
ETHOS_U_DRIVER_INCLUDE and ETHOS_U_DRIVER_LIB env variables.`

You can now verify that ethosu_driver library is installed and check
ethosu_driver version using:

```
pip show ethosu_driver
```

## Building Python ethosu_driver library locally

### Install SWIG

We suggest to use SWIG version 3.0.12 or newer. You can check available swig
version for you system here: https://pkgs.org/download/swig.

For example, install the tool with Ubuntu package manager as follows:
```
sudo apt install swig
```
If your system has swig version less than 3.0.12, please, build and install
from sources:

1. Download SWIG:
    ```
    wget https://github.com/swig/swig/archive/refs/tags/v4.0.2.zip
    unzip v4.0.2.zip
    ```
2. Build and install SWIG:
    ```
    cd swig-4.0.2
    ./autogen.sh
    ./configure --prefix=<set your system installation prefix>
    make
    make install
    ```

### Building as part of cmake flow

To build Python ethosu_driver as part of Ethos-U NPU Linux driver stack provide
the following cmake flags:
1) For source distribution
    ```
    -DBUILD_PYTHON_SRC=1
    ```
2) For wheel distribution
    ```
    -DBUILD_PYTHON_WHL=1
    ```
Note: you will need to have a Python instance for your target platform to build
wheel. For example, if you are building for an AArch64 platform, you will need
Python installation for aarch64-linux-gnu tool-chain.

Build result can be found in `<your cmake build dir>/python/dist`.

### Building standalone

Navigate to `driver_libarary/python` and execute:
```
python3 setup.py clean --all
python3 ./swig_generate.py
python3 setup.py sdist
```
Build result can be found in `driver_libarary/python/dist`.

## Python ethosu_driver API overview

### Getting started

After the Python driver library is installed with pip and can be accessed
within your work environment, import it in your script:

```python
import ethosu_driver as driver
```

Create a device. You can ping Ethos-U device with `ping` method:

```python
device = driver.Device("/dev/ethosu0")
device.ping()
```

You can create memory buffer with data from a binary file or Python buffer
object:

```python
# from file:
data_file = "/path/to/data.bin"
buffer = driver.Buffer(device, data_file)

# from numpy:
ifm_zeros = numpy.zeros(ifm_size, dtype=np.uint8)
ifm_buffer = driver.Buffer(device, ifm_size)
ifm_buffer.from_buffer(ifm_zeros.data)
```

To create a network object, provide the model file or a byte array with the
network data and the created device:

```python
# from file:
network = driver.Network(device, "path/to/model.tflite")

# from byte array:
network = driver.Network(device, network_data)
```

Inference object is instantiated with a network object and lists of input
memory buffers and output memory buffers.
For example:

```python
ifms = [ifm_buffer]

ofms = []
for ofm_size in network.getOfmDims():
    ofm_buffer = driver.Buffer(device, ofm_size)
    ofms.append(ofm_buffer)

inference = driver.Inference(network, ifms, ofms)
```

To execute the inference and wait for the callback:

```python
# wait infinitely
inference.wait()

# wait with a timeout in nano seconds:
inference.wait(timeoutNanos=60e9)
```

To read results of the inference, iterate through available outputs and convert
them to numpy array:

```python
for buffer_out in inference.getOfmBuffers():
array = np.frombuffer(buffer_out.data(), dtype=np.uint8)
```

See inline py docs for more info on driver public API.

## Inference runner

Python ethosu_driver library comes with a script `inference_runner` that is
installed to the Python environment `bin` directory and could be invoked
from a command line by the file name:

```cmd
inference_runner <args>
```

Arguments:

* `--device` : Npu device name. Default: ethosu0.
* `--model` : Tflite model file path.
* `--timeout` : inference timeout in seconds, Default: infinite.
* `--inputs` : list of files containing input feature maps.
* `--output_dir` : directory to store inference results, output feature maps.
Default: current directory.
* `--npy` : Use npy input/output. Default: 0.
* `--profile_counters`: profile counters to measure during inference, accepts
four integers chosen from
[HW Supported Ethos-U PMU
Events](https://review.mlplatform.org/plugins/gitiles/ml/ethos-u/ethos-u-core-driver/+/refs/heads/master/include/pmu_ethosu.h).

Example:

```cmd
inference_runner --device ethosu0 --model ./mobilenet_v2_vela.tflite --timeout
60 --npy 1 --inputs ./input1.npy --output_dir ./ofms
```

## Using inference runner with numpy

Python ethosu_driver libarary could be installed with numpy support.
Numpy will be automatically downloaded and installed alongside ethosu_driver.
If your machine does not have access to pypi repositories you might need to
install NumPy in advance by following public instructions:
<https://scipy.org/install.html>, or to have it installed from wheel package
built for your platform.

Now, if you provide inference runner command line parameter `--npy 1` the
script will interpret input files as numpy array exports and will save output
feature maps as numpy arrays.

## Setup development environment

Before, proceeding to the next steps, make sure that:

1. You have Python 3.6+ installed system-side. The package is not compatible
with older Python versions.
2. You have python3.6-dev installed system-side. This contains header files
needed to build ethosu_driver extension module.
3. In case you build Python from sources manually, make sure that the following
libraries are installed and available in you system:
``python3.6-dev build-essential checkinstall libreadline-gplv2-dev
libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev
libbz2-dev``
4. install SWIG,  swig must be version 4.*

## Setup virtual environment

Now you can proceed with setting up workspace:

1. Set environment variables ETHOS_U_DRIVER_LIB (pointing to Ethos-U driver
library) and ETHOS_U_DRIVER_INCLUDE (pointing to Ethos-U driver library
headers)
2. Create development env using script ``init_devenv.sh``

## Generating SWIG wrappers

Before building package or running tests you need to generate SWIG wrappers
based on the interface [files](src/ethosu_driver/swig).
```commandline
python setup.py clean --all
python ./swig_generate.py
```

## Running unit-tests

Tests could be executed only on a system with Ethos-U NPU device.
Pytest is used as unit-test framework, before running the test you need to
install pytest and numpy or include into your rootfs image:

```commandline
pip install pytest
pip install numpy
```

Execute command from the project root dir:
```
pytest -v
```

## Regenerate SWIG stubs inplace

If you want to check that swig wrappers are compiling correctly, you can issue
extension compilation inplace:

1) clean old generated files:

    ```bash
    python setup.py clean --all
    ```

2) Run swig wrapper source generation as described in [Generating SWIG
wrappers](#generating-swig-wrappers)
3) Run `build_ext` command:

    ```bash
    export ETHOS_U_DRIVER_LIB=/path/to/driver/lib
    export ETHOS_U_DRIVER_INCLUDE=/path/to/driver/include
    python setup.py build_ext --inplace
    ```

It will put all generated files under ./src/ethosu_driver/_generated folder.
Command will fail on x86 machine during linkage phase because Ethos-U driver
is built for Arm platform, but compilation stage should pass successfully
(or give you indication of compilation problems).

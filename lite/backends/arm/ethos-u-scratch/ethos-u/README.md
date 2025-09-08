# Arm(R) Ethos(TM)-U

This is the root repository for all Arm(R) Ethos(TM)-U software. It is provided
to help users download required repositories and place them in a tree structure.

## Fetching externals

The externals can be downloaded with a Python script. The default configuration
is stored in `externals.json` which is a human readable JSON file.

```
$ ./fetch_externals.py fetch
```

The default configuration can be overridden with the `-c` argument, for
example.

```
$ ./fetch_externals.py -c 22.11.json fetch
```

## Directory structure

The script will build following core directory structure.

```
Directory
.
+-- core_platform
+-- core_software
|   +-- cmsis
|   +-- cmsis-nn
|   +-- core_driver
|   +-- tflite_micro
+-- linux_driver_stack
+-- vela
```

| Directory | Description |
--- | ---
| [.](https://git.mlplatform.org/ml/ethos-u/ethos-u.git) | This is the root directory for all Arm Ethos-U software. |
| [core_platform](https://git.mlplatform.org/ml/ethos-u/ethos-u-core-platform.git) | This directory contains drivers, target specific files and is provided as an example how core software can be built for target platforms. |
| [core_software](https://git.mlplatform.org/ml/ethos-u/ethos-u-core-software.git) | The software executing on Arm Cortex-M is referred to as _Core Software_. This folder provides a small build system that illustrates how to build the key components for the Arm Ethos-U core software. |
| [cmsis](https://github.com/ARM-software/CMSIS_5) | CMSIS 5 provides generic interfaces to boot and configure the Arm Cortex-M CPUs. |
| [cmsis_6](https://github.com/ARM-software/CMSIS_6) | CMSIS 6 provides generic interfaces to boot and configure the Arm Cortex-M CPUs. |
| [cmsis-nn](https://github.com/ARM-software/CMSIS-NN.git) | CMSIS-NN provides optimized neural network kernels for Arm Cortex-M CPUs. |
| [core_driver](https://git.mlplatform.org/ml/ethos-u/ethos-u-core-driver.git) | The Arm Ethos-U NPU driver. |
| [tflite_micro](https://github.com/tensorflow/tflite-micro) | The TensorFlow Lite microcontroller framework is used to run inferences. |
| [linux_driver_stack](https://git.mlplatform.org/ml/ethos-u/ethos-u-linux-driver-stack.git) | Example driver stack showing how Linux can dispatch inferences to an Arm Ethos-U subsystem. |
| [vela](https://git.mlplatform.org/ml/ethos-u/ethos-u-vela.git) | The Vela optimizer takes a TFLu file as input and replaces operators that are supported by the Arm Ethos-U NPU with custom operators designed to run on the NPU. Operators not supported by the NPU are executed in software. |

# License

The Arm Ethos-U is provided under an Apache-2.0 license. Please see
[LICENSE.txt](LICENSE.txt) for more information.

# Contributions

The Arm Ethos-U project welcomes contributions under the Apache-2.0 license.

Before we can accept your contribution, you need to certify its origin and give
us your permission. For this process we use the Developer Certificate of Origin
(DCO) V1.1 (https://developercertificate.org).

To indicate that you agree to the terms of the DCO, you "sign off" your
contribution by adding a line with your name and e-mail address to every git
commit message. You must use your real name, no pseudonyms or anonymous
contributions are accepted. If there are more than one contributor, everyone
adds their name and e-mail to the commit message.

```
Author: John Doe \<john.doe@example.org\>
Date:   Mon Feb 29 12:12:12 2016 +0000

Title of the commit

Short description of the change.

Signed-off-by: John Doe john.doe@example.org
Signed-off-by: Foo Bar foo.bar@example.org
```

The contributions will be code reviewed by Arm before they can be accepted into
the repository.

# Security

Please see [Security](SECURITY.md).

# Releases

## Release 24.08

The 24.08 release has been tested on Ubuntu 22.04.3 LTS using Tensorflow version 2.17 as reference.
Note that this release is using the new `main` branch, the old `master` branch is deprecated and will be
removed in the future.

## Release 24.05

The 24.05 release has been tested on Ubuntu 22.04.3 LTS using Tensorflow version 2.16.1 as reference.

## Release 24.02

The 24.02 release has been tested on Ubuntu 22.04.3 LTS using Tensorflow version 2.15 as reference.

## Release 23.11

The 23.11 release has been tested on Ubuntu 22.04.3 LTS using Tensorflow version 2.14 as reference, with the
additional patch applied:

* https://github.com/tensorflow/tensorflow/pull/58400

Without this patch there may be output diffs between TFL and TFLM for certain models.

## Release 23.08

The 23.08 release has been tested against Tensorflow version 2.12 as reference, with the additional patches applied:

* https://github.com/tensorflow/tensorflow/pull/58400
* https://github.com/tensorflow/tensorflow/pull/52014

Without these patches there may be output diffs between TFL and TFLM for certain models.

## Release 23.05

The 23.05 release has been tested against Tensorflow version 2.11 as reference, with the additional patches applied:

* https://github.com/tensorflow/tensorflow/pull/58400
* https://github.com/tensorflow/tensorflow/pull/52014

Without these patches there may be output diffs between TFL and TFLM for certain models.

There's a discrepancy in tensorflow/lite/micro/cortex_m_corstone_300/README.md fixed by https://github.com/tensorflow/tflite-micro/pull/1972.

# Known Issues

## TensorFlow Lite for Microcontrollers Out of Memory Error during Runtime
During runtime the TensorFlow Lite for Microcontrollers framework might report
the following fatal error:

```Failed to resize buffer. Requested: X, available: Y, missing: Z```

where X, Y and Z are numbers of bytes and X = Y + Z.

There can be several reasons for running out of memory during an inference but
one cause is that too much memory was allocated to the Ethos-U during the
offline compilation phase of the `.tflite` file using Vela. This can result in
not enough memory being available at runtime for the other software components
e.g. the application, the framework, or the reference kernels. The solution is
to calculate the amount of memory required at runtime by all components and then
update the amount allocated to the Ethos-U by using either the Vela CLI option
`--arena-cache-size` or the `arena_cache_size` attribute in Vela's `.ini`
configuration file. This calculation can be difficult to get right and so a
pragmatic solution would be to start by reducing the amount allocated to the
Ethos-U by the value ‘Z’ (from the error message) rounded up to the nearest
multiple 16 (the default tensor alignment used in Vela). This may be an
iterative process.

# Trademark notice

Arm, Cortex and Ethos are registered trademarks of Arm Limited (or its
subsidiaries) in the US and/or elsewhere.

# SPDX-FileCopyrightText: Copyright 2021-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from ._generated.driver import Device, Inference, Network, Buffer, \
        MAX_SUPPORTED_KERNEL_DRIVER_MAJOR_VERSION, MIN_SUPPORTED_KERNEL_DRIVER_MAJOR_VERSION, \
        DRIVER_LIBRARY_VERSION_MAJOR, DRIVER_LIBRARY_VERSION_MINOR, DRIVER_LIBRARY_VERSION_PATCH, \
        getLibraryVersion

from ._utilities import open_device, load_model, populate_buffers, \
                        allocate_buffers, get_results, InferenceRunner

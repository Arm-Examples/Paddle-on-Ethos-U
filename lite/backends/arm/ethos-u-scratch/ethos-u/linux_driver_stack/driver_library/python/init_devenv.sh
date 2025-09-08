#!/bin/bash
# SPDX-FileCopyrightText: Copyright 2021-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

set -e
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

python3 -m venv devenv
source devenv/bin/activate
python "${SCRIPT_DIR}"/test/testdata/download.py

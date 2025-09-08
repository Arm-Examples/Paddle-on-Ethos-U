#
# SPDX-FileCopyrightText: Copyright 2021-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import os
import pytest


@pytest.fixture(scope="module")
def data_folder_per_test(request):
    """
        This fixture returns path to folder with test resources (one per test module)
    """

    basedir, script = request.fspath.dirname, request.fspath.basename
    return str(os.path.join(basedir, "testdata", os.path.splitext(script)[0]))


@pytest.fixture(scope="module")
def shared_data_folder(request):
    """
        This fixture returns path to folder with shared test resources among all tests
    """

    return str(os.path.join(request.fspath.dirname, "testdata", "shared"))


@pytest.fixture(scope="function")
def tmpdir(tmpdir):
    """
        This fixture returns path to temp folder. Fixture was added for py35 compatibility
    """

    return str(tmpdir)

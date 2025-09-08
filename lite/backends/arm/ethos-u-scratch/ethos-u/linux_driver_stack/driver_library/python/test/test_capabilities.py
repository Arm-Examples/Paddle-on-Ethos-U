#
# SPDX-FileCopyrightText: Copyright 2021-2022, 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
from ethosu_driver._generated.driver import SemanticVersion
from ethosu_driver._generated.driver import HardwareId
from ethosu_driver._generated.driver import HardwareConfiguration
from ethosu_driver._generated.driver import Capabilities


def check_semantic_version(ma, mi, pa, sv):
    assert ma == sv.major
    assert mi == sv.minor
    assert pa == sv.patch


def test_semantic_version():
    sv = SemanticVersion(1, 2, 3)
    assert '{ major=1, minor=2, patch=3 }' == sv.__str__()
    check_semantic_version(1, 2, 3, sv)


def test_hardware_id():
    version = SemanticVersion(1, 2, 3)
    product = SemanticVersion(4, 5, 6)
    architecture = SemanticVersion(7, 8, 9)
    hw_id = HardwareId(1, version, product, architecture)

    assert 1 == hw_id.versionStatus

    check_semantic_version(1, 2, 3, hw_id.version)
    check_semantic_version(4, 5, 6, hw_id.product)
    check_semantic_version(7, 8, 9, hw_id.architecture)

    assert '{versionStatus=1, version={ major=1, minor=2, patch=3 }, product={ major=4, minor=5, patch=6 }, ' \
           'architecture={ major=7, minor=8, patch=9 }}' == hw_id.__str__()


def test_hw_configuration():
    hw_cfg = HardwareConfiguration(128, 1, HardwareConfiguration.DeviceType_SUBSYSTEM, True)

    assert 1 == hw_cfg.cmdStreamVersion
    assert 128 == hw_cfg.macsPerClockCycle
    assert hw_cfg.customDma
    assert HardwareConfiguration.DeviceType_SUBSYSTEM == hw_cfg.type

    assert "{macsPerClockCycle=128, cmdStreamVersion=1, type=subsystem, customDma=True}" == hw_cfg.__str__()


def test_capabilities():
    version = SemanticVersion(100, 200, 300)
    product = SemanticVersion(400, 500, 600)
    architecture = SemanticVersion(700, 800, 900)
    hw_id = HardwareId(1, version, product, architecture)
    hw_cfg = HardwareConfiguration(256, 1000, HardwareConfiguration.DeviceType_SUBSYSTEM, False)
    driver_v = SemanticVersion(10, 20, 30)

    cap = Capabilities(hw_id, hw_cfg, driver_v)

    check_semantic_version(10, 20, 30, cap.driver)

    check_semantic_version(100, 200, 300, cap.hwId.version)
    check_semantic_version(400, 500, 600, cap.hwId.product)
    check_semantic_version(700, 800, 900, cap.hwId.architecture)

    assert 1000 == cap.hwCfg.cmdStreamVersion
    assert 256 == cap.hwCfg.macsPerClockCycle
    assert HardwareConfiguration.DeviceType_SUBSYSTEM == cap.hwCfg.type
    assert not cap.hwCfg.customDma

    assert '{hwId={versionStatus=1, version={ major=100, minor=200, patch=300 }, ' \
           'product={ major=400, minor=500, patch=600 }, ' \
           'architecture={ major=700, minor=800, patch=900 }}, ' \
           'hwCfg={macsPerClockCycle=256, cmdStreamVersion=1000, type=subsystem, customDma=False}, ' \
           'driver={ major=10, minor=20, patch=30 }}' == cap.__str__()

#
# SPDX-FileCopyrightText: Copyright 2021-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import inspect
import pytest
import ethosu_driver._generated.driver as driver_shadow


def get_classes():
    ignored_class_names = ('_SwigNonDynamicMeta', '_object', '_swig_property')
    return list(filter(lambda x: x[0] not in ignored_class_names,
                       inspect.getmembers(driver_shadow, inspect.isclass)))


@pytest.mark.parametrize("class_instance", get_classes(), ids=lambda x: 'class={}'.format(x[0]))
class TestOwnership:

    def test_destructors_exist_per_class(self, class_instance):
        assert getattr(class_instance[1], '__swig_destroy__', None)

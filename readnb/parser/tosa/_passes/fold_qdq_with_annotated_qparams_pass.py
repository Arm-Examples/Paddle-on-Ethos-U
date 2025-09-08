# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import copy
from typing import cast, Dict, Set, Tuple

def get_qparams(node: dict):
    attr_dict = {}
    for attr in node["attrs"]:
        attr_dict[attr["name"]] = attr["val"]

    weight_scale = attr_dict.get("Filter0_scale", [1])
    input_scale = attr_dict.get("Input0_scale", [1])
    output_scale = attr_dict.get("Output0_scale", [1])

    return weight_scale, input_scale, output_scale

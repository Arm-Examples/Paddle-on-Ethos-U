# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

import serializer.tosa_serializer as ts
import torch
from serializer.tosa_serializer import TosaOp

from parser.tosa.operators.node_visitor import(
    NodeVisitor,
    register_node_visitor,
)
from parser.tosa.tosa_mapping import TosaArg

from parser.tosa.tosa_specification import TosaSpecification
import parser.tosa.tosa_quant_utils as tqutils
from parser.tosa.tosa_utils import get_resize_parameters, tosa_shape


@register_node_visitor
class ReluVisitor(NodeVisitor):
    target = "relu"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80.0+BI"),
    ]

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: dict,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
        param_dt_type_dict: dict,
    ) -> None:
        attr = ts.TosaSerializerAttribute()

        clamp_min_fp = 0.0
        clamp_max_fp = 0.0
        clamp_min_qs = 0
        clamp_max_qs = 0

        if inputs[0].dtype == ts.DType.INT8:
            clamp_min_qs = 0
            clamp_max_qs = 127
        else:
            clamp_min_fp = 0
            clamp_max_fp = float("inf")

        attr.ClampAttribute(
            tosa_graph.builder,
            clamp_min_qs,
            clamp_max_qs,
            clamp_min_fp,
            clamp_max_fp,
        )

        tosa_graph.addOperator(TosaOp.Op().CLAMP, [inputs[0].name], [output.name], attr)

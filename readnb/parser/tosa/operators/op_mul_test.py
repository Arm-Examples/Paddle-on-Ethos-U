# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

import serializer.tosa_serializer as ts  # type: ignore
import torch

from parser.tosa.operators.node_visitor import(
    NodeVisitor,
    register_node_visitor,
)

from serializer.tosa_serializer import TosaOp

from parser.tosa.tosa_specification import TosaSpecification
from parser.tosa.tosa_mapping import TosaArg
import parser.tosa.tosa_quant_utils as tqutils
import parser.tosa.tosa_utils as tutils


@register_node_visitor
class MulVisitor_080_BI(NodeVisitor):
    target = "elementwise_mul"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80.0+BI"),
    ]

    def define_node(
        self,
        node: dict,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
        param_dt_type_dict: dict,
    ) -> None:
        assert inputs[0].dtype == inputs[1].dtype == output.dtype == ts.DType.INT8
        # X
        input_A = inputs[0]
        # Y
        input_B = inputs[1]

        attr_dict = {}
        for attr in node["attrs"]:
            attr_dict[attr["name"]] = attr["val"]

        input_A_scale = attr_dict.get("X0_scale", [1])[0]
        input_B_scale = attr_dict.get("Y0_scale", [1])[0]
        output_scale = attr_dict.get("Output0_scale", [1])[0]

        input_A.shape = tutils.tosa_shape(input_A.shape, input_A.dim_order)
        input_B.shape = tutils.tosa_shape(input_B.shape, input_B.dim_order)

        # Rescale inputs to INT32 with zp=0
        input_A_rescaled = tqutils.build_rescale_to_int32(
            tosa_graph,
            input_A,
            0,
            rescale_scale=1.0,
        )
        input_B_rescaled = tqutils.build_rescale_to_int32(
            tosa_graph,
            input_B,
            0,
            rescale_scale=1.0,
        )

        output_shape = tutils.tosa_shape(output.shape, output.dim_order)
        mul_output = tosa_graph.addIntermediate(output_shape, ts.DType.INT32)

        # Do the INT32 Mul
        attr = ts.TosaSerializerAttribute()
        attr.MulAttribute(shift=0)
        tosa_graph.addOperator(
            TosaOp.Op().MUL,
            [
                input_A_rescaled.name,
                input_B_rescaled.name,
            ],
            [mul_output.name],
            attr,
        )
        mul_output_scale = input_A_scale * input_B_scale
        tqutils.insert_rescale_op_to_int8(tosa_graph, mul_output, mul_output_scale, output_scale, output.name)

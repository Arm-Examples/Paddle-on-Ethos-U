# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List
import serializer.tosa_serializer as ts
from parser.tosa.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from parser.tosa.tosa_mapping import TosaArg
from serializer.tosa_serializer import TosaOp
import parser.tosa.tosa_quant_utils as tqutils
import parser.tosa.tosa_utils as tutils

@register_node_visitor
class ScaleVisitor(NodeVisitor):
    target = "scale"

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
        input_name = node['inputs'][0]['arguments'][0]['name']
        output_name = node['outputs'][0]['arguments'][0]['name']
        attr_dict = {}
        for attr in node["attrs"]:
            attr_dict[attr["name"]] = attr["val"]
        if "scale" not in attr_dict:
            raise ValueError("Can not find scale key in attributes")
        scale_factor = attr_dict["scale"]

        # Currently, only support INT8 as inputs and outputs
        assert inputs[0].dtype == ts.DType.INT8, "ScaleVisitor only supports INT8 datatype as inputs"
        assert output.dtype == ts.DType.INT8, "ScaleVisitor only supports INT8 datatype as outputs"


        intermediate_scale = scale_factor

        input_rescaled = tqutils.build_rescale_to_int32(
            tosa_graph,
            inputs[0],
            0,
            rescale_scale=1.0,
        )

        # def addConst(self, shape, dtype, vals, name=None):
        #     return self.currRegion.addConst(shape, dtype, vals, name)
        scale_const = tosa_graph.addConst(
            [1],
            ts.DType.INT32,
            [scale_factor],
        )

        mul_output = tosa_graph.addIntermediate(
            output.shape,
            ts.DType.INT32
        )
        attr = ts.TosaSerializerAttribute()
        attr.MulAttribute(shift=0)
        tosa_graph.addOperator(
            TosaOp.Op().MUL,
            [input_rescaled.name, scale_const.name],
            [mul_output.name],
            attr
        )

        tqutils.insert_rescale_op_to_int8(
            tosa_graph,
            mul_output,
            intermediate_scale,
            1.0,
            output_name
        )

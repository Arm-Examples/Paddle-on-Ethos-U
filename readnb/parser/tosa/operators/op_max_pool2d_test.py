# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import List

import serializer.tosa_serializer as ts
import torch
from parser.tosa.operators.node_visitor import(
    NodeVisitor,
    register_node_visitor,
)
from parser.tosa.tosa_mapping import TosaArg
from parser.tosa.tosa_specification import TosaSpecification


@register_node_visitor
class MaxPool2dVisitor(NodeVisitor):
    target = "max_pool2d"

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

        attr_dict = {}
        for attr in node["attrs"]:
            attr_dict[attr["name"]] = attr["val"]
        
        input_tensor = inputs[0]
       

        kernel_size = attr_dict['ksize']
        stride = attr_dict.get("strides", [1,1])
        # expand pad_size_list
        padding = attr_dict.get("paddings", [0,0]) * 2
        print(f"kernel_size = {kernel_size}")
        print(f"stride_size = {stride}")
        print(f"padding = {padding}")

        accumulator_type = input_tensor.dtype

        if is_quant_node:
            # Accumulator type always is int8 when input tensor is an integer type.
            accumulator_type = ts.DType.INT8

        # Initilize zero point to zero.
        input_zp = 0
        output_zp = 0

        if is_quant_node:
            input_zp = get_quant_arg_upstream(node.all_input_nodes[0]).zp
            output_zp = get_quant_arg_downstream(list(node.users)[0]).zp

        attr = ts.TosaSerializerAttribute()
        attr.PoolAttribute(
            kernel=kernel_size,
            stride=stride,
            pad=padding,
            input_zp=input_zp,
            output_zp=output_zp,
            accum_dtype=accumulator_type,
        )

        tosa_graph.addOperator(
            ts.TosaOp.Op().MAX_POOL2D,
            [input_tensor.name],
            [output.name],
            attr,
        )

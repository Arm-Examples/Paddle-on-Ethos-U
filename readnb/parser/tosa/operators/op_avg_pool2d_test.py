# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import List

import serializer.tosa_serializer as ts
# import torch
# from parser.tosa._passes.fold_qdq_with_annotated_qparams_pass import (
#     get_input_qparams,
#     get_output_qparams,
# )
from parser.tosa.operators.node_visitor import(
    NodeVisitor,
    register_node_visitor,
)
from parser.tosa.tosa_mapping import TosaArg
from parser.tosa.tosa_specification import TosaSpecification


@register_node_visitor
class AvgPool2dVisitor_0_80_BI(NodeVisitor):
    target = "avg_pool2d"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80.0+BI"),
    ]

    def __init__(self, *args):
        super().__init__(*args)

    def _build_generic_avgpool2d(
        self,
        node: dict,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        input_zp: int,
        output_zp: int,
        accumulator_type,
    ) -> None:
        input_tensor = inputs[0]

#        kernel_size_list = inputs[1].special
#        stride_size_list = inputs[2].special
#        try:
#            pad_size_list = inputs[3].special
#        except IndexError:
#            pad_size_list = [0, 0, 0, 0]

        # get pool kernel params
        attr_dict = {}
        for attr in node["attrs"]:
            attr_dict[attr["name"]] = attr["val"]

        stride_size_list = attr_dict.get("strides", [1,1])
        # expand pad_size_list
        pad_size_list = attr_dict.get("paddings", [0,0]) * 2

        is_global_pooling = attr_dict.get("global_pooling")
        if is_global_pooling:
            # global pooling kernel size equals to input [i_h i_w]
            kernel_size_list = [input_tensor.shape[2], input_tensor.shape[3]]
        else:
            # if not, get from k_size
            input_shape = [input_tensor.shape[2], input_tensor.shape[3]]
            kernel_size_list = [input_shape[i]//ksize for i,ksize in enumerate(attr_dict.get("ksize"))]
            stride_size_list = kernel_size_list

        attr = ts.TosaSerializerAttribute()
        attr.PoolAttribute(
            kernel=kernel_size_list,
            stride=stride_size_list,
            pad=pad_size_list,
            input_zp=input_zp,
            output_zp=output_zp,
            accum_dtype=accumulator_type,
        )

        tosa_graph.addOperator(
            ts.TosaOp.Op().AVG_POOL2D,
            [input_tensor.name],
            [output.name],
            attr,
        )

    def define_node(
        self,
        node: dict,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
        param_dt_type_dict: dict,
    ) -> None:
        input_tensor = inputs[0]
        print(f"{input_tensor.dtype}")
        # assert input_tensor.dtype == ts.DType.INT8

        accumulator_type = ts.DType.INT32

#        input_qargs = get_input_qparams(node)
#        input_zp = input_qargs[0].zp
#
#        output_qargs = get_output_qparams(node)
#        output_zp = output_qargs[0].zp

        input_zp = 0
        output_zp = 0

        self._build_generic_avgpool2d(
            node, tosa_graph, inputs, output, input_zp, output_zp, accumulator_type
        )


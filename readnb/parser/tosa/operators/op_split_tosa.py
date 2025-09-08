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
from parser.tosa.tosa_utils import build_reshape, tosa_shape
import parser.tosa.tosa_quant_utils as tqutils
import parser.tosa.tosa_utils as tutils

@register_node_visitor
class SplitVisitor(NodeVisitor):
    # Use to deal with split operator in paddle.
    target = "split"

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: dict,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
        param_dt_type_dict: dict
    ) -> None:
        assert len(inputs) == 1
        assert inputs[0].dtype == output.dtype == ts.DType.INT8

        attr_dict = {}
        for attr in node["attrs"]:
            attr_dict[attr["name"]] = attr["val"]

        dim = attr_dict.get("axis", 0)
        num = attr_dict.get("num")

        # [1,32,64,48]
        input_shape = inputs[0].shape
        # 32
        total_size = input_shape[dim]
        
        section_size = total_size // num

        start = 0
        for i, output in enumerate(node["outputs"][0]['arguments']):
            end = start + section_size
            if i == num - 1:
                end = input_shape[dim]

            attr = ts.TosaSerializerAttribute()
            start_attr = [start if j == dim else 0 for j in range(len(input_shape))]
            size_attr = [end - start if j == dim else input_shape[j] for j in range(len(input_shape))]

            # transpose start & size attr
            start_attr = [start_attr[i] for i in inputs[0].dim_order]
            size_attr = [size_attr[i] for i in inputs[0].dim_order]

            attr.SliceAttribute(start_attr, size_attr)

            # build rescale intermediate
            output = TosaArg(output)
            split_output = tosa_graph.addIntermediate(
                tosa_shape(output.shape, output.dim_order), ts.DType.INT8
            )

            tosa_graph.addOperator(
                TosaOp.Op().SLICE,
                [inputs[0].name],
                [split_output.name],
                attr
            )

            start = end

            # rescale
            input_scales = attr_dict.get("Input0_scale")
            output_scale = attr_dict.get("Output0_scale")[i]

            # Scale output to 32 bit
            rescaled_output, scale_back = tqutils.insert_rescale_ops_to_int32(
                tosa_graph, [split_output], node, input_scales
            )

            # Scale output back to 8 bit
            tqutils.insert_rescale_op_to_int8(tosa_graph, rescaled_output[0], scale_back, output_scale, output.name)


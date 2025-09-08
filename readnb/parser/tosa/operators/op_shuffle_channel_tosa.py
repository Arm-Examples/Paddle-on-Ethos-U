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

@register_node_visitor
class ShuffleChannelVisitor(NodeVisitor):
    """
    This node visitor targets the torch.cat op.
    Inserts a TOSA CONCAT operator.
    """

    target = "shuffle_channel"

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
        group_size = attr_dict["group"]

        # Reshape with group size
        input = inputs[0]
        input_post_shape = [
            #input.shape[0],
            input.shape[2],
            input.shape[3],
            group_size,
            input.shape[1] // group_size
        ]

        input_reshaped = tosa_graph.addIntermediate(
            input_post_shape,
            input.dtype,
        )
        build_reshape(
            tosa_graph, input.name, input_post_shape, input_reshaped.name
        )

        # Transpose
        output_pre_shape = [
            #input.shape[0],
            input.shape[2],
            input.shape[3],
            input.shape[1] // group_size,
            group_size
        ]

        output_reshaped = tosa_graph.addIntermediate(
            output_pre_shape,
            output.dtype,
        )

        # [N, H, W, g, C/g] -> [N, H, W, C/g, g]
        #perms = [0, 1, 2, 4, 3]
        perms = [0, 1, 3, 2]
        attr = ts.TosaSerializerAttribute()
        attr.TransposeAttribute(perms)
        tosa_graph.addOperator(
            TosaOp.Op().TRANSPOSE, [input_reshaped.name], [output_reshaped.name], attr
        )

        output_shape = [output.shape[i] for i in output.dim_order]
        # Reshape back
        build_reshape(
            tosa_graph, output_reshaped.name, output_shape, output.name
        )


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


@register_node_visitor
class ConcatVisitor(NodeVisitor):
    """
    This node visitor targets the torch.cat op.
    Inserts a TOSA CONCAT operator.
    """

    target = "concat"

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
        dim = attr_dict["axis"]

        input_names = [input['name'] for input in node['inputs'][0]['arguments']]

        attr = ts.TosaSerializerAttribute()
        attr.AxisAttribute(dim)

        tosa_graph.addOperator(
            TosaOp.Op().CONCAT, input_names, [output['name'] for output in node['outputs'][0]['arguments']], attr
        )

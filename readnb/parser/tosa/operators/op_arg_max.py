# Copyright 2024 Arm Limited and/or its affiliates.
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
class ArgMaxVisitor(NodeVisitor):
    target = "arg_max"

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
        axis = 0
        for sub_attr in node['attrs']:
            if sub_attr['name'] == 'axis':
                axis = sub_attr['val']
        
        attr = ts.TosaSerializerAttribute()
        attr.AxisAttribute(axis)

        input_name = node['inputs'][0]['arguments'][0]['name']
        output_name = node['outputs'][0]['arguments'][0]['name']

        tosa_graph.addOperator(
            TosaOp.Op().ARGMAX, [input_name], [output_name], attr
        )

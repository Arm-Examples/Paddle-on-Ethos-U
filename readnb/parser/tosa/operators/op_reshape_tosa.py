# Copyright 2023-2024 Arm Limited and/or its affiliates.
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
class ViewVisitor(NodeVisitor):
    target = "reshape2"

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
    # attr = ts.TosaSerializerAttribute()
    # attr.ReshapeAttribute(new_shape)
    # tosa_fb.addOperator(TosaOp.Op().RESHAPE, [input_name], [output_name], attr)        

        attr = ts.TosaSerializerAttribute()
        input_name = node['inputs'][0]['arguments'][0]['name']
        for node_output in node['outputs']:
            new_shape = node_output['arguments'][0]['type']['dense_tensor']['dt_dims']
            output_name = node_output['arguments'][0]['name']
            attr.ReshapeAttribute(new_shape)
            tosa_graph.addOperator(
                TosaOp.Op().RESHAPE, [input_name], [output_name], attr
            )

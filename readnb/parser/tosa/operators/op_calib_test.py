# Copyright 2023 Arm Limited and/or its affiliates.
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

from parser.tosa.tosa_specification import TosaSpecification
from parser.tosa.tosa_mapping import TosaArg

from parser.tosa.operators.node_visitor import(
    NodeVisitor,
    register_node_visitor,
)
from serializer.tosa_serializer import TosaOp


@register_node_visitor
class CalibVisitor(NodeVisitor):
    target = "calib"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80.0+BI"),
    ]

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
        param_dt_type_dict: dict,
    ) -> None:
        item_name = inputs[0].name
        ## Simply add an identityOp
        tosa_graph.addOperator(TosaOp.Op().IDENTITY, [item_name], [output.name])

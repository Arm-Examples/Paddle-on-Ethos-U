# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import List

import numpy as np

import serializer.tosa_serializer as ts
from serializer.tosa_serializer import TosaOp

from parser.tosa.operators.node_visitor import(
    NodeVisitor,
    register_node_visitor,
)
from parser.tosa.tosa_specification import TosaSpecification
from parser.tosa.tosa_mapping import TosaArg
import parser.tosa.tosa_quant_utils as tqutils
import parser.tosa.tosa_utils as tutils
#from executorch.backends.arm.tosa_mapping import TosaArg
#from executorch.backends.arm.tosa_utils import tosa_shape
from torch.fx import Node


@register_node_visitor
class FullVisitor(NodeVisitor):
    target = "fill_constant"
    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80.0+BI"),
    ]

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
        

        shape = tutils.tosa_shape(output.shape, output.dim_order)
        value = attr_dict.get("value", [1]) 

        if output.dtype == ts.DType.INT8:
            fill_dtype = np.int8
        else:
            fill_dtype = np.float32
        data = np.full(shape, value, dtype=fill_dtype)

        tosa_graph.addConst(shape, output.dtype, data, output.name + "full-const")
        tosa_graph.addOperator(
            ts.TosaOp.Op.IDENTITY, [output.name + "full-const"], [output.name]
        )

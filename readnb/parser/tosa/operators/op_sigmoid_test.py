# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import List

import numpy as np
import torch

import serializer.tosa_serializer as ts
#from executorch.backends.arm.operators.node_visitor import (
#    NodeVisitor,
#    register_node_visitor,
#)
#from executorch.backends.arm.tosa_mapping import TosaArg

#from executorch.backends.arm.tosa_quant_utils import (
#    dequantize_value,
#    get_quant_arg_downstream,
#    get_quant_arg_upstream,
#    QuantArgs,
#    quantize_value,
#)
from serializer.tosa_serializer import TosaOp
#from torch.fx import Node

from parser.tosa.operators.node_visitor import(
    NodeVisitor,
    register_node_visitor,
)
from parser.tosa.tosa_specification import TosaSpecification
from parser.tosa.tosa_mapping import TosaArg
from parser.tosa.tosa_quant_utils import dequantize_value, quantize_value
from parser.tosa.tosa_utils import build_reshape, tosa_shape

from parser.tosa._passes.fold_qdq_with_annotated_qparams_pass import get_qparams

@register_node_visitor
class SigmoidVisitor(NodeVisitor):
    target = "sigmoid"

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
        assert len(inputs) == 1
        assert inputs[0].dtype == output.dtype == ts.DType.INT8

        _, input_scale, output_scale = get_qparams(node)

        table = generate_int8_table_values(input_scale[0], output_scale[0])
        table_attr = ts.TosaSerializerAttribute()
        table_attr.TableAttribute(np.array(table))

        tosa_graph.addOperator(
            TosaOp.Op().TABLE, [inputs[0].name], [output.name], table_attr
        )


def generate_int8_table_values(
    dq_scale,
    q_scale,
) -> torch.Tensor:
    qmin = -128
    qmax = 127

    def f(x: torch.Tensor) -> torch.Tensor:
        x = dequantize_value(x, dq_scale, 0)
        x = torch.sigmoid(x)
        return quantize_value(x, q_scale, 0, qmin, qmax, torch.int8)

    input_dtype = torch.int8
    steps = qmax - qmin + 1
    return f(
        torch.linspace(
            start=qmin,
            end=qmax,
            steps=steps,
            # use torch.int64 to avoid overflow when dequantizing (subtracting zp).
            # e.g. torch.tensor(-50, dtype=torch.int8) - 100 == torch.tensor(106, dtype=torch.int8)
            dtype=torch.int64,
        )
    ).to(dtype=input_dtype)

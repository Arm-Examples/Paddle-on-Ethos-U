# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import List

import serializer.tosa_serializer as ts
import torch
from serializer.tosa_serializer import TosaOp

from tosa.ResizeMode import ResizeMode

from parser.tosa.operators.node_visitor import(
    NodeVisitor,
    register_node_visitor,
)
from parser.tosa.tosa_mapping import TosaArg

from parser.tosa.tosa_specification import TosaSpecification
import parser.tosa.tosa_utils as tutils
import parser.tosa.tosa_quant_utils as tqutils
from parser.tosa.tosa_utils import get_resize_parameters, tosa_shape

@register_node_visitor
class UpsampleBilinear2dVisitor(NodeVisitor):
    target = "bilinear_interp_v2"

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
        assert (
            inputs[0].shape is not None and output.shape is not None
        ), "Only static shapes are supported"

        # tosa_shape output is NHWC, take HW
        input_size_yx = torch.tensor(
            tosa_shape(inputs[0].shape, inputs[0].dim_order)[1:3]
        )
        # Ignore scale and size parameters, directly use the output size as
        # we only support static shapes currently
        output_size_yx = torch.tensor(tosa_shape(output.shape, output.dim_order)[1:3])

        scale_n_yx, scale_d_yx, offset_yx, border_yx = get_resize_parameters(
            input_size_yx, output_size_yx, ResizeMode.BILINEAR, align_corners=True
        )

        def in_int16_range(x):
            return torch.all(x >= -(2**15)) and torch.all(x <= 2**15 - 1)

        assert in_int16_range(scale_n_yx)
        assert in_int16_range(scale_d_yx)
        assert in_int16_range(border_yx)

        attr = ts.TosaSerializerAttribute()
        attr.ResizeAttribute(
            scale=[scale_n_yx[0], scale_d_yx[0], scale_n_yx[1], scale_d_yx[1]],
            offset=offset_yx.tolist(),
            border=border_yx.tolist(),
            #mode=ResizeMode.BILINEAR,
            mode=ResizeMode.NEAREST,
        )

        #FIXME: Rescale int32 output to int8
#        if output.dtype == ts.DType.INT8:
#            resized_shape = tutils.tosa_shape(output.shape, output.dim_order)
#            resize_output = tosa_graph.addIntermediate(resized_shape, ts.DType.INT32)
#            tosa_graph.addOperator(
#                TosaOp.Op().RESIZE, [inputs[0].name], [resize_output.name], attr
#            )
#
#            attr_dict = {}
#            for attr in node["attrs"]:
#                attr_dict[attr["name"]] = attr["val"]
#
#            input_scale = attr_dict.get("Input0_scale")[0]
#            output_scale = attr_dict.get("Output0_scale")[0]
#
#            tqutils.insert_rescale_op_to_int8(tosa_graph, resize_output, 0.000080298, output_scale, output.name)
#        else:
#            tosa_graph.addOperator(
#                TosaOp.Op().RESIZE, [inputs[0].name], [output.name], attr
#            )

        tosa_graph.addOperator(
            TosaOp.Op().RESIZE, [inputs[0].name], [output.name], attr
        )

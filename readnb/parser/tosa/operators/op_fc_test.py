# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import List
import math

import serializer.tosa_serializer as ts
from parser.tosa.tosa_mapping import TosaArg
from parser.tosa.operators.node_visitor import(
    NodeVisitor,
    register_node_visitor,
)

from parser.tosa.tosa_quant_utils import build_rescale_conv_output, build_rescale
from parser.tosa.tosa_utils import build_reshape, tosa_shape, expand_dims

from serializer.tosa_serializer import TosaOp
from parser.tosa.tosa_specification import TosaSpecification


@register_node_visitor
class FCVisitor(NodeVisitor):
    target = "fc"
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
        #input0, input1 = get_two_inputs(node)

        # For atem.mm, the two inputs are of rank 2
        # For TOSA it needs to be rank 3
        # So they need to be reshaped from (H, W) to (1, H, W)
        # NOTE: For now, only INT8 & FP32 is supported


        if len(inputs[1].shape) > 2:
            # when ranks > 2, means the shape of input tensor is from a none squeeze output
            # usually need to add a reshape, but it's easy to just modify the shape only.
            # Without modify others.
            inputs[1].shape = (1, math.prod(inputs[1].shape[1:]))
            inputs[1].dim_order = tuple([0, 1])

        reshape_dtype = ts.DType.INT8
        input0_reshaped = expand_dims(tosa_graph, inputs[0], reshape_dtype, 0)
        input1_reshaped = expand_dims(tosa_graph, inputs[1], reshape_dtype, 0)
        input2_reshaped = expand_dims(tosa_graph, inputs[2], reshape_dtype, 0)
        print(f"FCVisitor input0 = {inputs[0].shape}")
        print(f"FCVisitor input1 = {inputs[1].shape}")
        print(f"FCVisitor input2 = {inputs[2].shape}")
        print(f"FCVisitor input0_reshaped = {input0_reshaped}")
        print(f"FCVisitor input1_reshaped = {input1_reshaped}")
        print(f"FCVisitor input2_reshaped = {input2_reshaped}")

        # The output also needs to be rank 3
        output_new_shape = (1, output.shape[0], output.shape[1])
        print(f"outputshape  = {output.shape}")
        print(f"output_new_shape  = {output_new_shape}")

        # For INT8, we need to get the zero point, otherwise it is 0
        input0_zp, input1_zp = 0, 0

        mat_mul_result = tosa_graph.addIntermediate(
            output_new_shape, ts.DType.INT32
        )
        print(f"mat_mul_result {mat_mul_result}")
        attr = ts.TosaSerializerAttribute()
        attr.MatMulAttribute(A_zp=input0_zp, B_zp=input1_zp)

        tosa_graph.addOperator(
            TosaOp.Op().MATMUL,
            [input1_reshaped.name, input2_reshaped.name],
            [mat_mul_result.name],
            attr,
        )
            
        reshape_intermediate = tosa_graph.addIntermediate(
            output.shape, ts.DType.INT32
        )
        reshape_output_name = reshape_intermediate.name

        # Reshape the final output back to rank 2
        build_reshape(
            tosa_graph, mat_mul_result.name, output.shape, reshape_output_name
        )

        print(f"mat_mul_result {mat_mul_result}")
        print(f"reshape_output_name {reshape_output_name}")

        attr_dict = {}
        for attr in node["attrs"]:
            attr_dict[attr["name"]] = attr["val"]

        weight_scale = attr_dict.get("W0_scale", [1])
        input_scale = attr_dict.get("Input0_scale", [1])
        output_scale = attr_dict.get("Output0_scale", [1])[0]

        print(f"weight_scale {weight_scale}")
        print(f"input_scale {input_scale}")
        print(f"output_scale {output_scale}")


        print(f"reshape_intermediate {reshape_intermediate}")
        final_output_scale =  [(input_scale[0] * w_scale) / output_scale for w_scale in weight_scale]

        # As the input will be INT32, the input_zp must be set to 0
        build_rescale(
            tosa_fb=tosa_graph,
            scale=final_output_scale,
            # pyre-ignore[61]: Uninitialized local [61]: Local variable `reshape_intermediate` is undefined, or not always defined.
            input_node=reshape_intermediate,
            output_name=output.name,
            output_type=ts.DType.INT8,
            output_shape=reshape_intermediate.shape,
            input_zp=0,
            output_zp=0,
            is_double_round=False,
        )

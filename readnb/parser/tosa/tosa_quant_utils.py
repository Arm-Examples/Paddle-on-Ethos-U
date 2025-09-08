# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# Utiliy functions for TOSA quantized lowerings

import math
import numpy as np
from typing import cast, NamedTuple

import serializer.tosa_serializer as ts  # type: ignore
import tosa.Op as TosaOp  # type: ignore
from parser.tosa.tosa_mapping import TosaArg
from serializer.tosa_serializer import TosaSerializerTensor

import torch

def insert_rescale_ops_to_int32(
        tosa_graph: ts.TosaSerializer, inputs: list[TosaArg], node: dict, input_scales: list[float]
) -> tuple[list[TosaSerializerTensor], float]:
    """Rescales all 'nodes' to int32, adding suitable RESCALE ops to 'tosa_graph'.
    The scales are adjusted using the smallest scale of all 'nodes'.

    Returns a list of the rescaled nodes and the scale factor used,
    needed by rescale_node_back_to_int8.

    This functions is used in serialization to TOSA for target ops that are
    handled by the DQ/D folding pass, which stores the quantization parameters
    in the node meta dict.
    """
    tensors = inputs.copy()

    # Reshape tensor according to TOSA dim order
    for tensor in tensors:
        if hasattr(tensor, 'dim_order'):
            dim_order = tensor.dim_order
            tensor.shape = [tensor.shape[i] for i in dim_order]

    min_scale = min(input_scales)
    scales = [scale / min_scale for scale in input_scales]

    rescaled_nodes: list[TosaSerializerTensor] = []
    for tensor, scale in zip(tensors, scales):
        rescaled_nodes.append(
            build_rescale_to_int32(
                tosa_graph,
                tensor,
                0,
                scale,
            )
        )
    return rescaled_nodes, min_scale

def insert_rescale_op_to_int8(
    tosa_graph: ts.TosaSerializer,
    last_tensor: TosaArg,
    in_scale: float,
    out_scale: float,
    output_name
) -> None:
    """Rescales the node back to int8, adding a suitable RESCALE op to 'tosa_graph'.
    Parameters:
        node: The original node that is being handled by the rescales.
        last_tensor:the tosa tensor to rescale back.
        scale: the scaling factor used to rescale to int32, from the function 'insert_rescale_op_to_int32'
        tosa_graph: the tosa_graph to manipulate.

    This functions is used in serialization to TOSA for target ops that are
    handled by the DQ/D folding pass, which stores the quantization parameters
    in the node meta dict.
    """
    output_rescale_scale = in_scale / out_scale

    # Rescale Back to INT8
    build_rescale_from_int32(
        tosa_graph,
        last_tensor.name,
        output_name,
        0,
        output_rescale_scale,
    )


def quantize_value(x, scale, zp, qmin, qmax, qdtype):
    if not isinstance(x, torch.Tensor):
        x = torch.Tensor([x])
    return torch.clip(
        torch.round(x / scale) + zp,
        qmin,
        qmax,
    ).to(qdtype)

def dequantize_value(qx: torch.Tensor, scale, zp) -> torch.Tensor:
    print(qx, zp, scale)
    return (qx - zp) * scale

# Check if scale32 mode is used for given output element type
def is_scale32(type):
    return type == ts.DType.INT8


# TOSA uses the RESCALE operation to scale between values with differing precision.
# The RESCALE operator is defined using an integer multiply, add, and shift.
# This utility function is for calculating the multier and shift given a scale.
# Ref: https://www.mlplatform.org/tosa/tosa_spec.html#_precision_scaling
def compute_multiplier_and_shift(scale, scaleWidth=32):
    if scaleWidth == 16:
        offset = 15
    elif scaleWidth == 32:
        offset = 31
    else:
        raise AssertionError("unsupported scale width")

    assert (isinstance(scale, float) or isinstance(scale, np.float32))

    mantissa, exponent = math.frexp(scale)
    shift = exponent

    const_2_power_15_or_31 = 1 << offset
    shifted_mantissa = round(mantissa * const_2_power_15_or_31)

    assert shifted_mantissa <= const_2_power_15_or_31

    if shifted_mantissa == const_2_power_15_or_31:
        shifted_mantissa = shifted_mantissa / 2
        shift += 1

    # TOSA expects right shift to be positive, and embed (1 << offset) into right shift bits.
    shift = offset - shift

    # INT32_MAX, 2^31 - 1
    assert shifted_mantissa <= (const_2_power_15_or_31 - 1)

    multiplier = shifted_mantissa

    if shift > 62:
        multiplier = multiplier >> min(31, shift - 62)
        shift = 62
    return multiplier, shift


def build_rescale(
    tosa_fb,
    scale,
    input_node,
    output_name,
    output_type,
    output_shape,
    input_zp,
    output_zp,
    is_double_round=False,
):
    scale_width = 32 if is_scale32(output_type) else 16

    #multiplier, shift = compute_multiplier_and_shift(scale, scale_width)
    multiplier = []
    shift = []
    for scale_i in scale:
        m,s = compute_multiplier_and_shift(scale_i, scale_width)
        multiplier.append(m)
        shift.append(s)

    attr_rescale = ts.TosaSerializerAttribute()
    attr_rescale.RescaleAttribute(
        input_zp=input_zp,
        output_zp=output_zp,
        #multiplier=[multiplier],
        #shift=[shift],
        multiplier=multiplier,
        shift=shift,
        scale32=is_scale32(output_type),
        double_round=is_double_round,
        per_channel=True,
        input_unsigned=False,
        output_unsigned=False,
    )

    tosa_fb.addOperator(
        TosaOp.Op().RESCALE, [input_node.name], [output_name], attr_rescale
    )

    return


def build_rescale_to_int32(
    tosa_fb, input, input_zp, rescale_scale, is_scale32=True, is_double_round=False
) -> TosaSerializerTensor:
    multiplier, shift = compute_multiplier_and_shift(rescale_scale)
    attr_rescale = ts.TosaSerializerAttribute()
    attr_rescale.RescaleAttribute(
        input_zp=input_zp,
        output_zp=0,
        multiplier=[multiplier],
        shift=[shift],
        scale32=is_scale32,
        double_round=is_double_round,
        per_channel=False,
        input_unsigned=False,
        output_unsigned=False,
    )
    input_A_rescaled_to_int32 = tosa_fb.addIntermediate(input.shape, ts.DType.INT32)
    tosa_fb.addOperator(
        TosaOp.Op().RESCALE,
        [input.name],
        [input_A_rescaled_to_int32.name],
        attr_rescale,
    )

    return input_A_rescaled_to_int32

def build_rescale_from_int32(
    tosa_fb,
    input_name,
    output_name,
    output_zp,
    rescale_scale,
    is_scale32=True,
    is_double_round=False,
) -> None:
    multiplier, shift = compute_multiplier_and_shift(rescale_scale)
    attr_rescale_output = ts.TosaSerializerAttribute()
    attr_rescale_output.RescaleAttribute(
        input_zp=0,
        output_zp=output_zp,
        multiplier=[multiplier],
        shift=[shift],
        scale32=is_scale32,
        double_round=is_double_round,
        per_channel=False,
        input_unsigned=False,
        output_unsigned=False,
    )

    tosa_fb.addOperator(
        TosaOp.Op().RESCALE, [input_name], [output_name], attr_rescale_output
    )

    return

""" Creates a TOSA rescale op based on conv2d parameters. """


def build_rescale_conv_output(
    tosa_fb,
    op,
    output_name,
    output_type,
    input_scale,
    weight_scale,
    output_scale,
    output_zp,
):
    # TODO add check to verify if this is a Per-channel quantization.
    #post_conv2d_scale = (input_scale * weight_scale) / output_scale
    if isinstance(output_scale, list) and len(set(output_scale)) == 1:
        output_scale = output_scale[0]
    elif isinstance(output_scale, float):
        pass
    else:
        raise ValueError("Found special output_scale values.")
    post_conv2d_scale = [(input_scale * w_scale) / output_scale for w_scale in weight_scale]

    # Since we assume the input tensor that is being rescaled is int32 date type, zero point must be 0.
    build_rescale(
        tosa_fb,
        post_conv2d_scale,
        op,
        output_name,
        output_type,
        op.shape,
        0,
        output_zp,
    )
    return


def build_relu(
    tosa_graph,
    input_node,
    output
):
    attr = ts.TosaSerializerAttribute()

    clamp_min_fp = 0.0
    clamp_max_fp = 0.0
    clamp_min_qs = 0
    clamp_max_qs = 0
    if input_node.dtype == ts.DType.INT8:
        clamp_min_qs = 0
        clamp_max_qs = 127
    else:
        clamp_min_fp = 0
        clamp_max_fp = float("inf")

    attr.ClampAttribute(
        tosa_graph.builder,
        clamp_min_qs,
        clamp_max_qs,
        clamp_min_fp,
        clamp_max_fp,
    )

    tosa_graph.addOperator(TosaOp.Op().CLAMP, [input_node.name], [output.name], attr)

    return


def build_hardswish(
    tosa_graph,
    input_node,
    output,
    input_scale,
    output_scale
):
    table = generate_int8_hardswish_table_values(input_scale, output_scale)
    table_attr = ts.TosaSerializerAttribute()
    table_attr.TableAttribute(np.array(table))

    tosa_graph.addOperator(
        TosaOp.Op().TABLE, [input_node.name], [output.name], table_attr
    )

    return


def generate_int8_hardswish_table_values(
    dq_scale,
    q_scale,
) -> torch.Tensor:
    qmin = -128
    qmax = 127

    def f(x: torch.Tensor) -> torch.Tensor:
        x = dequantize_value(x, dq_scale, 0)
        x = torch.nn.functional.hardswish(x)
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

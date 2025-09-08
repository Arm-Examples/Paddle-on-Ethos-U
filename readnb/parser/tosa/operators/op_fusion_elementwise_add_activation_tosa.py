from typing import List
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
from parser.tosa.tosa_quant_utils import build_relu

@register_node_visitor
class AddVisitor_080_BI(NodeVisitor):
    target = "fusion_elementwise_add_activation"
    # target = "aten.add.Tensor"

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
        # Specification (0.80.0) states that input and output types
        # should all be the same
        assert inputs[0].dtype == inputs[1].dtype == output.dtype
        assert inputs[0].dtype in [ts.DType.INT8, ts.DType.INT32, ts.DType.FP32]

        # only suppoert X,Y Add
        attr_dict = {}
        for attr in node["attrs"]:
            attr_dict[attr["name"]] = attr["val"]

        input_X_scale = attr_dict.get("X0_scale", [1])[0]
        input_Y_scale = attr_dict.get("Y0_scale", [1])[0]
        output_scale = attr_dict.get("Output0_scale", [1])[0]
        input_scales = [input_X_scale, input_Y_scale]

        if inputs[0].dtype == ts.DType.INT8:
            rescaled_inputs, scale_back = tqutils.insert_rescale_ops_to_int32(
                tosa_graph, inputs, node, input_scales
            )
        else:
            # input[0].dtype == ts.DType.INT32
            # Non quantized input, natively support by TOSA.ADD
            rescaled_inputs = inputs

        if output.dtype == ts.DType.INT8:
            broadcasted_shape = tutils.tosa_shape(output.shape, output.dim_order)
            add_output = tosa_graph.addIntermediate(broadcasted_shape, ts.DType.INT32)
        else:
            # output.dtype == ts.DType.INT32
            add_output = output

        # Do the INT32 Add
        tosa_graph.addOperator(
            TosaOp.Op().ADD,
            [
                rescaled_inputs[0].name,
                rescaled_inputs[1].name,
            ],
            [add_output.name],
            None,
        )

        if output.dtype == ts.DType.INT8:
            # Scale output back to 8 bit
            # pyre-ignore
            tqutils.insert_rescale_op_to_int8(tosa_graph, add_output, scale_back, output_scale, output.name)

        # TODO: confirm this
        if attr_dict["act_type"] == "relu":
            build_relu(tosa_graph, add_output, output)
        else:
            raise RuntimeError(
                "Only [Relu] is supported as activation function in fused_conv2d node"
            )



@register_node_visitor
class AddVisitor_080_MI(AddVisitor_080_BI):
    # inheriting 'target' from BI class

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80.0+MI"),
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
        param_dt_type_dict: dict
    ) -> None:
        # Specification (0.80.0) states that input and output types
        # should all be the same
        assert inputs[0].dtype == inputs[1].dtype == output.dtype

        if inputs[0].dtype in [ts.DType.INT8, ts.DType.INT32]:
            # Call the inherited define_node for handling integers
            super().define_node(node, tosa_graph, inputs, output, is_quant_node)
        else:
            # FP32 Add lowering
            assert inputs[0].dtype == ts.DType.FP32
            assert output.dtype == ts.DType.FP32

            # MI lowering
            tosa_graph.addOperator(
                TosaOp.Op().ADD,
                [inputs[0].name, inputs[1].name],
                [output.name],
                None,
            )

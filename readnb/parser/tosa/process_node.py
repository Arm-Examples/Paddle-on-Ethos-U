import serializer.tosa_serializer as ts
import numpy as np
from typing import cast, Dict

from parser.tosa.tosa_mapping import map_dtype, TosaArg, map_dtype
from parser.tosa.tosa_utils import (
    tosa_shape,
    getNodeArgs,
    getNodeOutArgs
)
from parser.nb.nb_graph import (
    OpBlock,
    VarParam
)

from parser.tosa.operators.node_visitor import NodeVisitor
from parser.tosa.tosa_specification import TosaSpecification

# test case add
from parser.tosa.operators.op_add_test import AddVisitor_080_BI

def process_call_function(
    node: dict,
    tosa_graph: ts.TosaSerializer,
    node_visitors: Dict[str, NodeVisitor],
    tosa_spec: TosaSpecification,
    param_dt_type_dict: dict,
):
    # Unpack arguments and convert
    inputs = getNodeArgs(node['inputs'])

    # Convert output (this node itself)
    output = TosaArg(node['outputs'][0]['arguments'][0])

    outputs = getNodeOutArgs(node['outputs'])

    # is_quant_node = is_node_quantized(node)
    # if is_quant_node:
    #     output_dtype = map_dtype(get_quantized_node_output_dtype(node))
    # else:
    #     output_dtype = output.dtype

    #TODO: not support quant node
    is_quant_node = False
    for output in outputs:
        output_dtype = output.dtype
        tosa_graph.currRegion.currBasicBlock.addTensor(
            output.name,
            tosa_shape(output.shape, output.dim_order),
            output_dtype,
        )

    # Visiting each Node
    # pyre-ignore[16]: Undefined attribute.
    if node['type'] in node_visitors:
        # pyre-ignore[16]: Undefined attribute.
        node_visitors[node['type']].define_node(
            node,
            tosa_graph,
            inputs,
            output,
            is_quant_node,
            param_dt_type_dict
        )
    else:
        raise RuntimeError(f"Unknown operator {node['type']} for TOSA : {tosa_spec}")


def process_inputs(
    node: dict,
    tosa_graph: ts.TosaSerializer,
    # tosa_spec: TosaSpecification,
):
    """Serialize an input node"""
    # inputs need to be in default dim_order (contiguous memory format)
    # # meta = node.meta["val"]
    # if meta.dim_order() != tuple(range(meta.dim())):
    #     raise RuntimeError(
    #         f"Arm backend only supports contiguous memory format for inputs. "
    #         f"Expected dim_order: {tuple(range(meta.dim()))}, but got: {meta.dim_order()} for node {node.name}"
    #     )
    print(node)
    inputs = [TosaArg(node)]
    input_shape = inputs[0].shape
    input_dim_order = inputs[0].dim_order
    # tensor = ts.TosaSerializerTensor(
    #     inputs[0].name,
    #     tosa_shape(input_shape, input_dim_order),
    #     (
    #         map_dtype(get_quantized_node_output_dtype(node))
    #         if is_node_quantized(node)
    #         else inputs[0].dtype
    #     ),
    #     data=None,
    #     placeholderFilename=inputs[0].name + ".npy",
    # )
    tensor = ts.TosaSerializerTensor(
        inputs[0].name,
        tosa_shape(input_shape, input_dim_order),
        inputs[0].dtype,
        data=None,
        placeholderFilename=inputs[0].name + ".npy",
    )
    tosa_graph.addInputTensor(tensor)


# def process_quantized_bias(
#     node: torch.fx.Node,
#     tosa_graph: ts.TosaSerializer,
#     parameter_values,
# ):
#     """
#     Serialize bias node that needs to be quantized.
#     """
#     consumer_node = list(node.users)[0]
#     (
#         input_node,
#         weight_node,
#         _,
#     ) = consumer_node.all_input_nodes

#     input_qargs = get_input_qparams(consumer_node)

#     input_node_scale = input_qargs[0].scale
#     weight_node_scale = input_qargs[1].scale
#     bias_values_quantized = (
#         (parameter_values / (input_node_scale * weight_node_scale))
#         .round()
#         .astype(np.int32)
#     )

#     tosa_graph.addConst(
#         bias_values_quantized.shape,
#         ts.DType.INT32,
#         bias_values_quantized,
#         name=node.name,
#     )


# def process_inputs_to_parameters(
#     node: torch.fx.Node,
#     tosa_graph: ts.TosaSerializer,
#     edge_program: ExportedProgram,
#     tosa_spec: TosaSpecification,
# ):
#     """Serialize bias and non-quantized weights"""
#     inputs = [TosaArg(node)]
#     parameter_name = edge_program.graph_signature.inputs_to_parameters[node.name]
#     parameter_data = edge_program.state_dict[parameter_name]

#     assert isinstance(parameter_data, torch.Tensor), "Expect Attr to be tensor"
#     parameter_values = parameter_data.detach().numpy()

#     if is_bias_node_for_quantized_conv(node):
#         # BI bias
#         assert tosa_spec.support_integer(), f"{tosa_spec} doesnt't support integer"
#         process_quantized_bias(node, tosa_graph, parameter_values)
#     else:
#         # MI weights or bias
#         if inputs[0].dtype == torch.float32:
#             assert tosa_spec.support_float(), f"{tosa_spec} doesn't support float"

#         parameter_values = np.transpose(parameter_values, inputs[0].dim_order)

#         tosa_graph.addConst(
#             parameter_values.shape, inputs[0].dtype, parameter_values, name=node.name
#         )


def process_inputs_to_buffers(
    node: dict,
    tosa_graph: ts.TosaSerializer,
    tensor: VarParam,
    # edge_program: ExportedProgram,
):
    """Serialize quantized weights"""
    inputs = [TosaArg(node)]
    buffer_name = tensor['name']
    buffer_data_type = tensor['data_type']
    # buffer_data =
    # buffer_data = param
    buffer_values = tensor['data']
    buffer_dims = tensor['dims']
    buffer_values = np.reshape(buffer_values, buffer_dims)
    type(buffer_values)
    # assert isinstance(buffer_data, torch.Tensor), "Expect Attr to be tensor"
    # buffer_values = buffer_data.detach().numpy()

    # TODO: fragile code for temporary fix
    # the mean and var tensors are also stored here but they have shape (1, )
    # we only transpose weights here
    buffer_values = np.transpose(buffer_values, inputs[0].dim_order)

    if inputs[0].dtype != map_dtype(buffer_data_type):
        inputs[0].dtype = map_dtype(buffer_data_type)

    tosa_graph.addConst(
        buffer_values.shape, inputs[0].dtype, buffer_values, name=buffer_name
    )


# def process_inputs_to_lifted_tensor_constants(
#     node: torch.fx.Node,
#     tosa_graph: ts.TosaSerializer,
#     edge_program: ExportedProgram,
# ):
#     arg = TosaArg(node)
#     tensor_name = edge_program.graph_signature.inputs_to_lifted_tensor_constants[
#         arg.name
#     ]
#     tensor = edge_program.tensor_constants[tensor_name]
#     tensor_data = tensor.detach().numpy()

#     tosa_graph.addConst(tensor_data.shape, arg.dtype, tensor_data, name=arg.name)

def process_placeholder(
    node: dict,
    tosa_graph: ts.TosaSerializer,
    params: VarParam,
):
    # if node['name'] in edge_program.graph_signature.user_inputs:
    #     process_inputs(node, tosa_graph, tosa_spec)
    #     raise RuntimeError(f"Placeholder '{node.name}' is of unknown type.")
    node_name = node['name']
    match_tensor = [p.tensor for p in params if p.tensor['name'] == node_name]
    if match_tensor != None and len(match_tensor) == 1:
    # if any(node_name in p.tensor['name'] for p in params):
        process_inputs_to_buffers(node, tosa_graph, match_tensor[0])
    else:
        print(f"{node_name} is a normal tensor which use as simple output.")
        # raise RuntimeError(f"Placeholder '{node_name}' is of not suit type. tensor {match_tensor} {len(match_tensor)}")
    
# def process_placeholder(
#     node: torch.fx.Node,
#     tosa_graph: ts.TosaSerializer,
#     edge_program: ExportedProgram,
#     tosa_spec: TosaSpecification,
# ):
#     """Wrapper for processing and serializing all types of placeholders"""
#     assert node.name == node.target, "Expect placeholder name and target to match"
#     assert 0 == len(node.args), "Can't handle default input values"

#     if node.name in edge_program.graph_signature.user_inputs:
#         process_inputs(node, tosa_graph, tosa_spec)
#     elif node.name in edge_program.graph_signature.inputs_to_parameters:
#         process_inputs_to_parameters(node, tosa_graph, edge_program, tosa_spec)
#     elif node.name in edge_program.graph_signature.inputs_to_buffers:
#         process_inputs_to_buffers(node, tosa_graph, edge_program)
#     elif node.name in edge_program.graph_signature.inputs_to_lifted_tensor_constants:
#         process_inputs_to_lifted_tensor_constants(node, tosa_graph, edge_program)
#     elif node.name in edge_program.graph_signature.inputs_to_lifted_custom_objs:
#         raise NotImplementedError(
#             "Placeholder is of type 'lifted custom object' which is not supported."
#         )
#     else:
#         raise RuntimeError(f"Placeholder '{node.name}' is of unknown type.")


# def process_output(
#     node: dict,
#     tosa_graph: ts.TosaSerializer,
# ):
#     for output in cast(tuple[torch.fx.Node, ...], node.args[0]):
#         tosa_graph.addOutputTensor(
#             tosa_graph.currRegion.currBasicBlock.tensors[output.name]
#         )

def process_output(
    node: dict,
    tosa_graph: ts.TosaSerializer,
    # tosa_spec: TosaSpecification,
):
    """Serialize an input node"""
    print(node)
    if None is not tosa_graph.currRegion.currBasicBlock.tensors[node['name']]:
        tosa_graph.addOutputTensor(
            tosa_graph.currRegion.currBasicBlock.tensors[node['name']]
        )
    else:
        outputs = [TosaArg(node)]
        input_shape = outputs[0].shape
        input_dim_order = outputs[0].dim_order
        tensor = ts.TosaSerializerTensor(
            outputs[0].name,
            tosa_shape(input_shape, input_dim_order),
            outputs[0].dtype,
            data=None,
            placeholderFilename=outputs[0].name + ".npy",
        )
        tosa_graph.addOutputTensor(tensor)

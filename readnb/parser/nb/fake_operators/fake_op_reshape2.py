import copy
import numpy as np
from typing import Dict, List

# from parser.nb.fake_operators.fake_node_visitor import(
#     FakeNodesVisitor,
#     register_fake_nodes,
# )
from .fake_node_visitor import (
    FakeNodesVisitor,
    register_fake_nodes,
)

@register_fake_nodes
class FakeReshapeVisitor(FakeNodesVisitor):

    target = "reshape2"

    def __init__(self):
        super().__init__()

    def shape_infer(self, **kwargs):
        input_shape = kwargs.get('input_shape')
        shape = kwargs.get('shape')
        input_shape_order = kwargs.get('input_shape_order')

        if input_shape and shape:
            total_input_element = np.prod(input_shape)
            non_minus_one_shape = [i for i in shape if i != -1]
            total_non_minus_one_element = np.prod(non_minus_one_shape)
            output_shape = [int(total_input_element // total_non_minus_one_element) if i == -1 else i for i in shape]
            # TODO: Optimize here, is there any better method?
            print (f"Input shape: {input_shape}, Shape: {shape}, Inputshape order {input_shape_order}, Output shape: {output_shape}")
            if len(output_shape) < len(input_shape_order):
                reduce_dim = len(input_shape_order) - len(output_shape)
                output_layout = [i for i in input_shape_order if i < len(input_shape_order) - reduce_dim]
            elif len(output_shape) > len(input_shape_order):
                output_layout = input_shape_order + [i for i in range(len(input_shape_order), len(output_shape))]
                print(f"Warning: Output shape {output_shape} has more dimensions than input shape order {input_shape_order} {output_layout}")
        else:
            raise ValueError(f"Missing necessary parameters {input_shape} {shape}")
        return output_shape, output_layout

    def infer(self, **kwargs):
        op = kwargs.get('op')
        attr_dict = kwargs.get('attrs')

        assert(op["inputs"][0]['arguments'] != None)
        assert(len(op["inputs"][0]['arguments']) != 0)

        input_info = op["inputs"][0]['arguments'][0]
        output_infos = op["outputs"]
        #FIXME: This is for now to only for op scale which have a None use input "ScaleTensor" wo arguments.
        input_dtype = input_info["type"]["dense_tensor"]["dt_type"]
        input_shape = input_info["type"]["dense_tensor"]["dt_dims"]
        # input_shape_order = input_info["type"]["dense_tensor"]["dt_dim_order"]
        if "dt_dim_order" not in input_info["type"]["dense_tensor"].keys():
            input_shape_order = list(range(len(input_shape)))
            print(f"Warning: input {op['id']} has no dt_dim_order, use default order {input_shape_order}")
        else:
            input_shape_order = input_info["type"]["dense_tensor"]["dt_dim_order"]

        for output_info in output_infos:
            output_info_argus = output_info['arguments'][0]
            output_info_argus["type"]["dense_tensor"]["dt_type"] = input_dtype
            # E.g. output_shape [1, 17, 32, 24] -> [-1, 17, 768] then output_order [0, 2, 3, 1] -> [0, 2, 1]
            output_shape, output_order = self.shape_infer(input_shape=input_shape, shape=attr_dict["shape"], input_shape_order=input_shape_order)
            if not isinstance(output_shape, list) or not isinstance(output_order, list):
                print("Warning output shape type: ", type(output_shape), " output order type: ", type(output_order))
            output_info_argus["type"]["dense_tensor"]["dt_dims"] = list(output_shape)
            output_info_argus["type"]["dense_tensor"]["dt_dim_order"] = copy.deepcopy(output_order)

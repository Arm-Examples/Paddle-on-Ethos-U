import copy
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
class FakeConcatVisitor(FakeNodesVisitor):

    target = "concat"

    def __init__(self):
        super().__init__()

    def shape_infer(self, **kwargs):
        input_shape_list = kwargs.get('input_shape_list')
        axis = kwargs.get('axis')

        # axis is the dimension to concatenate along
        if input_shape_list and axis is not None:
            sum = 0
            if axis < 0:
                axis += len(input_shape_list[0])
            else:
                if axis >= len(input_shape_list[0]):
                    raise ValueError(f"Axis {axis} is out of bounds for input shape {input_shape_list[0]}")
            for shape in input_shape_list:
                sum += shape[axis]

            output_shape = copy.deepcopy(input_shape_list[0])
            output_shape[axis] = sum
            return output_shape
        else:
            raise ValueError(f"Missing necessary parameters {input_shape_list} {axis}")

    def infer(self, **kwargs):
        op = kwargs.get('op')
        attr_dict = kwargs.get('attrs')
        params = kwargs.get('params')

        # Each inputs only has one element
        inputs_info = op["inputs"][0]['arguments']
        outputs_info = op["outputs"][0]['arguments']
        #FIXME: This is for now to only for op scale which have a None use input "ScaleTensor" wo arguments.
        if inputs_info == None or len(inputs_info) == 0:
            print("Waring the current concat op has no input arguments!!!")
            output_shape = []
        else :
            input_data_type = inputs_info[0]["type"]["dense_tensor"]["dt_type"]
            input_shape_list = []
            input_shape_order_list = []
            for input_info in inputs_info:
                input_shape_list.append(list(input_info["type"]["dense_tensor"]["dt_dims"]))
                input_shape_order_list.append(input_info["type"]["dense_tensor"]["dt_dim_order"])

            # FIXME: input_shape list has error value [(1, 16, 32, 24), [1, 3, 128, 96]]
            # TODO: Add input shape check logic
            output_shape = self.shape_infer(input_shape_list=input_shape_list, axis=attr_dict["axis"])
            output_shape_order = input_shape_order_list[0]
            output_info = outputs_info[0]
            output_info["type"]["dense_tensor"]["dt_type"] = input_data_type
            output_info["type"]["dense_tensor"]["dt_dims"] = output_shape
            output_info["type"]["dense_tensor"]["dt_dim_order"] = output_shape_order

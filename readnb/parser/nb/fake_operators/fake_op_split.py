from typing import Dict, List
import copy

# from parser.nb.fake_operators.fake_node_visitor import(
#     FakeNodesVisitor,
#     register_fake_nodes,
# )
from .fake_node_visitor import (
    FakeNodesVisitor,
    register_fake_nodes,
)

@register_fake_nodes
class FakeSplitVisitor(FakeNodesVisitor):

    target = "split"

    def __init__(self):
        super().__init__()

    def shape_infer(self, **kwargs):
        input_shape_list = kwargs.get('input_shape_list')
        axis = kwargs.get('axis')
        output_num = kwargs.get('output_num')

        # axis is the dimension to concatenate along
        if input_shape_list and axis:
            input_shape = list(input_shape_list[0])
            output_shape_list = []
            for i in range(output_num):
                output_shape = copy.deepcopy(input_shape)
                output_shape[axis] = input_shape[axis] // output_num
                output_shape_list.append(output_shape)
        else:
            raise ValueError(f"Missing necessary parameters {input_shape_list} {axis}")
        return output_shape_list

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
            output_shape_list = []
        else :
            input_dtype = inputs_info[0]["type"]["dense_tensor"]["dt_type"]
            input_shape_list = []
            input_shape_order_list = []
            for input_info in inputs_info:
                input_shape_list.append(list(input_info["type"]["dense_tensor"]["dt_dims"]))
                input_shape_order_list.append(input_info["type"]["dense_tensor"]["dt_dim_order"])

            # FIXME: input_shape list has error value [(1, 16, 32, 24), [1, 3, 128, 96]]
            output_num = len(outputs_info)
            output_shape_list = self.shape_infer(input_shape_list=input_shape_list,
                                            axis=attr_dict["axis"],
                                            output_num=output_num)
            output_shape_order_list = copy.deepcopy(input_shape_order_list)
            output_shape_order_list.append(copy.deepcopy(input_shape_order_list[0]))

            for index, output_info in enumerate(outputs_info):
                output_info["type"]["dense_tensor"]["dt_dims"] = output_shape_list[index]
                output_info["type"]["dense_tensor"]["dt_dim_order"] = output_shape_order_list[index]
                output_info["type"]["dense_tensor"]["dt_type"] = input_dtype

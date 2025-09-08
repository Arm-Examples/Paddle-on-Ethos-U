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
class FakeTransposeVisitor(FakeNodesVisitor):

    target = "transpose2"

    def __init__(self):
        super().__init__()

    def shape_infer(self, **kwargs):
        input_shape = kwargs.get('input_shape')
        axis = kwargs.get('axis')

        if input_shape and axis:
            output_shape = [input_shape[i] for i in axis]
        else:
            raise ValueError(f"Missing necessary parameters {input_shape} {axis}")
        return output_shape


    def infer(self, **kwargs):
        op = kwargs.get('op')
        attrs = kwargs.get('attrs')

        for idx, input in enumerate(op["inputs"]):
            #FIXME: This is for now to only for op scale which have a None use input "ScaleTensor" wo arguments.
            if op["inputs"][idx]["arguments"] == None or len(op["inputs"][idx]["arguments"]) == 0:
                continue
            it = op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"]
            in_type = op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_type"]
            input_dim_order = op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dim_order"]

            # op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"] = input_shape #FIXME: TEMP

        output_shape = self.shape_infer(input_shape=it, axis=attrs['axis'])

        # TODO：Make sure dt_dim_order is no need
        for idx, output in enumerate(op['outputs']):
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_type"] = in_type
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"] = output_shape
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dim_order"] = input_dim_order
            # op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"] = (3,3,4,4)
        return output_shape

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
class FakeSqueeze2Visitor(FakeNodesVisitor):

    target = "squeeze2"

    def __init__(self):
        super().__init__()

    def shape_infer(self, **kwargs):
        input_shape = kwargs.get('input_shape')
        input_dim_order = kwargs.get('input_dim_order')
        op_attrs = kwargs.get('attrs')

        axes = op_attrs['axes']
        try:
            for axe in axes:
                if input_shape[axe] != 1:
                    raise ValueError(f'Wrong axes value {axe}')
                else: 
                    output_tensor = (list(input_shape))
                    output_tensor.pop(axe)
        except ValueError as e:
            print(e)
        dim_order = list(input_dim_order)
        dim_order.remove(axe)
        dim_order = [x-1 if x > axe else x for x in dim_order]

        return tuple(output_tensor), dim_order

    def infer(self, **kwargs):
        op = kwargs.get('op')
        attrs = kwargs.get('attrs')
        for idx, input in enumerate(op["inputs"]):
            #FIXME: This is for now to only for op scale which have a None use input "ScaleTensor" wo arguments.
            if op["inputs"][idx]["arguments"] == None or len(op["inputs"][idx]["arguments"]) == 0:
                continue
            it = op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"]
            in_type = op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_type"]
            # op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"] = input_shape #FIXME: TEMP
            input_dim_order = op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dim_order"]

        output_shape, out_dim_order = self.shape_infer(input_shape=it, input_dim_order=input_dim_order, attrs=attrs)

        for idx, output in enumerate(op['outputs']):
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_type"] = in_type
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"] = output_shape
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dim_order"] = out_dim_order

        return output_shape



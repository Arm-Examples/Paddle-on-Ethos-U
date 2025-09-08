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
class FakeMatmulV2Visitor(FakeNodesVisitor):

    target = "matmul_v2"

    def __init__(self):
        super().__init__()

    def shape_infer(self, **kwargs):
        input_shapes = kwargs.get('input_shape')
        return input_shapes
        if input_shapes:
            if len(input_shapes) == 1:
                output_shape = input_shapes[0]
            elif len(input_shapes) == 2:
                assert input_shapes[0][:-2] == input_shapes[1][:-2]
                try:
                    mn = input_shapes[0][-2:]
                    nm = input_shapes[1][-2:]
                    output_shape = (*(input_shapes[0][:-2]), mn[0], nm[1])
                except:
                    raise ValueError(f"Matmul shape error inputshape:{input_shapes}")
        else:
            raise ValueError(f"Missing necessary parameters {input_shapes}")
        return output_shape



@register_fake_nodes
class FakeMatmulVisitor(FakeNodesVisitor):

    target = "matmul"

    def __init__(self):
        super().__init__()

    def shape_infer(self, **kwargs):
        X = kwargs.get('x')
        Y = kwargs.get('y')
        if X and Y:
            assert X[-1] == Y[-2]
            try:
                mn = X[-2:]
                nm = Y[-2:]
                output_shape = (*(X[:-2]), mn[0], nm[1])
            except:
                raise ValueError(f"Matmul shape error inputshape:{X}")
        else:
            raise ValueError(f"Missing necessary parameters {X} {Y}")
        return output_shape


    def infer(self, **kwargs):
        op = kwargs.get('op')
        attrs = kwargs.get('attrs')

        X = [input for idx, input in enumerate(op["inputs"]) if input['parameter'] == "X" ]
        x_it = X[0]["arguments"][0]["type"]["dense_tensor"]["dt_dims"]
        it_type = X[0]["arguments"][0]["type"]["dense_tensor"]["dt_type"]
        input_dim_order = X[0]["arguments"][0]["type"]["dense_tensor"]["dt_dim_order"]

        Y = [input for idx, input in enumerate(op["inputs"]) if input['parameter'] == "Y" ]
        y_it = Y[0]["arguments"][0]["type"]["dense_tensor"]["dt_dims"]

        output_shape = self.shape_infer(x=x_it, y=y_it)

        for idx, output in enumerate(op['outputs']):
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_type"] = it_type
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"] = output_shape
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dim_order"] = input_dim_order

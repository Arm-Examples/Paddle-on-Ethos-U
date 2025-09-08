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
        X = kwargs.get('x')
        Y = kwargs.get('y')
        if X and Y:
            if len(X) != len(Y):
                Y.append(1)

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

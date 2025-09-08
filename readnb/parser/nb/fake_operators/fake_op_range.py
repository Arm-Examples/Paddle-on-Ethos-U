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
class FakeRangeVisitor(FakeNodesVisitor):
    target = "range"

    def __init__(self):
        super().__init__()

    def shape_infer(self, **kwargs):
        input_shape = kwargs.get('input_shape')
        return input_shape

    def infer(self, **kwargs):
        op = kwargs.get('op')
        for idx, output in enumerate(op['outputs']):
            dense_tensor = output["arguments"][0]["type"]["dense_tensor"]
            output_shape = dense_tensor["dt_dims"]

            if 'dt_dim_order' not in op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"].keys():
                input_dim_order = list(range(len(output_shape)))
                print(f"Warning: output {op['id']} has no dt_dim_order, use default order {input_dim_order}")
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dim_order"] = input_dim_order

        return output_shape

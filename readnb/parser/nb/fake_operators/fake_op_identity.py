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
class FakeFeedVisitor(FakeNodesVisitor):

    target = "feed"

    def __init__(self):
        super().__init__()

    def shape_infer(self, **kwargs):
        pass

@register_fake_nodes
class FakeFetchVisitor(FakeNodesVisitor):

    target = "fetch"

    def __init__(self):
        super().__init__()

    def shape_infer(self, **kwargs):
        pass


@register_fake_nodes
class FakeIdentityVisitor(FakeNodesVisitor):
    '''
        "softmax",
        "shuffle_channel",
        "calib",
        "scale",
        "sigmoid",
        "hard_sigmoid",
        "relu",
        "hard_swish",
        "nearest_interp_v2",
        "matmul_v2",
        "multiclass_nms3",
        "shuffle_channel",
        "sqrt",
        "arg_max",
        "bilinear_interp_v2",
        "fusion_elementwise_add_activation",
        "shape",
        "slice",
    '''
    target = "identity"

    def __init__(self):
        super().__init__()

    def shape_infer(self, **kwargs):
        input_shape = kwargs.get('input_shape')
        return input_shape

    def infer(self, **kwargs):
        op = kwargs.get('op')
        for idx, input in enumerate(op["inputs"]):
            #FIXME: This is for now to only for op scale which have a None use input "ScaleTensor" wo arguments.
            if op["inputs"][idx]["arguments"] == None or len(op["inputs"][idx]["arguments"]) == 0:
                continue
            it = op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"]
            in_type = op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_type"]
            # op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"] = input_shape #FIXME: TEMP
            input_dim_order = op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dim_order"]

        output_shape = self.shape_infer(input_shape=it)

        for idx, output in enumerate(op['outputs']):
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_type"] = in_type
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"] = output_shape
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dim_order"] = input_dim_order

        return output_shape

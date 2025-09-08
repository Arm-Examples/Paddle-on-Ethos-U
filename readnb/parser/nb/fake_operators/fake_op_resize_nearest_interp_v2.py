
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
class FakeResizeNearestInterPV2Visitor(FakeNodesVisitor):

    target = "nearest_interp_v2"

    def __init__(self):
        super().__init__()

    def shape_infer(self, **kwargs):
        input_shape = kwargs.get('input_shape')
        scales = kwargs.get('scales')

        N, C, H, W = input_shape
        if scales is not None and len(scales) == 2:
            output_h = int(H * scales[0])
            output_w = int(W * scales[1])
        else:
            raise ValueError(f"Missing necessary parameters {input_shape}")
        output_shape = [N, C, output_h, output_w]
        return output_shape

    def infer(self, **kwargs):
        op = kwargs.get('op')
        attr_dict = kwargs.get('attrs')

        assert(op["inputs"][0]['arguments'] != None)
        assert(len(op["inputs"][0]['arguments']) != 0)

        input_info = op["inputs"][0]['arguments'][0]
        output_info = op["outputs"][0]['arguments'][0]
        #FIXME: This is for now to only for op scale which have a None use input "ScaleTensor" wo arguments.
        input_dtype = input_info["type"]["dense_tensor"]["dt_type"]
        input_shape = input_info["type"]["dense_tensor"]["dt_dims"]
        input_shape_order = input_info["type"]["dense_tensor"]["dt_dim_order"]

        output_info["type"]["dense_tensor"]["dt_type"] = input_dtype
        output_info["type"]["dense_tensor"]["dt_dims"] = list(self.shape_infer(input_shape=input_shape, scales=attr_dict["scale"]))
        output_info["type"]["dense_tensor"]["dt_dim_order"] = copy.deepcopy(input_shape_order)

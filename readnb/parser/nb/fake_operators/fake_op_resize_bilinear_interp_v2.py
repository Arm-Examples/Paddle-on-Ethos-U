
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
class FakeResizeBilinearInterpV2Visitor(FakeNodesVisitor):

    target = "bilinear_interp_v2"

    def __init__(self):
        super().__init__()

    def shape_infer(self, **kwargs):
        input_shape = kwargs.get('input_shape')
        attrs = kwargs.get('attrs')
        scales = attrs.get('scales', None)

        N, C, H, W = input_shape
        if scales is not None and len(scales) == 2:
            output_h = int(H * scales[0])
            output_w = int(W * scales[1])
            output_shape = [N, C, output_h, output_w]
        elif "out_h" in attrs and "out_w" in attrs:
            out_d = attrs.get("out_d", -1)
            out_h = attrs.get("out_h", -1)
            out_w = attrs.get("out_w", -1)
            if out_w != -1 or out_h != -1:
                output_h = out_h
                output_w = out_w
                output_shape = [N, C, output_h, output_w]
            else:
               output_shape = input_shape
        else:
            output_shape = input_shape
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


        output_shape = self.shape_infer(input_shape=input_shape, attrs=attr_dict)

        output_info["type"]["dense_tensor"]["dt_type"] = input_dtype
        output_info["type"]["dense_tensor"]["dt_dims"] = list(output_shape)
        output_info["type"]["dense_tensor"]["dt_dim_order"] = copy.deepcopy(input_shape_order)

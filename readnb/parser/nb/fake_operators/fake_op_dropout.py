from typing import Dict, List
import numpy as np
from paddle.lite.fbs.proto.VarType_.Type import Type
from parser.nb.nb_utils import get_param, attr_to_dict, update_attr

# from parser.nb.fake_operators.fake_node_visitor import(
#     FakeNodesVisitor,
#     register_fake_nodes,
# )
from .fake_node_visitor import (
    FakeNodesVisitor,
    register_fake_nodes,
)
from .op_scale_scan import find_outscale_from_next

@register_fake_nodes
class FakeDropoutVisitor(FakeNodesVisitor):

    target = "dropout"

    def __init__(self):
        super().__init__()

    def shape_infer(self, **kwargs):
        input_shape = kwargs.get('input_shape')
        return input_shape

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

        output_shape = self.shape_infer(input_shape=it)

        self.__scale_infer(op, attrs)

        mask_id = [id for id, data in enumerate(op["outputs"]) if data['parameter'] == "Mask"][0]
        op['outputs'].pop(mask_id)
        for idx, output in enumerate(op['outputs']):
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_type"] = in_type
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"] = output_shape
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dim_order"] = input_dim_order
        return output_shape

    def __type_infer(self, type):
        return type


    def __scale_infer(self, op, op_attrs):
        nexts = op['next_op']
        alias = op_attrs['alias']
        if alias == 'int8_out':
            pass
        elif alias == 'def':
            out_scale = op_attrs['out_threshold'] / 127
            for next_id, next in enumerate(op['next_op']):
                update_attr(next['attrs'], "Input0_scale", [out_scale])
        else:
            raise ValueError(f"Not support other alias type.")

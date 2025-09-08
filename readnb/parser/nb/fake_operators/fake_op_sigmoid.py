from typing import Dict, List
from parser.nb.nb_utils import attr_to_dict

# from parser.nb.fake_operators.fake_node_visitor import(
#     FakeNodesVisitor,
#     register_fake_nodes,
# )
from .fake_node_visitor import (
    FakeNodesVisitor,
    register_fake_nodes,
)

# Eelement-wises
@register_fake_nodes
class FakesigmoidVisitor(FakeNodesVisitor):

    target = "sigmoid"

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

        for idx, output in enumerate(op['outputs']):
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_type"] = in_type
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"] = output_shape
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dim_order"] = input_dim_order

        return output_shape

    def __scale_infer(self, op, op_attrs):
        alias = op_attrs['alias']
        input_scale = op_attrs.get('Input0_scale', [-1])
        if alias == "def" and "Input0_scale" in op_attrs.keys():
            output_name = op['outputs'][0]['arguments'][0]['name']
            next_op = op['next_op']

            if len(next_op) == 1:
                next_op = next_op[0]
                next_op_input_para = ""
                for idx, input in enumerate(next_op['inputs']):
                    for arug in input['arguments']:
                        if output_name == arug['name']:
                            next_op_input_para = input['parameter']
                            scale_idx = idx

                next_op_attr_dict = attr_to_dict(next_op['attrs'])
                scale_p = []
                for key, val in next_op_attr_dict.items():
                    if "_scale" in key:
                        scale_p.append(val)
                if len(scale_p) == 0:
                    scale_p.append(1.0)
                    scale_idx = 0
                    print(f"scale_idx {scale_idx} is not found in next op {next_op['id']}, use default scale 1.0")
                op['attrs'].append({"name": "Output0_scale", "val": scale_p[scale_idx]})
                print(next_op_input_para, scale_p[scale_idx])
            else:
                raise ValueError("Sigmoid not support scale caculate when nextop size > 1")

        else:
            # do nothing.
            pass

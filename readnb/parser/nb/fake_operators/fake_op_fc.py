from typing import Dict, List
import numpy as np
from parser.nb.nb_utils import get_param, update_attr
from paddle.lite.fbs.proto.VarType_.Type import Type
# from parser.nb.fake_operators.fake_node_visitor import(
#     FakeNodesVisitor,
#     register_fake_nodes,
# )
from .fake_node_visitor import (
    FakeNodesVisitor,
    register_fake_nodes,
)

@register_fake_nodes
class FakeFullconnectVisitor(FakeNodesVisitor):

    target = "fc"

    def __init__(self):
        super().__init__()

    def shape_infer(self, **kwargs):
        input_shape = kwargs.get('input_shape')
        weight_shape = kwargs.get('weight_shape')
        if input_shape and weight_shape:
            B, C_i = input_shape[:2]

            C_i, C_o = weight_shape
            output_shape = (B, C_o)
        else:
            raise ValueError(f"Missing necessary parameters {input_shape} {weight_shape}")
        return output_shape

    def infer(self, **kwargs):
        op = kwargs.get('op')
        attrs = kwargs.get('attrs')
        params = kwargs.get('params')

        for idx, input in enumerate(op["inputs"]):
            if input["parameter"] == "Input":
                in_type = op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_type"]
                it = op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"]

            elif input["parameter"] == "W":
                weight_shape = op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"]

        output_shape = self.shape_infer(
            input_shape=it,
            weight_shape=weight_shape)

        out_type = self.__type_infer(in_type)

        self.__scale_infer(op, attrs)

        for idx, output in enumerate(op['outputs']):
            if output["parameter"] == "Out":
                op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_type"] = out_type
                op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"] = output_shape
                op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dim_order"] = [0, 1]

        self.__refill_fp32_fc(op, attrs, params)
        return output_shape

    def __type_infer(self, type):
        return type


    def __refill_fp32_fc(self, op, attrs, params):
        attr_dict = attrs
        for input in op['inputs']:
            if input['parameter'] == "W" and (input['arguments'] != None and len(input['arguments']) > 0):
                argu_name = input['arguments'][0]['name']
                weight_param = get_param(params, argu_name)
                weight_dtype = weight_param.tensor['data_type']
                weight = weight_param.tensor['data']

                if weight_dtype == Type().FP32:
                    # quantize fp32 weight to int8
                    weight_tensor = np.array(weight).reshape(tuple(weight_param.tensor["dims"]))

                    max_abs_values = np.max(np.abs(weight_tensor), axis=0)
                    max_abs_values[max_abs_values == 0] = 1e-8  # prevent to divide by zero
                    w_scales = max_abs_values / 127

                    weight_int8 = np.round(weight_tensor / w_scales).astype(np.int8)
                    weight_param.tensor['data'] = weight_int8.flatten().tolist()
                    weight_param.tensor['data_type'] = Type().INT8
                    op["attrs"].append({"name": "W0_scale", "val": w_scales.flatten().tolist()})


    def __scale_infer(self, op, op_attrs):
        nexts = op['next_op']
        alias = op_attrs['alias']
        if alias == 'int8_out' or alias == "fp32out" or alias == "fp32_out":
            pass
        elif alias == 'def':
            out_scale = op_attrs['out_threshold'] / 127
            update_attr(op['attrs'], "Output0_scale", [out_scale])
        else:
            raise ValueError(f"Not support other alias type.")
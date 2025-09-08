from typing import Dict, List
import numpy as np
from parser.nb.nb_utils import update_attr
# from parser.nb.fake_operators.fake_node_visitor import(
#     FakeNodesVisitor,
#     register_fake_nodes,
# )
from .fake_node_visitor import (
    FakeNodesVisitor,
    register_fake_nodes,
)

@register_fake_nodes
class FakePool2dVisitor(FakeNodesVisitor):

    target = "pool2d"

    def __init__(self):
        super().__init__()

    def shape_infer(self, **kwargs):
        input_shape = kwargs.get('input_shape')
        K_h, K_w = kwargs.get('kernel_size', (1, 1))
        S_h, S_w = kwargs.get('strides', (1, 1))
        P_h, P_w = kwargs.get('padding', (0, 0))

        # print(f"Input Shape: {input_shape} , Filter Shape: {filter_shape}, Strides: {S_h} {S_w}, Padding: {P_h} {P_w}")
        if input_shape:
            B, C_i, I_h, I_w = input_shape

            # we assume input shape is NCHW and only.
            O_h = max( ((I_h - K_h + 2 * P_h) // S_h + 1), 1)
            O_w = max( ((I_w - K_w + 2 * P_w) // S_w + 1), 1)
            output_shape = (B, C_i, O_h, O_w)
        else:
            raise ValueError(f"Missing necessary parameters {input_shape}")
        return output_shape

    def infer(self, **kwargs):
        op = kwargs.get('op')
        attr_dict = kwargs.get('attrs')

        # for attr in op["attrs"]:
        #     attr_dict[attr["name"]] = attr["val"]
        if "pooling_type" in attr_dict.keys():
            if attr_dict["pooling_type"] == "avg":
                op['type'] = "avg_pool2d"
            elif attr_dict["pooling_type"] == "max":
                op['type'] = "max_pool2d"
            else:
                raise RuntimeError(f"Only Support Max/Avg Pooling {op['type']} {attr_dict['pooling_type']}")
        if "data_format" in attr_dict.keys():
            if attr_dict['data_format'] == "NCHW":
                dim_order = [0,2,3,1]
            elif attr_dict['data_format'] == "NHWC":
                dim_order = [0,1,2,3]
            else:
                raise ValueError(f"OP: {op['tytpe']} is not NCHW of NHWC instead of {attr_dict['data_format']}")

        # shape_infer
        # rewrite op io vars
        for idx, input in enumerate(op["inputs"]):
            # transpose NCHW to NHWC
            op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dim_order"] = dim_order
            if op["inputs"][idx]["parameter"] == "X":
                it = op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"]
                it_type = op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_type"]
                # op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"] = input_shape #FIXME: TEMP

        if True == attr_dict['global_pooling']:
            kernel_size = (it[2], it[3])
        else:
            kernel_size = attr_dict['ksize']

        update_attr(op['attrs'], 'ksize', np.array(kernel_size).astype(np.int32).tolist())

        output_shape = self.shape_infer(
            input_shape=it,
            kernel_size=kernel_size,
            strides=attr_dict['strides'],
            padding=attr_dict['paddings'])

        for idx, output in enumerate(op["outputs"]):
            argument = output["arguments"][0]
            # transpose NCHW to NHWC
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dim_order"] = dim_order
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_type"] = it_type
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"] = output_shape

        return output_shape

# @register_fake_nodes
# class FakeMaxpool2dVisitor(FakeNodesVisitor):

#     target = "max_pool2d"

#     def __init__(self):
#         super().__init__()

#     def shape_infer(self, **kwargs):
#         input_shape = kwargs.get('input_shape')
#         K_h, K_w = kwargs.get('kernel_size', (1, 1))
#         S_h, S_w = kwargs.get('strides', (1, 1))
#         P_h, P_w = kwargs.get('padding', (0, 0))

#         # print(f"Input Shape: {input_shape} , Filter Shape: {filter_shape}, Strides: {S_h} {S_w}, Padding: {P_h} {P_w}")
#         if input_shape:
#             B, C_i, I_h, I_w = input_shape

#             # we assume input shape is NCHW and only.
#             O_h = (I_h - K_h + 2 * P_h) // S_h + 1
#             O_w = (I_w - K_w + 2 * P_w) // S_w + 1
#             output_shape = (B, C_i, O_h, O_w)
#         else:
#             raise ValueError(f"Missing necessary parameters {input_shape}")
#         return output_shape

#     def get_dim_order(self, type):
#         if type == "NCHW":
#             return (0, 2, 3, 1)
#         elif type == "NHWC":
#             return (0, 1, 2, 3)
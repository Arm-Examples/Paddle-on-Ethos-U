from typing import Dict, List
import numpy as np

from paddle.lite.fbs.proto.VarType_.Type import Type
from parser.nb.nb_utils import (
    get_param,
    attr_to_dict,
    update_attr,
    find_my_prev_op
)
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
class FakeConv2dTransposeVisitor(FakeNodesVisitor):

    target = "conv2d_transpose"

    def __init__(self):
        super().__init__()

    def shape_infer(self, **kwargs):
        input_shape = kwargs.get('input_shape')
        filter_shape = kwargs.get('filter_shape')
        S_h, S_w = kwargs.get('strides', (1, 1))
        P_h, P_w = kwargs.get('padding', (0, 0))
        D_h, D_w = kwargs.get('dilation', (1, 1))

        if input_shape and filter_shape:
            B, C_i, I_h, I_w = input_shape
            C_i, C_o, K_h, K_w = filter_shape
            O_h = (I_h-1)*S_h - 2*P_h + D_h * (K_h-1) + 1
            O_w = (I_w-1)*S_w - 2*P_w + D_w * (K_w-1) + 1

            print("!!", B, C_o, O_h, O_w, S_h, S_w, P_h, P_w, D_h, D_w)
            O_h = list(range(O_h, O_h+S_h))[:-(S_h-1)]
            O_w = list(range(O_w, O_w+S_w))[:-(S_w-1)]

            if len(O_h) > 1 or len(O_h) > 1:
                raise ValueError(f"Conv2d out shape more than one size h:{O_h} w:{O_w}")
            
            print("!!", B, C_o, O_h, O_w)
            output_shape = (B, C_o, O_h[0], O_w[0])
            return output_shape


    def infer(self, **kwargs):
        op = kwargs.get('op')
        attr_dict = kwargs.get('attrs')
        params = kwargs.get('params')

        # get data_format
        if "data_format" in attr_dict.keys():
            if attr_dict['data_format'] == "NCHW":
                dim_order = [0,2,3,1]
            elif attr_dict['data_format'] == "NHWC":
                dim_order = [0,1,2,3]
            else:
                raise ValueError(f"OP: {op['type']} is not NCHW of NHWC instead of {attr_dict['data_format']}")

        # rewrite op io vars
        for idx, input in enumerate(op["inputs"]):
            if input["parameter"] == "Bias" \
                or (op["type"] == "depthwise_conv2d" and input["parameter"] == "Filter"):
                # bias doesn't need to be transposed
                pass
            else:
                # rewrite dim_order based on data_format
                op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dim_order"] = dim_order

            if input["parameter"] == "Input":
                it = op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"]
                it_type = op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_type"]
                # op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"] = input_shape #FIXME: TEMP
            elif input["parameter"] == "Filter":
                filter_shape = op["inputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"]

        output_shape = self.shape_infer(
            input_shape=it,
            filter_shape=filter_shape,
            strides=attr_dict['strides'],
            padding=attr_dict['paddings'],
            dilation=attr_dict['dilations']
            )

        out_type = self.__type_infer(it_type, attr_dict)

        self.__scale_infer(op, attr_dict)

        for idx, output in enumerate(op["outputs"]):
            argument = output["arguments"][0]
            # rewrite dim_order based on data_format
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dim_order"] = dim_order
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_dims"] = output_shape
            op["outputs"][idx]["arguments"][0]["type"]["dense_tensor"]["dt_type"] = out_type


        self.__refill_bias(op, attr_dict, params)
        return output_shape

    def __type_infer(self, in_type, attrs):
        out_type = in_type
        return out_type

    def __refill_bias(self, op, attrs, params):
        attr_dict = attrs
        for input in op['inputs']:
            if input['arguments'] != None and len(input['arguments']) > 0:
                if input['parameter'] == "Bias" :
                    argu_name = input['arguments'][0]['name']
                    # bias_param = [p for p in params if p.tensor['name'] == argu_name][0]
                    bias_param = get_param(params, argu_name)
                    bias = bias_param.tensor['data']
                    if f"{argu_name}_quant_scale" in attr_dict.keys() or \
                        (f"{argu_name}_quant_scale" in attr_dict.keys() and "bias_after_scale" in attr_dict.keys()):

                        if f"{argu_name}_quant_scale" in attr_dict.keys():
                            bias = (np.array(bias) * attr_dict[f"{argu_name}_quant_scale"])
                        # rescale FP32 Bias to Int8
                        if "Input0_scale" in attr_dict.keys() and "Filter0_scale" in attr_dict.keys():
                            bias = np.round(np.array(bias) / (np.array(attr_dict["Input0_scale"]) * np.array(attr_dict["Filter0_scale"])))
                        bias_param.tensor['data'] = bias.astype(np.int32).tolist()
                        bias_param.tensor['data_type'] = Type().INT32
                    else:
                        bias_param.tensor['data'] = bias
                        bias_param.tensor['data_type'] = Type().FP32

    def __scale_infer(self, op, op_attrs):
        nexts = op['next_op']
        alias = op_attrs['alias']
        if alias == 'int8_out':
            pass
        elif alias == "def" or alias == 'fp32_out':
            print("need do add op operator.")
            out_scales = []
            for next_id, next in enumerate(op['next_op']):
                # out_scales.append( self.__find_outscale(op, op_attrs, next_id, next))
                out_scales.append(find_outscale_from_next(op, op_attrs, next_id, next))

            update_attr(op['attrs'], "Output0_scale", list(set(out_scales)))

            if "Input0_scale" not in op_attrs.keys():
                my_name = [op_in for op_in in op['inputs'] if op_in['parameter'] == "Input"][0]['arguments'][0]['name']

                prev_op = find_my_prev_op(op, my_name)
                prev_op_attrs = attr_to_dict(prev_op["attrs"])
                if prev_op_attrs['alias'] == "def" or prev_op_attrs['alias'] == "fp32_out" or prev_op_attrs['alias'] == "fp32out":
                    if "Output0_scale" in prev_op_attrs.keys():
                        update_attr(op['attrs'], "Input0_scale", prev_op_attrs['Output0_scale'])
                    elif "scale" in prev_op_attrs.keys():
                        update_attr(op['attrs'], "Input0_scale", prev_op_attrs['scale'])
                    else:
                        raise ValueError(f"OP:{op['id']} Not support Cases of {op['type']} in scale_infer")
        else:
            raise ValueError(f"Not support other alias type.")


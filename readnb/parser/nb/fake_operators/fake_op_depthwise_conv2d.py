from typing import Dict, List
import numpy as np
import copy

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
class FakeDepthwiseConv2dVisitor(FakeNodesVisitor):

    target = "depthwise_conv2d"

    def __init__(self):
        super().__init__()

    def shape_infer(self, **kwargs):
        input_shape = kwargs.get('input_shape')
        filter_shape = kwargs.get('filter_shape')
        S_h, S_w = kwargs.get('strides', (1, 1))
        P_h, P_w = kwargs.get('padding', (0, 0))
        D_h, D_w = kwargs.get('dilation', (1, 1))
        # print(f"Input Shape: {input_shape} , Filter Shape: {filter_shape}, Strides: {S_h} {S_w}, Padding: {P_h} {P_w}, Dilation: {D_h} {D_w}")

        if input_shape and filter_shape:
            B, C_i, I_h, I_w = input_shape
            C_o, M, K_h, K_w = filter_shape
            K_h = K_h + (K_h - 1) * (D_h - 1)
            K_w = K_w + (K_w - 1) * (D_w - 1)

            # we assume input shape is NCHW and only. which for filter shape is Co,Ci,H,W
            O_h = (I_h - K_h + 2 * P_h) // S_h + 1
            O_w = (I_w - K_w + 2 * P_w) // S_w + 1
            output_shape = (B, C_o, O_h, O_w)
            return output_shape
        else:
            raise ValueError(f"Missing necessary parameters {input_shape} or {filter_shape}")


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

    def expand(self, **kwargs):
        op_gram = kwargs.get('gram')
        op = kwargs.get('op')
        dag = op_gram.dag

        op_attrs = attr_to_dict(op['attrs'])
        scales = op_attrs['Output0_scale']
        if len(scales) > 1:
            # This is for mulit-output-path from depthwise_conv2d, each path have own scale to the next op.
            # Thus, we need to make a mirror of dpw_conv2d and sperate output to each path.
            # Following is make a new dpwconv2d and connect from orig-dpwconv2d input to one of the pathes.
            for idx, scale in enumerate(scales[1:]):
                o_id = idx + 1

                # Create a new op and an output tensor.
                new_dpconv2d = copy.deepcopy(op)
                # use the referances of orig-op
                new_dpconv2d['prev_op'] = op['prev_op']
                new_dpconv2d['next_op'] = [op['next_op'][o_id]]

                update_attr(new_dpconv2d['attrs'], 'Output0_scale', [op_attrs['Output0_scale'][o_id]])

                new_dpconv2d['id'] = op['id'] + o_id

                new_output = copy.deepcopy(new_dpconv2d['outputs'][0])
                new_output['arguments'][0]['name'] = f"{new_output['arguments'][0]['name']}_o_{o_id}"

                # connect new output tensor to next op input.
                for next_input in op['next_op'][o_id]['inputs']:
                    for idx, argu in enumerate(next_input['arguments']):
                        if argu == new_dpconv2d['outputs'][0]['arguments'][0]:
                            next_input['arguments'][idx] = new_output['arguments'][0]

                for next_prev_op in op['next_op'][o_id]['prev_op']:
                    if next_prev_op['id'] == op['id']:
                        next_prev_op = new_dpconv2d

                # connect my new op output to new tensor
                new_dpconv2d['outputs'][0] = new_output

                # add op to graph. the new ops are Batch-First,
                # E.g. Orig Id is 10 which new ids are 11, 12, 13 ...
                # insert to 10-(new_id)-11
                dag.add_operation(new_dpconv2d, op['id']+o_id)
            op['next_op'] = [op['next_op'][0]]
            update_attr(op['attrs'], 'Output0_scale', [op_attrs['Output0_scale'][0]])


    def __type_infer(self, in_type, attrs):
        out_type = in_type
        return out_type

    def __find_outscale(self, op, op_attr, next_id, next_op):

        jump_op_type = [
            'pool2d'
        ]

        calcu_scale_op_type = [
            'sigmoid',
        ]

        calcu_scale_next_op_type = [
            'dropout'
        ]
        next_attrs = attr_to_dict(next_op["attrs"])

        if next_op['type'] in jump_op_type:
            if next_op['next_op'] != [] and next_op['next_op'][0]['type'] == 'calib':
                calib_op = next_op['next_op'][0]
                attr_dict = attr_to_dict(calib_op["attrs"])
                return attr_dict['scale']
            else:
                raise ValueError(f"OP:{op['id']} Not support Cases of DWConv2d in jump_next")

        else:
            if next_op['type'] in calcu_scale_op_type:
                scale = op_attr['out_threshold'] / 127
                #set scale to next sigmoid
                next_op['attrs'].append({"name": "Input0_scale", "val": [scale]})
                return scale
            elif next_op['type'] in calcu_scale_next_op_type:
                next_attrs = attr_to_dict(next_op['attrs'])
                scale = next_attrs['out_threshold'] / 127
                return scale
            elif "Input0_scale" not in next_attrs.keys() and any('_scale' in key for key in next_attrs.keys()):
                output_name = op["outputs"][0]['arguments'][0]['name']

                next_op_input_para = ""
                for idx, input in enumerate(next_op['inputs']):
                    for arug in input['arguments']:
                        if output_name == arug['name']:
                            next_op_input_para = input['parameter']
                            scale_idx = idx
                scale_p = []
                for key, val in next_attrs.items():
                    if "_scale" in key:
                        scale_p.append(*val)

                return scale_p[scale_idx]
            elif "Input0_scale" in next_attrs.keys():
                next_inscale = next_attrs.get("Input0_scale", -1)
                return next_inscale

            elif "scale" in next_attrs.keys():
                next_inscale = next_attrs.get("scale", -1)
                return next_inscale

            else:
                raise ValueError(f"OP:{op['id']} Not support Cases of DWConv2d not in jump_next")


    def __scale_infer(self, op, op_attrs):
        nexts = op['next_op']
        alias = op_attrs['alias']
        if alias == 'int8_out':
            pass
        elif alias == 'fp32_out':
            out_scales = []
            for next_id, next in enumerate(op['next_op']):
                # out_scales.append( self.__find_outscale(op, op_attrs, next_id, next))
                out_scales.append(find_outscale_from_next(op, op_attrs, next_id, next))
            update_attr(op['attrs'], "Output0_scale", list(set(out_scales)))
            # op['attrs'].append({"name": "Output0_scale", "val": list(set(out_scales)) })
        else:
            raise ValueError(f"Not support other alias type.")


    def __refill_bias(self, op, attrs, params):
        attr_dict = attrs
        for input in op['inputs']:
            if input['parameter'] == "Bias":
                argu_name = input['arguments'][0]['name']
                # bias_param = [p for p in params if p.tensor['name'] == argu_name][0]
                bias_param = get_param(params, argu_name)
                if False == bias_param.tensor.get('refill', False):
                    bias = bias_param.tensor['data']
                    if "Input0_scale" in attr_dict.keys() and "Filter0_scale" in attr_dict.keys():
                        bias = np.round(np.array(bias) / (np.array(attr_dict["Input0_scale"]) * np.array(attr_dict["Filter0_scale"])))
                    bias_param.tensor['data'] = bias.astype(np.int32).tolist()
                    bias_param.tensor['data_type'] = Type().INT32
                    bias_param.tensor['refill'] = True
            elif input['parameter'] == "Filter":
                argu_name = input['arguments'][0]['name']
                # filter_param = [p for p in params if p.tensor['name'] == argu_name][0]
                filter_param = get_param(params, argu_name)
                if False == filter_param.tensor.get('refill', False):
                    filter = np.array(filter_param.tensor['data']).reshape(filter_param.tensor['dims'])
                    # FIXME: this is a workaround pass,
                    #  which depthwiseconv2d in op have an reshape which not transpose.
                    #  we do it here but not to change the reshape.
                    new_shape_order = (2, 3, 0, 1)
                    filter_param.tensor['data'] = np.transpose(filter, new_shape_order).flatten().tolist()
                    filter_param.tensor['refill'] = True
